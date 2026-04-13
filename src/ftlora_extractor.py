from __future__ import annotations

import copy
import math
import random
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


class OpinionExtractor:
    method: Literal["NOFT", "FT"] = "FT"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.accelerator = Accelerator()

        self.aspects = ["Price", "Food", "Service"]
        self.num_classes = 4

        self.label_to_id = {
            "Positive": 0,
            "Negative": 1,
            "Mixed": 2,
            "No Opinion": 3,
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        # Authorized encoder-only models from the assignment.
        # Best default for French restaurant reviews.
        self.candidate_model_ids = [
            "almanach/moderncamembert-base",
            "FacebookAI/xlm-roberta-base",
            "flaubert/flaubert_base_cased",
            "flaubert/flaubert_base_uncased",
            "flaubert/flaubert_small_cased",
            "jhu-clsp/mmBERT-base",
            "jhu-clsp/mmBERT-small",
        ]

        self.model_id = None
        self.tokenizer = None
        self.model = None

        self.max_length = 256
        self.label_smoothing = 0.04
        self.aux_loss_weight = 0.20

        self._set_seed(42)

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def train(self, train_data: list[dict], val_data: list[dict]) -> None:
        self._set_seed(42 + self.accelerator.process_index)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model_id = self._pick_model_id()
        if self.accelerator.is_main_process:
            print(f"Using encoder model: {self.model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.model = MultiAspectOpinionModel(self.model_id, num_classes=self.num_classes)

        train_rows = self._prepare_rows(train_data)
        val_rows = self._prepare_rows(val_data)

        train_dataset = ReviewDataset(train_rows, self.tokenizer, self.max_length)
        val_dataset = ReviewDataset(val_rows, self.tokenizer, self.max_length)

        class_weights, opinion_pos_weight = self._compute_loss_weights(train_rows)

        num_devices = max(1, self.accelerator.num_processes)
        per_device_batch_size = 8
        target_effective_batch_size = 48
        grad_accum_steps = max(
            1,
            math.ceil(target_effective_batch_size / (per_device_batch_size * num_devices)),
        )
        effective_batch_size = per_device_batch_size * num_devices * grad_accum_steps

        base_lr = 2e-5
        learning_rate = base_lr * math.sqrt(effective_batch_size / target_effective_batch_size)
        learning_rate = float(min(3e-5, max(1.5e-5, learning_rate)))

        train_loader = DataLoader(
            train_dataset,
            batch_size=per_device_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=train_dataset.collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=per_device_batch_size * 2,
            shuffle=False,
            drop_last=False,
            collate_fn=val_dataset.collate_fn,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        num_epochs = 8
        steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum_steps))
        total_train_steps = max(1, num_epochs * steps_per_epoch)
        warmup_steps = max(1, int(0.1 * total_train_steps))

        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps,
        )

        (
            self.model,
            optimizer,
            train_loader,
            val_loader,
            scheduler,
        ) = self.accelerator.prepare(
            self.model,
            optimizer,
            train_loader,
            val_loader,
            scheduler,
        )

        class_weights = class_weights.to(self.accelerator.device)
        opinion_pos_weight = opinion_pos_weight.to(self.accelerator.device)

        best_state = None
        best_metric = -1.0
        patience = 2
        patience_counter = 0

        if self.accelerator.is_main_process:
            print(
                f"per_device_batch_size={per_device_batch_size}, "
                f"grad_accum_steps={grad_accum_steps}, "
                f"effective_batch_size={effective_batch_size}, "
                f"lr={learning_rate:.2e}"
            )

        for epoch in range(num_epochs):
            self.model.train()
            optimizer.zero_grad()

            running_loss = 0.0
            num_batches = 0

            for step, batch in enumerate(train_loader):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

                loss = self._compute_total_loss(
                    outputs=outputs,
                    labels=batch["labels"],
                    opinion_targets=batch["opinion_targets"],
                    class_weights=class_weights,
                    opinion_pos_weight=opinion_pos_weight,
                )
                loss = loss / grad_accum_steps
                self.accelerator.backward(loss)

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                running_loss += float(loss.detach().item())
                num_batches += 1

            val_macro_acc, per_aspect = self._evaluate(val_loader)

            if self.accelerator.is_main_process:
                avg_train_loss = running_loss / max(1, num_batches)
                print(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"train_loss={avg_train_loss:.4f} | "
                    f"val_macro_acc={val_macro_acc:.2f} | "
                    f"Price={per_aspect['Price']:.2f} "
                    f"Food={per_aspect['Food']:.2f} "
                    f"Service={per_aspect['Service']:.2f}"
                )

            if val_macro_acc > best_metric:
                best_metric = val_macro_acc
                patience_counter = 0
                best_state = copy.deepcopy(self.accelerator.unwrap_model(self.model).state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if self.accelerator.is_main_process:
                        print("Early stopping triggered.")
                    break

        if best_state is not None:
            self.accelerator.unwrap_model(self.model).load_state_dict(best_state)

        self.model.eval()
        self.accelerator.wait_for_everyone()

    def predict(self, texts: list[str]) -> list[dict]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("The model is not trained yet. Call train() first.")

        rows = []
        for text in texts:
            rows.append(
                {
                    "Restaurant": "",
                    "Review": text,
                    "Price": "No Opinion",
                    "Food": "No Opinion",
                    "Service": "No Opinion",
                }
            )

        dataset = ReviewDataset(rows, self.tokenizer, self.max_length)
        loader = DataLoader(
            dataset,
            batch_size=max(8, int(getattr(self.cfg, "eval_batch_size", 10))),
            shuffle=False,
            drop_last=False,
            collate_fn=dataset.collate_fn,
        )

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                preds = self._decode_predictions(outputs)
                all_preds.extend(preds)

        return all_preds

    # ---------------------------------------------------------
    # Data preparation
    # ---------------------------------------------------------
    def _pick_model_id(self) -> str:
        last_error = None
        for model_id in self.candidate_model_ids:
            try:
                _ = AutoTokenizer.from_pretrained(model_id, use_fast=True)
                return model_id
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Could not load any authorized encoder model. Last error: {last_error}")

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _normalize_label(self, value: str) -> str:
        value = str(value).strip()

        if "#" in value:
            value = value.split("#", 1)[0].strip()

        value = value.replace("_", " ")
        value = " ".join(value.split())

        if value in self.label_to_id:
            return value

        lower = value.lower()
        mapping = {
            "positive": "Positive",
            "negative": "Negative",
            "mixed": "Mixed",
            "no opinion": "No Opinion",
            "noopinion": "No Opinion",
            "no-opinion": "No Opinion",
            "none": "No Opinion",
            "neutral": "No Opinion",
            "na": "No Opinion",
            "n/a": "No Opinion",
        }
        if lower in mapping:
            return mapping[lower]

        raise ValueError(f"Unknown label: {value}")

    def _clean_text(self, text: str) -> str:
        text = str(text).strip()
        text = " ".join(text.split())
        return text

    def _build_input_text(self, restaurant: str, review: str) -> str:
        restaurant = self._clean_text(restaurant)
        review = self._clean_text(review)
        if restaurant:
            return f"Restaurant: {restaurant} [SEP] Review: {review}"
        return f"Review: {review}"

    def _prepare_rows(self, rows: list[dict]) -> list[dict]:
        prepared = []
        skipped = 0

        for row in rows:
            try:
                restaurant = row.get("Restaurant", row.get("Name", ""))
                review = row["Review"]

                price = self._normalize_label(row["Price"])
                food = self._normalize_label(row["Food"])
                service = self._normalize_label(row["Service"])

                prepared.append(
                    {
                        "Restaurant": restaurant,
                        "Review": review,
                        "Price": price,
                        "Food": food,
                        "Service": service,
                    }
                )
            except Exception:
                skipped += 1

        if self.accelerator.is_main_process and skipped > 0:
            print(f"Skipped {skipped} noisy rows during preparation.")

        return prepared

    def _compute_loss_weights(self, train_rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        # class_weights shape: [3, 4]
        # opinion_pos_weight shape: [3]
        class_weights = []
        opinion_pos_weight = []

        for aspect in self.aspects:
            counts = np.zeros(self.num_classes, dtype=np.float32)
            opinion_pos = 0.0
            opinion_neg = 0.0

            for row in train_rows:
                y = self.label_to_id[row[aspect]]
                counts[y] += 1.0

                has_opinion = 0 if row[aspect] == "No Opinion" else 1
                if has_opinion == 1:
                    opinion_pos += 1.0
                else:
                    opinion_neg += 1.0

            # Inverse-frequency style weights, normalized around 1
            weights = counts.sum() / np.maximum(counts, 1.0)
            weights = weights / weights.mean()
            class_weights.append(weights)

            pos_w = (opinion_neg + 1.0) / (opinion_pos + 1.0)
            opinion_pos_weight.append(pos_w)

        class_weights = torch.tensor(np.stack(class_weights, axis=0), dtype=torch.float32)
        opinion_pos_weight = torch.tensor(opinion_pos_weight, dtype=torch.float32)
        return class_weights, opinion_pos_weight

    # ---------------------------------------------------------
    # Loss / evaluation / decoding
    # ---------------------------------------------------------
    def _compute_total_loss(
        self,
        outputs: dict,
        labels: torch.Tensor,
        opinion_targets: torch.Tensor,
        class_weights: torch.Tensor,
        opinion_pos_weight: torch.Tensor,
    ) -> torch.Tensor:
        logits = outputs["logits"]  # [B, 3, 4]
        opinion_logits = outputs["opinion_logits"]  # [B, 3]

        total_ce = 0.0
        for aspect_idx in range(len(self.aspects)):
            ce = cross_entropy_with_label_smoothing(
                logits=logits[:, aspect_idx, :],
                targets=labels[:, aspect_idx],
                weight=class_weights[aspect_idx],
                label_smoothing=self.label_smoothing,
            )
            total_ce = total_ce + ce

        total_ce = total_ce / len(self.aspects)

        opinion_loss = 0.0
        for aspect_idx in range(len(self.aspects)):
            bce = nn.BCEWithLogitsLoss(pos_weight=opinion_pos_weight[aspect_idx])
            opinion_loss = opinion_loss + bce(
                opinion_logits[:, aspect_idx],
                opinion_targets[:, aspect_idx].float(),
            )
        opinion_loss = opinion_loss / len(self.aspects)

        return total_ce + self.aux_loss_weight * opinion_loss

    def _decode_predictions(self, outputs: dict) -> list[dict]:
        logits = outputs["logits"]  # [B, 3, 4]
        preds = torch.argmax(logits, dim=-1)  # [B, 3]
        preds = self.accelerator.gather_for_metrics(preds).detach().cpu().numpy()

        decoded = []
        for row in preds:
            decoded.append(
                {
                    "Price": self.id_to_label[int(row[0])],
                    "Food": self.id_to_label[int(row[1])],
                    "Service": self.id_to_label[int(row[2])],
                }
            )
        return decoded

    def _evaluate(self, val_loader: DataLoader) -> tuple[float, dict[str, float]]:
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                pred_ids = torch.argmax(outputs["logits"], dim=-1)  # [B, 3]

                gathered_preds = self.accelerator.gather_for_metrics(pred_ids)
                gathered_labels = self.accelerator.gather_for_metrics(batch["labels"])

                all_preds.append(gathered_preds.detach().cpu())
                all_labels.append(gathered_labels.detach().cpu())

        preds = torch.cat(all_preds, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()

        per_aspect = {}
        accs = []
        for j, aspect in enumerate(self.aspects):
            acc = 100.0 * float(np.mean(preds[:, j] == labels[:, j]))
            per_aspect[aspect] = acc
            accs.append(acc)

        macro_acc = float(np.mean(accs))
        return macro_acc, per_aspect


class ReviewDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.label_to_id = {
            "Positive": 0,
            "Negative": 1,
            "Mixed": 2,
            "No Opinion": 3,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]

        restaurant = str(row.get("Restaurant", "")).strip()
        review = str(row["Review"]).strip()
        text = f"Restaurant: {restaurant} [SEP] Review: {review}" if restaurant else f"Review: {review}"

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        labels = torch.tensor(
            [
                self.label_to_id[row["Price"]],
                self.label_to_id[row["Food"]],
                self.label_to_id[row["Service"]],
            ],
            dtype=torch.long,
        )

        opinion_targets = torch.tensor(
            [
                0 if row["Price"] == "No Opinion" else 1,
                0 if row["Food"] == "No Opinion" else 1,
                0 if row["Service"] == "No Opinion" else 1,
            ],
            dtype=torch.float32,
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "opinion_targets": opinion_targets,
        }

    def collate_fn(self, batch: list[dict]) -> dict:
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        attention_mask = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]
        labels = torch.stack([x["labels"] for x in batch], dim=0)
        opinion_targets = torch.stack([x["opinion_targets"] for x in batch], dim=0)

        padded = self.tokenizer.pad(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            padding=True,
            return_tensors="pt",
        )

        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "labels": labels,
            "opinion_targets": opinion_targets,
        }


class MultiAspectOpinionModel(nn.Module):
    def __init__(self, model_id: str, num_classes: int = 4):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.2)
        rep_dim = hidden_size * 2  # CLS + mean pooling

        self.shared_proj = nn.Sequential(
            nn.Linear(rep_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.price_head = nn.Linear(hidden_size, num_classes)
        self.food_head = nn.Linear(hidden_size, num_classes)
        self.service_head = nn.Linear(hidden_size, num_classes)

        self.price_opinion_head = nn.Linear(hidden_size, 1)
        self.food_opinion_head = nn.Linear(hidden_size, 1)
        self.service_opinion_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        last_hidden = outputs.last_hidden_state  # [B, T, H]
        cls_vec = last_hidden[:, 0, :]  # [B, H]

        mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
        masked_hidden = last_hidden * mask
        sum_hidden = masked_hidden.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        mean_vec = sum_hidden / denom

        rep = torch.cat([cls_vec, mean_vec], dim=-1)
        rep = self.shared_proj(self.dropout(rep))

        price_logits = self.price_head(rep)
        food_logits = self.food_head(rep)
        service_logits = self.service_head(rep)

        logits = torch.stack([price_logits, food_logits, service_logits], dim=1)  # [B, 3, 4]

        opinion_logits = torch.cat(
            [
                self.price_opinion_head(rep),
                self.food_opinion_head(rep),
                self.service_opinion_head(rep),
            ],
            dim=1,
        )  # [B, 3]

        return {
            "logits": logits,
            "opinion_logits": opinion_logits,
        }


def cross_entropy_with_label_smoothing(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    num_classes = logits.size(-1)
    log_probs = torch.log_softmax(logits, dim=-1)

    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(label_smoothing / max(1, num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)

    if weight is not None:
        sample_weights = weight[targets]  # [B]
        loss = -(true_dist * log_probs).sum(dim=-1) * sample_weights
        return loss.mean()

    loss = -(true_dist * log_probs).sum(dim=-1)
    return loss.mean()