from __future__ import annotations

import math
import random
import re
from collections import Counter
from typing import Literal

import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


ASPECTS = ["Price", "Food", "Service"]
LABELS = ["Negative", "Mixed", "No Opinion", "Positive"]


class OpinionExtractor:

    # Approach 3 = fine-tuned encoder-only LM
    method: Literal["NOFT", "FT"] = "FT"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.model_id = "flaubert/flaubert_large_cased"
        self.max_length = 320
        self.num_labels = len(LABELS)

        self.label2id = {label: i for i, label in enumerate(LABELS)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.model = None

        self.seed = 1337

        # Simple calibrated decision rules for difficult classes
        self.no_op_threshold = {
            "Price": 0.45,
            "Food": 0.40,
            "Service": 0.40,
        }
        self.mixed_threshold = {
            "Price": 0.22,
            "Food": 0.20,
            "Service": 0.20,
        }

    def train(self, train_data: list[dict], val_data: list[dict]) -> None:
        """
        Trains the model, if OpinionExtractor.method=="FT"
        """
        self._set_seed(self.seed)

        clean_train = self._prepare_dataframe(train_data, is_train=True)
        clean_val = self._prepare_dataframe(val_data, is_train=False)

        train_ds = ReviewDataset(clean_train)
        val_ds = ReviewDataset(clean_val)

        num_devices = max(1, torch.cuda.device_count())
        per_device_batch_size = 8 if torch.cuda.is_available() else 4
        target_effective_batch = 32
        grad_accum_steps = max(1, target_effective_batch // (per_device_batch_size * num_devices))
        effective_batch = per_device_batch_size * num_devices * grad_accum_steps

        base_lr = 2.0e-5
        lr = base_lr * (effective_batch / 32.0)
        lr = min(3.0e-5, max(1.5e-5, lr))

        train_loader = DataLoader(
            train_ds,
            batch_size=per_device_batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=per_device_batch_size * 2,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        class_weights = self._compute_class_weights(clean_train).to(self.device)

        model = MultiHeadXLMR(
            model_id=self.model_id,
            num_labels=self.num_labels,
            class_weights=class_weights,
        )

        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8)

        num_epochs = 7
        num_update_steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
        max_train_steps = num_epochs * num_update_steps_per_epoch
        warmup_steps = max(1, int(0.1 * max_train_steps))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )

        model, optimizer, train_loader, val_loader, scheduler = self.accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )

        best_metric = -1.0
        best_state = None
        patience = 2
        bad_epochs = 0

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            for batch in train_loader:
                with self.accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs["loss"]
                    self.accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            val_metric = self._evaluate(model, val_loader)

            if self.accelerator.is_main_process:
                print(f"Epoch {epoch + 1}/{num_epochs} - val_macro_acc={val_metric:.4f}")

            if val_metric > best_metric:
                best_metric = val_metric
                bad_epochs = 0
                unwrapped = self.accelerator.unwrap_model(model)
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in unwrapped.state_dict().items()
                }
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    if self.accelerator.is_main_process:
                        print("Early stopping triggered.")
                    break

        self.accelerator.wait_for_everyone()
        final_model = self.accelerator.unwrap_model(model)
        if best_state is not None:
            final_model.load_state_dict(best_state, strict=True)
        final_model.eval()

        self.model = final_model.to(self.device)

    def predict(self, texts: list[str]) -> list[dict]:
        """
        :param texts: list of reviews from which to extract the opinion values
        :return: a list of dicts, one per input review, containing the opinion values for the 3 aspects.
        """
        if self.model is None:
            raise RuntimeError("The model is not trained yet. Call train() first.")

        clean_texts = [self._normalize_text(text) for text in texts]
        encoded = self.tokenizer(
            clean_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encoded)
            preds = {
                aspect: self._decode_with_rules(aspect, logits)
                for aspect, logits in outputs["logits"].items()
            }

        results = []
        for i in range(len(clean_texts)):
            row = {}
            for aspect in ASPECTS:
                row[aspect] = self.id2label[preds[aspect][i]]
            results.append(row)
        return results

    def _prepare_dataframe(self, data: list[dict], is_train: bool) -> pd.DataFrame:
        rows = []
        for row in data:
            text = self._normalize_text(row["Review"])
            if not text:
                continue

            item = {"Review": text}
            for aspect in ASPECTS:
                item[aspect] = self._normalize_label(row.get(aspect, "No Opinion"))
            rows.append(item)

        df = pd.DataFrame(rows)

        if is_train:
            # 1) remove exact duplicates with identical labels
            df = df.drop_duplicates(subset=["Review", "Price", "Food", "Service"]).reset_index(drop=True)

            # 2) aggregate duplicate reviews with noisy label disagreements by majority vote per aspect
            grouped_rows = []
            for review, group in df.groupby("Review", sort=False):
                agg = {"Review": review}
                for aspect in ASPECTS:
                    votes = [self._normalize_label(x) for x in group[aspect].tolist()]
                    agg[aspect] = Counter(votes).most_common(1)[0][0]
                grouped_rows.append(agg)
            df = pd.DataFrame(grouped_rows)

            # 3) keep only non-empty normalized reviews
            df = df[df["Review"].str.len() >= 3].reset_index(drop=True)

            # 4) light oversampling for difficult labels
            augmented_rows = []
            for _, row in df.iterrows():
                repeat = 1
                labels = [row[a] for a in ASPECTS]

                if "Mixed" in labels:
                    repeat += 1

                if sum(1 for x in labels if x == "No Opinion") >= 2:
                    repeat += 1

                for _ in range(repeat):
                    augmented_rows.append(row.to_dict())

            df = pd.DataFrame(augmented_rows).reset_index(drop=True)

        return df

    def _normalize_label(self, label: str) -> str:
        if label is None:
            return "No Opinion"
        label = str(label).strip()
        fixes = {
            "Positive#NE": "Positive",
            "Positif": "Positive",
            "Négatif": "Negative",
            "Sans Opinion": "No Opinion",
            "No opinion": "No Opinion",
            "Mixed/Neutral": "Mixed",
        }
        label = fixes.get(label, label)
        if label not in self.label2id:
            return "No Opinion"
        return label

    def _normalize_text(self, text: str) -> str:
        text = "" if text is None else str(text)

        # Strip CSV/TSV quote artifacts.
        if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        text = text.replace('""', '"')

        # Basic Unicode/spacing cleanup.
        text = text.replace("\u00a0", " ").replace("\u200b", " ")
        text = text.replace("\r", " ").replace("\n", " ")
        text = re.sub(r"\s+", " ", text)

        # Compress very long punctuation streaks but keep sentiment cues.
        text = re.sub(r"([!?.,;:])\1{2,}", r"\1\1", text)

        # Compress pathological character repetition: "trooooop" -> "trooop"
        text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)

        return text.strip()

    def _compute_class_weights(self, df: pd.DataFrame) -> torch.Tensor:
        weights = []
        eps = 1e-6
        for aspect in ASPECTS:
            counts = torch.zeros(self.num_labels, dtype=torch.float)
            for label in df[aspect].tolist():
                counts[self.label2id[label]] += 1.0

            inv = 1.0 / torch.sqrt(counts + eps)
            inv = inv / inv.mean()

            # Boost difficult classes
            inv[self.label2id["Mixed"]] *= 1.35
            inv[self.label2id["No Opinion"]] *= 1.15

            inv = inv / inv.mean()
            weights.append(inv)

        return torch.stack(weights, dim=0)

    def _collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        texts = [item["Review"] for item in batch]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        for aspect in ASPECTS:
            enc[f"labels_{aspect.lower()}"] = torch.tensor(
                [item[aspect] for item in batch],
                dtype=torch.long,
            )
        return enc

    def _evaluate(self, model: nn.Module, dataloader: DataLoader) -> float:
        model.eval()

        gathered_preds = {aspect: [] for aspect in ASPECTS}
        gathered_refs = {aspect: [] for aspect in ASPECTS}

        with torch.no_grad():
            for batch in dataloader:
                outputs = model(**batch)
                for aspect in ASPECTS:
                    logits = outputs["logits"][aspect]
                    preds = torch.argmax(logits, dim=-1)
                    refs = batch[f"labels_{aspect.lower()}"]

                    preds, refs = self.accelerator.gather_for_metrics((preds, refs))
                    gathered_preds[aspect].append(preds.cpu())
                    gathered_refs[aspect].append(refs.cpu())

        macro = 0.0
        for aspect in ASPECTS:
            preds = torch.cat(gathered_preds[aspect], dim=0)
            refs = torch.cat(gathered_refs[aspect], dim=0)
            acc = (preds == refs).float().mean().item()
            macro += acc
        macro /= len(ASPECTS)

        model.train()
        return macro

    def _decode_with_rules(self, aspect: str, logits: torch.Tensor) -> list[int]:
        probs = torch.softmax(logits, dim=-1)

        neg_id = self.label2id["Negative"]
        mix_id = self.label2id["Mixed"]
        no_id = self.label2id["No Opinion"]
        pos_id = self.label2id["Positive"]

        predictions = []
        for p in probs:
            p_neg = p[neg_id].item()
            p_mix = p[mix_id].item()
            p_no = p[no_id].item()
            p_pos = p[pos_id].item()

            # Prefer No Opinion when its posterior is high and polarity is weak
            if p_no >= self.no_op_threshold[aspect] and max(p_neg, p_pos) < 0.40:
                predictions.append(no_id)
                continue

            # Prefer Mixed when both positive and negative are sufficiently present
            if min(p_neg, p_pos) >= self.mixed_threshold[aspect]:
                predictions.append(mix_id)
                continue

            predictions.append(int(torch.argmax(p).item()))

        return predictions

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class ReviewDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.records = df.to_dict(orient="records")
        self.label2id = {label: i for i, label in enumerate(LABELS)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        item = {"Review": row["Review"]}
        for aspect in ASPECTS:
            item[aspect] = self.label2id[row[aspect]]
        return item


class MultiHeadXLMR(nn.Module):
    def __init__(self, model_id: str, num_labels: int, class_weights: torch.Tensor):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(hidden_size * 2)

        self.price_head = AspectHead(hidden_size * 2, num_labels)
        self.food_head = AspectHead(hidden_size * 2, num_labels)
        self.service_head = AspectHead(hidden_size * 2, num_labels)

        self.register_buffer("class_weights", class_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels_price: torch.Tensor | None = None,
        labels_food: torch.Tensor | None = None,
        labels_service: torch.Tensor | None = None,
        **kwargs
    ) -> dict:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden = outputs.last_hidden_state

        cls = hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        pooled = torch.cat([cls, mean], dim=-1)
        pooled = self.norm(self.dropout(pooled))

        logits = {
            "Price": self.price_head(pooled),
            "Food": self.food_head(pooled),
            "Service": self.service_head(pooled),
        }

        result = {"logits": logits}

        if labels_price is not None and labels_food is not None and labels_service is not None:
            losses = []
            ce_price = nn.CrossEntropyLoss(weight=self.class_weights[0], label_smoothing=0.03)
            ce_food = nn.CrossEntropyLoss(weight=self.class_weights[1], label_smoothing=0.03)
            ce_service = nn.CrossEntropyLoss(weight=self.class_weights[2], label_smoothing=0.03)

            losses.append(ce_price(logits["Price"], labels_price))
            losses.append(ce_food(logits["Food"], labels_food))
            losses.append(ce_service(logits["Service"], labels_service))

            result["loss"] = sum(losses) / len(losses)

        return result


class AspectHead(nn.Module):
    def __init__(self, input_dim: int, num_labels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(input_dim // 2, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)