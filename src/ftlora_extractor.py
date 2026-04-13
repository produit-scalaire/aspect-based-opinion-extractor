from __future__ import annotations

import copy
import math
import os
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


class OpinionExtractor:
    # Fine-tuning approach
    method: Literal["NOFT", "FT"] = "FT"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.accelerator = Accelerator()

        self.aspects = ["Price", "Food", "Service"]
        self.aspect_to_fr = {
            "Price": "prix",
            "Food": "nourriture",
            "Service": "service",
        }

        self.label_to_id = {
            "Positive": 0,
            "Negative": 1,
            "Mixed": 2,
            "No Opinion": 3,
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        # Authorized encoder-only models from the assignment.
        # Best practical default for French reviews:
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
        self.num_labels_binary = 2  # [has_positive, has_negative]

        # Calibrated on validation after training
        self.thresholds = {
            aspect: {"pos": 0.5, "neg": 0.5} for aspect in self.aspects
        }

        self._set_seed(42)

    # -----------------------------
    # Public API
    # -----------------------------
    def train(self, train_data: list[dict], val_data: list[dict]) -> None:
        """
        Trains the model, if OpinionExtractor.method=="FT"
        """
        self._set_seed(42 + self.accelerator.process_index)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model_id = self._pick_model_id()
        if self.accelerator.is_main_process:
            print(f"Using encoder model: {self.model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.model = AspectBinaryClassifier(self.model_id)

        train_samples = self._build_samples(train_data)
        val_samples = self._build_samples(val_data)

        train_dataset = AspectOpinionDataset(
            train_samples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        val_dataset = AspectOpinionDataset(
            val_samples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        # Reasonable default for a base encoder model
        num_devices = max(1, self.accelerator.num_processes)
        per_device_batch_size = 8
        target_effective_batch_size = 48
        grad_accum_steps = max(
            1,
            math.ceil(target_effective_batch_size / (per_device_batch_size * num_devices)),
        )
        effective_batch_size = per_device_batch_size * num_devices * grad_accum_steps

        # Slight LR scaling with effective batch size
        base_lr = 2e-5
        learning_rate = base_lr * math.sqrt(effective_batch_size / target_effective_batch_size)
        learning_rate = float(min(3e-5, max(1.5e-5, learning_rate)))

        train_loader = DataLoader(
            train_dataset,
            batch_size=per_device_batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=per_device_batch_size * 2,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
            drop_last=False,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        num_epochs = 6
        steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
        total_train_steps = max(1, num_epochs * steps_per_epoch)
        warmup_steps = max(1, int(0.1 * total_train_steps))

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
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
                    labels=batch["labels"],
                    pos_weight=batch["pos_weight"],
                )
                loss = outputs["loss"] / grad_accum_steps
                self.accelerator.backward(loss)

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                running_loss += float(loss.detach().item())
                num_batches += 1

            val_metric, raw_val_outputs = self._evaluate_binary_model(val_loader)
            if self.accelerator.is_main_process:
                avg_train_loss = running_loss / max(1, num_batches)
                print(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"train_loss={avg_train_loss:.4f} | "
                    f"val_macro_acc={val_metric:.2f}"
                )

            improved = val_metric > best_metric
            if improved:
                best_metric = val_metric
                patience_counter = 0
                unwrapped = self.accelerator.unwrap_model(self.model)
                best_state = copy.deepcopy(unwrapped.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if self.accelerator.is_main_process:
                        print("Early stopping triggered.")
                    break

        # Restore best model
        if best_state is not None:
            self.accelerator.unwrap_model(self.model).load_state_dict(best_state)

        # Calibrate thresholds on validation set
        self._calibrate_thresholds_from_val(val_loader)

        self.model.eval()
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            print("Calibrated thresholds:")
            for aspect in self.aspects:
                print(aspect, self.thresholds[aspect])

    def predict(self, texts: list[str]) -> list[dict]:
        """
        :param texts: list of reviews from which to extract the opinion values
        :return: a list of dicts, one per input review, containing the opinion values for the 3 aspects.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("The model is not trained yet. Call train() first.")

        predict_samples = []
        sample_map = []
        for text_idx, text in enumerate(texts):
            for aspect in self.aspects:
                predict_samples.append(
                    {
                        "text": self._format_input(text, aspect),
                        "aspect": aspect,
                    }
                )
                sample_map.append((text_idx, aspect))

        dataset = PredictDataset(
            predict_samples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        loader = DataLoader(
            dataset,
            batch_size=max(8, int(getattr(self.cfg, "eval_batch_size", 10))),
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=None,
                    pos_weight=None,
                )
                probs = torch.sigmoid(outputs["logits"]).detach().cpu().numpy()
                all_probs.append(probs)

        probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 2), dtype=np.float32)

        predictions = [{aspect: "No Opinion" for aspect in self.aspects} for _ in texts]

        for idx, (text_idx, aspect) in enumerate(sample_map):
            pos_prob = float(probs[idx, 0])
            neg_prob = float(probs[idx, 1])

            pos_thr = self.thresholds[aspect]["pos"]
            neg_thr = self.thresholds[aspect]["neg"]

            label = self._binary_probs_to_label(pos_prob, neg_prob, pos_thr, neg_thr)
            predictions[text_idx][aspect] = label

        return predictions

    # -----------------------------
    # Internal helpers
    # -----------------------------
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

        # Remove noisy suffixes such as Positive#NE, Negative#foo, etc.
        if "#" in value:
            value = value.split("#", 1)[0].strip()

        # Normalize spaces / underscores
        value = value.replace("_", " ")
        value = " ".join(value.split())

        # Direct match
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
            "no opinion ": "No Opinion",
        }

        if lower in mapping:
            return mapping[lower]

        raise ValueError(f"Unknown label: {value}")

    def _label_to_binary_targets(self, label: str) -> list[float]:
        label = self._normalize_label(label)
        if label == "Positive":
            return [1.0, 0.0]
        if label == "Negative":
            return [0.0, 1.0]
        if label == "Mixed":
            return [1.0, 1.0]
        if label == "No Opinion":
            return [0.0, 0.0]
        raise ValueError(label)

    def _format_input(self, review: str, aspect: str) -> str:
        aspect_fr = self.aspect_to_fr[aspect]
        # Short aspect-aware prompt; better than raw review-only input for ABSA.
        return (
            f"Aspect : {aspect_fr}. "
            f"Détecter s'il y a un avis positif et/ou négatif. "
            f"Avis : {str(review).strip()}"
        )

    def _build_samples(self, rows: list[dict]) -> list[dict]:
        samples = []
        for row in rows:
            review = str(row["Review"])
            for aspect in self.aspects:
                label = self._normalize_label(row[aspect])
                binary_targets = self._label_to_binary_targets(label)
                samples.append(
                    {
                        "text": self._format_input(review, aspect),
                        "aspect": aspect,
                        "targets": binary_targets,
                        "original_label": label,
                    }
                )
        return samples

    def _evaluate_binary_model(self, val_loader: DataLoader) -> tuple[float, dict]:
        self.model.eval()

        all_logits = []
        all_labels = []
        all_aspects = []

        with torch.no_grad():
            for batch in val_loader:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=None,
                    pos_weight=None,
                )

                logits = outputs["logits"]
                labels = batch["labels"]

                gathered_logits = self.accelerator.gather_for_metrics(logits)
                gathered_labels = self.accelerator.gather_for_metrics(labels)

                all_logits.append(gathered_logits.detach().cpu())
                all_labels.append(gathered_labels.detach().cpu())

                if "aspect_ids" in batch:
                    gathered_aspect_ids = self.accelerator.gather_for_metrics(batch["aspect_ids"])
                    all_aspects.append(gathered_aspect_ids.detach().cpu())

        logits = torch.cat(all_logits, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        aspect_ids = torch.cat(all_aspects, dim=0).numpy()

        probs = 1.0 / (1.0 + np.exp(-logits))

        pred_labels = []
        gold_labels = []

        for i in range(len(probs)):
            aspect = self.aspects[int(aspect_ids[i])]
            pred = self._binary_probs_to_label(
                float(probs[i, 0]),
                float(probs[i, 1]),
                self.thresholds[aspect]["pos"],
                self.thresholds[aspect]["neg"],
            )
            gold = self._binary_targets_to_label(labels[i, 0], labels[i, 1])

            pred_labels.append(pred)
            gold_labels.append(gold)

        per_aspect_correct = {aspect: [] for aspect in self.aspects}
        for idx, pred, gold in zip(aspect_ids, pred_labels, gold_labels):
            aspect = self.aspects[int(idx)]
            per_aspect_correct[aspect].append(float(pred == gold))

        macro_acc = 100.0 * np.mean(
            [np.mean(v) if len(v) > 0 else 0.0 for v in per_aspect_correct.values()]
        )

        raw_outputs = {
            "logits": logits,
            "labels": labels,
            "aspect_ids": aspect_ids,
            "probs": probs,
            "pred_labels": pred_labels,
            "gold_labels": gold_labels,
        }
        return float(macro_acc), raw_outputs

    def _binary_targets_to_label(self, pos: float, neg: float) -> str:
        pos = int(round(float(pos)))
        neg = int(round(float(neg)))
        if pos == 1 and neg == 1:
            return "Mixed"
        if pos == 1 and neg == 0:
            return "Positive"
        if pos == 0 and neg == 1:
            return "Negative"
        return "No Opinion"

    def _binary_probs_to_label(
        self,
        pos_prob: float,
        neg_prob: float,
        pos_thr: float,
        neg_thr: float,
    ) -> str:
        has_pos = pos_prob >= pos_thr
        has_neg = neg_prob >= neg_thr

        if has_pos and has_neg:
            return "Mixed"
        if has_pos:
            return "Positive"
        if has_neg:
            return "Negative"
        return "No Opinion"

    def _calibrate_thresholds_from_val(self, val_loader: DataLoader) -> None:
        self.model.eval()

        all_logits = []
        all_labels = []
        all_aspects = []

        with torch.no_grad():
            for batch in val_loader:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=None,
                    pos_weight=None,
                )
                logits = outputs["logits"]
                labels = batch["labels"]

                gathered_logits = self.accelerator.gather_for_metrics(logits)
                gathered_labels = self.accelerator.gather_for_metrics(labels)
                gathered_aspect_ids = self.accelerator.gather_for_metrics(batch["aspect_ids"])

                all_logits.append(gathered_logits.detach().cpu())
                all_labels.append(gathered_labels.detach().cpu())
                all_aspects.append(gathered_aspect_ids.detach().cpu())

        logits = torch.cat(all_logits, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        aspect_ids = torch.cat(all_aspects, dim=0).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))

        grid = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

        for aspect_idx, aspect in enumerate(self.aspects):
            mask = aspect_ids == aspect_idx
            if np.sum(mask) == 0:
                continue

            aspect_probs = probs[mask]
            aspect_labels = labels[mask]

            best_acc = -1.0
            best_pair = (0.5, 0.5)

            for pos_thr in grid:
                for neg_thr in grid:
                    preds = []
                    golds = []
                    for i in range(len(aspect_probs)):
                        pred = self._binary_probs_to_label(
                            float(aspect_probs[i, 0]),
                            float(aspect_probs[i, 1]),
                            pos_thr,
                            neg_thr,
                        )
                        gold = self._binary_targets_to_label(
                            aspect_labels[i, 0],
                            aspect_labels[i, 1],
                        )
                        preds.append(pred)
                        golds.append(gold)
                    acc = 100.0 * np.mean([float(p == g) for p, g in zip(preds, golds)])
                    if acc > best_acc:
                        best_acc = acc
                        best_pair = (pos_thr, neg_thr)

            self.thresholds[aspect]["pos"] = best_pair[0]
            self.thresholds[aspect]["neg"] = best_pair[1]


class AspectOpinionDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.aspect_to_id = {"Price": 0, "Food": 1, "Service": 2}

        # Precompute class imbalance weights for the two binary subtasks
        labels = np.array([sample["targets"] for sample in samples], dtype=np.float32)
        pos_counts = labels.sum(axis=0)
        neg_counts = len(labels) - pos_counts
        self.pos_weight = (neg_counts + 1.0) / (pos_counts + 1.0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        enc = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": item["targets"],
            "aspect_id": self.aspect_to_id[item["aspect"]],
        }

    def collate_fn(self, batch):
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        attention_mask = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]
        labels = torch.tensor([x["labels"] for x in batch], dtype=torch.float32)
        aspect_ids = torch.tensor([x["aspect_id"] for x in batch], dtype=torch.long)

        padded = self.tokenizer.pad(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            padding=True,
            return_tensors="pt",
        )

        pos_weight = torch.tensor(self.pos_weight, dtype=torch.float32)

        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "labels": labels,
            "aspect_ids": aspect_ids,
            "pos_weight": pos_weight,
        }


class PredictDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        enc = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

    def collate_fn(self, batch):
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        attention_mask = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]

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
        }


class AspectBinaryClassifier(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)

        hidden_size = self.encoder.config.hidden_size

        # A bit more robust than raw CLS alone for RoBERTa/CamemBERT-like encoders:
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 2),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        pos_weight: torch.Tensor | None = None,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        last_hidden = outputs.last_hidden_state  # [B, T, H]
        cls_vec = last_hidden[:, 0, :]  # [B, H]

        # Masked mean pooling
        mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
        masked_hidden = last_hidden * mask
        sum_hidden = masked_hidden.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        mean_vec = sum_hidden / denom

        pooled = torch.cat([cls_vec, mean_vec], dim=-1)
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            if pos_weight is not None:
                # Broadcast from shape [2] to device
                pos_weight = pos_weight.to(logits.device)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fct(logits, labels.to(logits.device))

        return {
            "loss": loss,
            "logits": logits,
        }