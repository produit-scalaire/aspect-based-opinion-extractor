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
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


ASPECTS = ["Price", "Food", "Service"]
LABELS = ["Negative", "Mixed", "No Opinion", "Positive"]
ASPECT_TOKENS = {
    "Price": "[ASPECT_PRICE]",
    "Food": "[ASPECT_FOOD]",
    "Service": "[ASPECT_SERVICE]",
}
ASPECT_PROMPTS = {
    "Price": "prix",
    "Food": "nourriture",
    "Service": "service",
}


class OpinionExtractor:
    method: Literal["NOFT", "FT"] = "FT"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        # Plan A: French encoder + aspect-conditioned single-head classifier
        self.model_id = "almanach/moderncamembert-base"
        self.max_length = 320
        self.num_labels = len(LABELS)

        self.label2id = {label: i for i, label in enumerate(LABELS)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        special_tokens = {"additional_special_tokens": list(ASPECT_TOKENS.values())}
        self.tokenizer.add_special_tokens(special_tokens)

        self.model = None
        self.seed = 1337

    def train(self, train_data: list[dict], val_data: list[dict]) -> None:
        self._set_seed(self.seed)

        clean_train = self._prepare_dataframe(train_data, is_train=True)
        clean_val = self._prepare_dataframe(val_data, is_train=False)

        train_examples = self._to_aspect_examples(clean_train, is_train=True)
        val_examples = self._to_aspect_examples(clean_val, is_train=False)

        train_ds = AspectConditionedDataset(train_examples, self.label2id)
        val_ds = AspectConditionedDataset(val_examples, self.label2id)

        num_devices = max(1, torch.cuda.device_count())
        per_device_batch_size = 8 if torch.cuda.is_available() else 4
        target_effective_batch = 48
        grad_accum_steps = max(1, math.ceil(target_effective_batch / (per_device_batch_size * num_devices)))
        effective_batch = per_device_batch_size * num_devices * grad_accum_steps

        base_lr = 1.5e-5
        lr = base_lr * (effective_batch / 32.0)
        lr = min(2.5e-5, max(1.0e-5, lr))

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

        class_weights = self._compute_global_class_weights(train_examples).to(self.device)

        model = AspectConditionedCamembert(
            model_id=self.model_id,
            num_labels=self.num_labels,
            class_weights=class_weights,
        )
        model.encoder.resize_token_embeddings(len(self.tokenizer))

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
        warmup_steps = max(1, int(0.08 * max_train_steps))
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
                best_state = {k: v.detach().cpu().clone() for k, v in unwrapped.state_dict().items()}
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
        if self.model is None:
            raise RuntimeError("The model is not trained yet. Call train() first.")

        clean_texts = [self._normalize_text(text) for text in texts]
        aspect_predictions: dict[str, list[str]] = {aspect: [] for aspect in ASPECTS}

        self.model.eval()
        with torch.no_grad():
            for aspect in ASPECTS:
                prefixed_texts = [self._format_aspect_input(text, aspect) for text in clean_texts]
                encoded = self.tokenizer(
                    prefixed_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded)
                pred_ids = torch.argmax(outputs["logits"], dim=-1).cpu().tolist()
                aspect_predictions[aspect] = [self.id2label[i] for i in pred_ids]

        results = []
        for i in range(len(clean_texts)):
            results.append({aspect: aspect_predictions[aspect][i] for aspect in ASPECTS})
        return results

    def _prepare_dataframe(self, data: list[dict], is_train: bool) -> pd.DataFrame:
        rows = []
        for row in data:
            text = self._normalize_text(row.get("Review", ""))
            if not text:
                continue
            item = {"Review": text}
            for aspect in ASPECTS:
                item[aspect] = self._normalize_label(row.get(aspect, "No Opinion"))
            rows.append(item)

        df = pd.DataFrame(rows)
        if len(df) == 0:
            return df

        if is_train:
            df = df.drop_duplicates(subset=["Review", "Price", "Food", "Service"]).reset_index(drop=True)
            grouped_rows = []
            for review, group in df.groupby("Review", sort=False):
                agg = {"Review": review}
                for aspect in ASPECTS:
                    votes = [self._normalize_label(x) for x in group[aspect].tolist()]
                    agg[aspect] = Counter(votes).most_common(1)[0][0]
                grouped_rows.append(agg)
            df = pd.DataFrame(grouped_rows)
            df = df[df["Review"].str.len() >= 3].reset_index(drop=True)

        return df

    def _to_aspect_examples(self, df: pd.DataFrame, is_train: bool) -> list[dict]:
        examples = []
        for row in df.to_dict(orient="records"):
            for aspect in ASPECTS:
                label = row[aspect]
                item = {
                    "text": self._format_aspect_input(row["Review"], aspect),
                    "aspect": aspect,
                    "label": label,
                }
                examples.append(item)

                if is_train and label in {"Negative", "Mixed"}:
                    examples.append(item.copy())
                if is_train and aspect == "Price" and label != "No Opinion":
                    examples.append(item.copy())
        return examples

    def _format_aspect_input(self, text: str, aspect: str) -> str:
        return f"{ASPECT_TOKENS[aspect]} Aspect: {ASPECT_PROMPTS[aspect]}. Avis: {text}"

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
        if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        text = text.replace('""', '"')
        text = text.replace("\u00a0", " ").replace("\u200b", " ")
        text = text.replace("\r", " ").replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"([!?.,;:])\1{2,}", r"\1\1", text)
        text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
        return text.strip()

    def _compute_global_class_weights(self, examples: list[dict]) -> torch.Tensor:
        counts = torch.zeros(self.num_labels, dtype=torch.float)
        eps = 1e-6
        for ex in examples:
            counts[self.label2id[ex["label"]]] += 1.0
        inv = 1.0 / torch.sqrt(counts + eps)
        return inv / inv.mean()

    def _collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor([item["label_id"] for item in batch], dtype=torch.long)
        return enc

    def _evaluate(self, model: nn.Module, dataloader: DataLoader) -> float:
        model.eval()

        all_aspect_preds = {aspect: [] for aspect in ASPECTS}
        all_aspect_refs = {aspect: [] for aspect in ASPECTS}

        with torch.no_grad():
            for batch in dataloader:
                outputs = model(**batch)
                preds = torch.argmax(outputs["logits"], dim=-1)
                refs = batch["labels"]
                aspects = batch["aspects"]

                preds, refs = self.accelerator.gather_for_metrics((preds, refs))
                gathered_aspects = self.accelerator.gather_for_metrics(aspects)

                preds = preds.cpu().tolist()
                refs = refs.cpu().tolist()
                gathered_aspects = gathered_aspects.cpu().tolist()

                for pred, ref, aspect_idx in zip(preds, refs, gathered_aspects):
                    aspect = ASPECTS[aspect_idx]
                    all_aspect_preds[aspect].append(pred)
                    all_aspect_refs[aspect].append(ref)

        macro = 0.0
        for aspect in ASPECTS:
            preds = torch.tensor(all_aspect_preds[aspect], dtype=torch.long)
            refs = torch.tensor(all_aspect_refs[aspect], dtype=torch.long)
            acc = (preds == refs).float().mean().item() if len(refs) > 0 else 0.0
            macro += acc
        macro /= len(ASPECTS)

        model.train()
        return macro

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class AspectConditionedDataset(Dataset):
    def __init__(self, examples: list[dict], label2id: dict[str, int]):
        self.records = []
        aspect2id = {aspect: i for i, aspect in enumerate(ASPECTS)}
        for ex in examples:
            self.records.append(
                {
                    "text": ex["text"],
                    "label_id": label2id[ex["label"]],
                    "aspect_id": aspect2id[ex["aspect"]],
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


class AspectConditionedCamembert(nn.Module):
    def __init__(self, model_id: str, num_labels: int, class_weights: torch.Tensor):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, num_labels),
        )

        self.register_buffer("class_weights", class_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden = outputs.last_hidden_state

        cls = hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = torch.cat([cls, mean], dim=-1)
        pooled = self.norm(self.dropout(pooled))

        logits = self.classifier(pooled)
        result = {"logits": logits}

        if labels is not None:
            ce = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.03)
            result["loss"] = ce(logits, labels)
        return result
