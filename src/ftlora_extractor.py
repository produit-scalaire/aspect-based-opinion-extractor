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
ASPECT_TEXT = {
    "Price": "prix",
    "Food": "nourriture",
    "Service": "service",
}
ASPECT_TOKEN = {
    "Price": "[ASPECT_PRICE]",
    "Food": "[ASPECT_FOOD]",
    "Service": "[ASPECT_SERVICE]",
}
ASPECT2ID = {a: i for i, a in enumerate(ASPECTS)}
ID2ASPECT = {i: a for a, i in ASPECT2ID.items()}
LABELS = ["Negative", "Mixed", "No Opinion", "Positive"]


class OpinionExtractor:
    method: Literal["NOFT", "FT"] = "FT"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.model_id = "almanach/moderncamembert-base"
        self.max_length = 256
        self.num_labels = len(LABELS)

        self.label2id = {label: i for i, label in enumerate(LABELS)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        added = self.tokenizer.add_special_tokens(list(ASPECT_TOKEN.values()))
        self.model = None

        self.seed = 1337
        self._added_tokens = added

    def train(self, train_data: list[dict], val_data: list[dict]) -> None:
        self._set_seed(self.seed)

        clean_train = self._prepare_dataframe(train_data, is_train=True)
        clean_val = self._prepare_dataframe(val_data, is_train=False)

        train_ds = AspectConditionedDataset(clean_train, oversample=True)
        val_ds = AspectConditionedDataset(clean_val, oversample=False)

        num_devices = max(1, torch.cuda.device_count())
        per_device_batch_size = 8 if torch.cuda.is_available() else 4
        target_effective_batch = 48
        grad_accum_steps = max(1, math.ceil(target_effective_batch / (per_device_batch_size * num_devices)))
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

        class_weights = self._compute_class_weights(train_ds.records).to(self.device)

        model = AspectConditionedCamembert(
            model_id=self.model_id,
            num_labels=self.num_labels,
            class_weights=class_weights,
            added_tokens=self._added_tokens,
            tokenizer_size=len(self.tokenizer),
        )

        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
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

        self.model.eval()
        clean_texts = [self._normalize_text(t) for t in texts]
        results = []

        for text in clean_texts:
            row = {}
            batch_texts = [self._build_prompt(text, aspect) for aspect in ASPECTS]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                preds = torch.argmax(outputs["logits"], dim=-1).cpu().tolist()

            for aspect, pred in zip(ASPECTS, preds):
                row[aspect] = self.id2label[pred]
            results.append(row)

        return results

    def _prepare_dataframe(self, data: list[dict], is_train: bool) -> pd.DataFrame:
        rows = []
        for row in data:
            text = self._normalize_text(row.get("Review"))
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

    def _build_prompt(self, review: str, aspect: str) -> str:
        return f"{ASPECT_TOKEN[aspect]} Aspecte: {ASPECT_TEXT[aspect]}. Avis: {review}"

    def _compute_class_weights(self, records: list[dict]) -> torch.Tensor:
        counts = torch.zeros(self.num_labels, dtype=torch.float)
        eps = 1e-6
        for row in records:
            counts[row["label"]] += 1.0
        inv = 1.0 / torch.sqrt(counts + eps)
        inv = inv / inv.mean()
        return inv

    def _collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        enc["aspect_ids"] = torch.tensor([item["aspect_id"] for item in batch], dtype=torch.long)
        return enc

    def _evaluate(self, model: nn.Module, dataloader: DataLoader) -> float:
        model.eval()
        preds_by_aspect = {aspect: [] for aspect in ASPECTS}
        refs_by_aspect = {aspect: [] for aspect in ASPECTS}

        with torch.no_grad():
            for batch in dataloader:
                outputs = model(**batch)
                preds = torch.argmax(outputs["logits"], dim=-1)
                refs = batch["labels"]
                aspect_ids = batch["aspect_ids"]

                preds, refs, aspect_ids = self.accelerator.gather_for_metrics((preds, refs, aspect_ids))
                preds = preds.cpu()
                refs = refs.cpu()
                aspect_ids = aspect_ids.cpu()

                for aid in range(len(ASPECTS)):
                    mask = aspect_ids == aid
                    if mask.any():
                        aspect = ID2ASPECT[aid]
                        preds_by_aspect[aspect].append(preds[mask])
                        refs_by_aspect[aspect].append(refs[mask])

        macro = 0.0
        for aspect in ASPECTS:
            if len(preds_by_aspect[aspect]) == 0:
                continue
            preds = torch.cat(preds_by_aspect[aspect], dim=0)
            refs = torch.cat(refs_by_aspect[aspect], dim=0)
            macro += (preds == refs).float().mean().item()
        macro /= len(ASPECTS)
        model.train()
        return macro

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class AspectConditionedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, oversample: bool = False):
        self.records = []
        if len(df) == 0:
            return

        for row in df.to_dict(orient="records"):
            for aspect in ASPECTS:
                label = LABELS.index(row[aspect])
                base_record = {
                    "text": f"{ASPECT_TOKEN[aspect]} Aspecte: {ASPECT_TEXT[aspect]}. Avis: {row['Review']}",
                    "label": label,
                    "aspect_id": ASPECT2ID[aspect],
                }
                self.records.append(base_record)

                if oversample:
                    repeat = 0
                    if row[aspect] == "Mixed":
                        repeat = 2
                    elif row[aspect] == "Negative":
                        repeat = 1
                    elif aspect == "Price" and row[aspect] != "No Opinion":
                        repeat = max(repeat, 1)
                    for _ in range(repeat):
                        self.records.append(dict(base_record))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


class AspectConditionedCamembert(nn.Module):
    def __init__(
        self,
        model_id: str,
        num_labels: int,
        class_weights: torch.Tensor,
        added_tokens: int,
        tokenizer_size: int,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        if added_tokens > 0:
            self.encoder.resize_token_embeddings(tokenizer_size)
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
        aspect_ids: torch.Tensor | None = None,
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
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.03)
            result["loss"] = loss_fct(logits, labels)
        return result
