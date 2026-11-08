"""Training + evaluation for the multi-task ticket classifier.

Run:  python -m src.classification.train
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from src.evaluation.report import metrics_report  # noqa: F401  (re-exported)
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast, get_linear_schedule_with_warmup

from src.classification.model import (LABEL_COLUMNS, MultiTaskTicketClassifier,
                                      build_label_maps, load_artifacts,
                                      save_artifacts)
from src.utils import load_config, project_path


class TicketDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, label_maps, max_length: int):
        self.texts = df["ticket_text"].astype(str).tolist()
        self.labels = {
            col: df[col].astype(str).map(label_maps[col]).fillna(0).astype(int).tolist()
            for col in LABEL_COLUMNS
        }
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, max_length=self.max_length,
        )
        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
        for col in LABEL_COLUMNS:
            item[col] = torch.tensor(self.labels[col][idx], dtype=torch.long)
        return item


def make_collate(tokenizer):
    """Dynamic padding: pad each batch to its longest sequence (big CPU win)."""
    def collate(batch):
        feats = tokenizer.pad(
            {"input_ids": [b["input_ids"] for b in batch],
             "attention_mask": [b["attention_mask"] for b in batch]},
            padding=True, return_tensors="pt",
        )
        out = dict(feats)
        for col in LABEL_COLUMNS:
            out[col] = torch.stack([b[col] for b in batch])
        return out
    return collate


def train_one_epoch(model, loader, opt, sched, device, class_weights):
    model.train()
    total_loss = 0.0
    ce = nn.CrossEntropyLoss(weight=None)
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = {c: batch.pop(c) for c in LABEL_COLUMNS}
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = sum(ce(logits[c], labels[c]) for c in LABEL_COLUMNS)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = {c: [] for c in LABEL_COLUMNS}
    all_probs = {c: [] for c in LABEL_COLUMNS}
    all_true = {c: [] for c in LABEL_COLUMNS}
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = {c: batch.pop(c) for c in LABEL_COLUMNS}
        logits = model(batch["input_ids"], batch["attention_mask"])
        for c in LABEL_COLUMNS:
            probs = torch.softmax(logits[c], dim=-1)
            all_probs[c].append(probs.cpu().numpy())
            all_preds[c].append(probs.argmax(-1).cpu().numpy())
            all_true[c].append(labels[c].cpu().numpy())
    return (
        {c: np.concatenate(v) for c, v in all_preds.items()},
        {c: np.concatenate(v) for c, v in all_probs.items()},
        {c: np.concatenate(v) for c, v in all_true.items()},
    )


def main():
    cfg = load_config()
    proc = Path(cfg["paths"]["processed_dir"])
    train_df = pd.read_csv(proc / "tickets_train.csv")
    val_df = pd.read_csv(proc / "tickets_val.csv")
    test_df = pd.read_csv(proc / "tickets_test.csv")
    _cfg = load_config()
    max_n = _cfg["classification"].get("max_train_samples")
    if max_n:
        train_df = train_df.sample(n=min(int(max_n), len(train_df)),
                                   random_state=int(_cfg["project"]["seed"]))

    ccfg = cfg["classification"]
    device = ccfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    tokenizer = DistilBertTokenizerFast.from_pretrained(ccfg["model_name"])
    label_maps = build_label_maps(train_df)
    num_labels = {c: len(m) for c, m in label_maps.items()}
    model = MultiTaskTicketClassifier(ccfg["model_name"], num_labels).to(device)

    max_len = int(ccfg["max_length"])
    bs = int(ccfg["batch_size"])
    train_ds = TicketDataset(train_df, tokenizer, label_maps, max_len)
    val_ds = TicketDataset(val_df, tokenizer, label_maps, max_len)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=make_collate(tokenizer))
    val_loader = DataLoader(val_ds, batch_size=bs, collate_fn=make_collate(tokenizer))

    epochs = int(ccfg["epochs"])
    steps = len(train_loader) * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=float(ccfg["learning_rate"]))
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    best_f1 = -1.0
    out_dir = Path(cfg["paths"]["models_dir"]) / "classification" / "distilbert_multitask"
    inv_type = {i: lab for lab, i in label_maps["ticket_type"].items()}
    for ep in range(epochs):
        loss = train_one_epoch(model, train_loader, opt, sched, device, None)
        yp, _, yt = evaluate(model, val_loader, device)
        f1 = f1_score(
            [inv_type[i] for i in yt["ticket_type"]],
            [inv_type[i] for i in yp["ticket_type"]],
            average="macro", zero_division=0,
        )
        print(f"epoch {ep+1}/{epochs} loss={loss:.4f} val_ticket_type_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            save_artifacts(out_dir, model, label_maps, cfg)

    # ---- test evaluation ----
    model, _, meta = load_artifacts(out_dir, device)
    test_ds = TicketDataset(test_df, tokenizer, label_maps, max_len)
    test_loader = DataLoader(test_ds, batch_size=bs, collate_fn=make_collate(tokenizer))
    preds, probs, trues = evaluate(model, test_loader, device)
    report = metrics_report(trues, preds, label_maps)

    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "classification_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({c: {"accuracy": r["accuracy"], "f1_macro": r["f1_macro"]}
                      for c, r in report.items()}, indent=2))


if __name__ == "__main__":
    main()
