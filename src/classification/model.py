"""Multi-task DistilBERT classifier.

One shared DistilBERT encoder with four classification heads:
  - ticket_type  (Incident / Request / Problem / Change)
  - urgency      (low / medium / high  -- see README: source has no 'critical')
  - sentiment    (negative / neutral / positive -- lexicon-derived labels)
  - team/queue   (support team assignment = routing ground truth)
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast

LABEL_COLUMNS = ["ticket_type", "urgency", "sentiment", "team"]


class MultiTaskTicketClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: dict[str, int]):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(model_name)
        hidden = self.encoder.config.dim
        self.heads = nn.ModuleDict(
            {name: nn.Linear(hidden, n) for name, n in num_labels.items()}
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # mean-pool over non-padded tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = self.dropout(pooled)
        return {name: head(pooled) for name, head in self.heads.items()}


def build_label_maps(df) -> dict[str, dict[str, int]]:
    """label -> index maps built from the training dataframe."""
    maps = {}
    for col in LABEL_COLUMNS:
        labels = sorted(df[col].astype(str).unique().tolist())
        maps[col] = {lab: i for i, lab in enumerate(labels)}
    return maps


def save_artifacts(model_dir: str | Path, model, label_maps: dict, cfg: dict):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(model_dir / "encoder")
    torch.save(model.heads.state_dict(), model_dir / "heads.pt")
    meta = {
        "base_model": cfg["classification"]["model_name"],
        "label_maps": label_maps,
        "max_length": cfg["classification"]["max_length"],
        "confidence_threshold": cfg["classification"]["confidence_threshold"],
    }
    with open(model_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_artifacts(model_dir: str | Path, device: str = "cpu"):
    model_dir = Path(model_dir)
    with open(model_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    num_labels = {k: len(v) for k, v in meta["label_maps"].items()}
    model = MultiTaskTicketClassifier(meta["base_model"], num_labels)
    model.encoder = DistilBertModel.from_pretrained(model_dir / "encoder")
    model.heads.load_state_dict(torch.load(model_dir / "heads.pt", map_location=device))
    model.eval().to(device)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir / "encoder")
    return model, tokenizer, meta
