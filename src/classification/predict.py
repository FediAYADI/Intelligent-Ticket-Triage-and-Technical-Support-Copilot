"""Inference wrapper: text -> multi-task predictions with confidence scores."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.classification.model import LABEL_COLUMNS, load_artifacts
from src.utils import load_config


class TicketClassifier:
    """Loads the trained multi-task model once and predicts on demand."""

    def __init__(self, model_dir: str | Path | None = None, device: str | None = None):
        cfg = load_config()
        self.cfg = cfg
        if model_dir is None:
            model_dir = Path(cfg["paths"]["models_dir"]) / "classification" / "distilbert_multitask"
        ccfg = cfg["classification"]
        self.device = device or ccfg.get("device", "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self.model, self.tokenizer, self.meta = load_artifacts(model_dir, self.device)
        self.max_length = int(self.meta["max_length"])
        self.threshold = float(self.meta["confidence_threshold"])
        self.label_maps = self.meta["label_maps"]
        self.index_to_label = {
            c: {i: lab for lab, i in m.items()} for c, m in self.label_maps.items()
        }

    @torch.no_grad()
    def predict_one(self, text: str) -> dict:
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(enc["input_ids"], enc["attention_mask"])
        result = {}
        min_conf = 1.0
        for col in LABEL_COLUMNS:
            probs = F.softmax(logits[col], dim=-1)[0].cpu().numpy()
            idx = int(probs.argmax())
            conf = float(probs[idx])
            min_conf = min(min_conf, conf)
            result[col] = {
                "label": self.index_to_label[col][idx],
                "confidence": conf,
                "probs": {self.index_to_label[col][i]: float(p)
                          for i, p in enumerate(probs)},
            }
        result["overall_confidence"] = min_conf
        result["needs_human_review"] = min_conf < self.threshold
        return result

    def predict_batch(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        return [self.predict_one(t) for t in texts]
