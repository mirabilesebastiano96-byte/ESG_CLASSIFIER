"""Training loop condiviso per DNN/Conv1D/BiLSTM, con:

- gradient clipping (evita aggiornamenti bruschi tra un'epoca e l'altra)
- scheduler ``ReduceLROnPlateau`` (dimezza il learning rate quando la
  validation loss ristagna, per una curva piu' stabile nelle epoche finali)
- tracciamento per-epoca di train loss, validation loss e validation
  macro-F1 (utile per diagnosticare overfitting, non solo guardare il
  numero finale)
- early stopping sul minimo della validation loss, con ripristino dei
  pesi migliori a fine training

Richiede l'extra ``torch``.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "esgpillar.training richiede PyTorch. Installa con: "
        "pip install esg-pillar-classifier[torch]"
    ) from e

import numpy as np
from sklearn.metrics import f1_score

from .config import Config


def to_t(a, dtype=None):
    dtype = dtype or torch.float32
    return torch.as_tensor(a, dtype=dtype)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Rileva il device migliore disponibile: CUDA > MPS (Mac Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_model(model, tr_inputs: Sequence, ytr, va_inputs: Sequence, yva, class_weights,
                config: Config | None = None, device: str | None = None) -> tuple[Any, dict]:
    """Addestra ``model`` con early stopping sulla validation loss.

    Ritorna ``(model, history)`` dove ``history`` ha le chiavi
    ``train_loss``, ``val_loss``, ``val_f1``, ``lr`` (una voce per epoca).
    """
    config = config or Config()
    device = device or get_device()
    set_seed(config.seed)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    crit = nn.CrossEntropyLoss(weight=to_t(class_weights).to(device))

    ytr_t = torch.as_tensor(np.asarray(ytr).copy(), dtype=torch.long)
    yva_t = torch.as_tensor(np.asarray(yva).copy(), dtype=torch.long, device=device)
    n = len(ytr)
    best, best_state, bad = float("inf"), None, 0
    va_dev = [t.to(device) for t in va_inputs]
    history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_f1": [], "lr": []}

    for _ in range(config.epochs):
        model.train()
        perm = torch.randperm(n)
        running_loss, n_batches = 0.0, 0
        for i in range(0, n, config.batch_size):
            idx = perm[i:i + config.batch_size]
            batch = [t[idx].to(device) for t in tr_inputs]
            opt.zero_grad()
            loss = crit(model(*batch), ytr_t[idx].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            opt.step()
            running_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            vout = model(*va_dev)
            vloss = crit(vout, yva_t).item()
            vf1 = f1_score(yva, vout.argmax(1).cpu().numpy(), average="macro", zero_division=0)
        sched.step(vloss)

        history["train_loss"].append(running_loss / max(n_batches, 1))
        history["val_loss"].append(vloss)
        history["val_f1"].append(vf1)
        history["lr"].append(opt.param_groups[0]["lr"])

        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict_proba(model, inputs: Sequence, device: str | None = None) -> np.ndarray:
    device = device or get_device()
    model.eval()
    logits = model(*[t.to(device) for t in inputs])
    return torch.softmax(logits, dim=1).cpu().numpy()
