"""Architetture neurali per la classificazione: DNN, Conv1D, BiLSTM, BiGRU.

Tutte accettano rappresentazioni pooled (DNN) o sequenziali
(Conv1D/BiLSTM/BiGRU) della stessa dimensione ``d``, cosi' si innestano
direttamente su qualunque embedding prodotto da :mod:`esgpillar.embeddings`.

Richiede l'extra ``torch`` (``pip install esg-pillar-classifier[torch]``).
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "esgpillar.models richiede PyTorch. Installa con: "
        "pip install esg-pillar-classifier[torch]"
    ) from e


class DNN(nn.Module):
    """MLP su un vettore pooled per frase."""

    def __init__(self, d: int, c: int = 3, p: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.Dropout(p),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, c),
        )

    def forward(self, x):  # x: (B, D)
        return self.net(x)


class ConvNet(nn.Module):
    """Conv1D multi-kernel su una sequenza di embedding per token."""

    def __init__(self, d: int, c: int = 3, ks=(2, 3, 4), nf: int = 128, p: float = 0.4):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(d, nf, k, padding=k // 2) for k in ks])
        self.drop = nn.Dropout(p)
        self.fc1 = nn.Linear(nf * len(ks), 128)
        self.fc2 = nn.Linear(128, c)

    def forward(self, x, lengths=None):  # x: (B, L, D)
        x = x.transpose(1, 2)  # (B, D, L)
        outs = [torch.relu(conv(x)).max(dim=2).values for conv in self.convs]
        h = self.drop(torch.cat(outs, dim=1))
        h = self.drop(torch.relu(self.fc1(h)))
        return self.fc2(h)


class BiLSTM(nn.Module):
    """BiLSTM su una sequenza di embedding per token."""

    def __init__(self, d: int, c: int = 3, h: int = 128, p: float = 0.4):
        super().__init__()
        self.lstm = nn.LSTM(d, h, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p)
        self.fc1 = nn.Linear(2 * h, 128)
        self.fc2 = nn.Linear(128, c)

    def forward(self, x, lengths):  # x: (B, L, D), lengths: (B,)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        h = torch.cat([hn[0], hn[1]], dim=1)
        h = self.drop(h)
        h = self.drop(torch.relu(self.fc1(h)))
        return self.fc2(h)


class BiGRU(nn.Module):
    """BiGRU (GRU bidirezionale) su una sequenza di embedding per token.

    Stessa architettura della BiLSTM (bidirezionale, stesso schema di
    pooling finale), solo con celle GRU al posto di LSTM: confronto
    pulito sul solo tipo di cella ricorrente, senza altre variabili
    confondenti. La GRU ha meno parametri della LSTM (niente cell state
    separato, un gate in meno) -- spesso comparabile in accuratezza ma
    piu' veloce da addestrare.
    """

    def __init__(self, d: int, c: int = 3, h: int = 128, p: float = 0.4):
        super().__init__()
        self.gru = nn.GRU(d, h, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p)
        self.fc1 = nn.Linear(2 * h, 128)
        self.fc2 = nn.Linear(128, c)

    def forward(self, x, lengths):  # x: (B, L, D), lengths: (B,)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hn = self.gru(packed)  # hn: (2, B, h) -- niente cell state, a differenza della LSTM
        h = torch.cat([hn[0], hn[1]], dim=1)
        h = self.drop(h)
        h = self.drop(torch.relu(self.fc1(h)))
        return self.fc2(h)


NET_BUILDERS = {"DNN": DNN, "Conv1D": ConvNet, "BiLSTM": BiLSTM, "BiGRU": BiGRU}
