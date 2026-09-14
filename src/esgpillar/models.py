"""Architetture neurali per la classificazione: DNN, Conv1D, BiLSTM, BiGRU,
Transformer encoder, rete con attention pooling.

Tutte accettano rappresentazioni pooled (DNN) o sequenziali (le altre)
della stessa dimensione ``d``, cosi' si innestano direttamente su
qualunque embedding prodotto da :mod:`esgpillar.embeddings`. Coprono
famiglie architetturali diverse: feed-forward (DNN), convoluzionale
locale (Conv1D), ricorrente (BiLSTM/BiGRU), attenzione pura
(Transformer, AttentionPool).

Richiede l'extra ``torch`` (``pip install esg-pillar-classifier[torch]``).
"""
from __future__ import annotations

import numpy as np

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


class TransformerEncoderNet(nn.Module):
    """Encoder Transformer (self-attention) su una sequenza di embedding per token.

    Famiglia architetturale diversa da Conv1D (locale) e BiLSTM/BiGRU
    (ricorrente): l'attenzione permette a ogni token di "vedere"
    direttamente ogni altro token della frase in un solo passo, senza il
    collo di bottiglia sequenziale delle RNN. Usa un encoding posizionale
    sinusoidale fisso (nessun parametro aggiuntivo da imparare) e mean
    pooling sull'output finale.
    """

    def __init__(self, d: int, c: int = 3, n_heads: int = 4, n_layers: int = 2,
                p: float = 0.4, max_len: int = 512):
        super().__init__()
        assert d % n_heads == 0, f"d={d} deve essere divisibile per n_heads={n_heads}"
        pe = torch.zeros(max_len, d)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=4 * d, dropout=p, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.drop = nn.Dropout(p)
        self.fc1 = nn.Linear(d, 128)
        self.fc2 = nn.Linear(128, c)

    def forward(self, x, lengths=None):  # x: (B, L, D)
        x = x + self.pe[:, : x.size(1)]
        h = self.encoder(x).mean(dim=1)
        h = self.drop(h)
        h = self.drop(torch.relu(self.fc1(h)))
        return self.fc2(h)


class AttentionPoolNet(nn.Module):
    """Rete leggera con attention pooling: un singolo layer di self-attention
    calcola un peso di importanza per ciascun token, poi combina i vettori
    di token in un'unica rappresentazione pesata (invece della media
    semplice usata dalla DNN). Meno espressiva di un Transformer completo
    o di una BiLSTM, ma molto piu' leggera e veloce da addestrare —
    un buon compromesso quando i dati sono pochi.
    """

    def __init__(self, d: int, c: int = 3, p: float = 0.4):
        super().__init__()
        self.attn = nn.Linear(d, 1)
        self.drop = nn.Dropout(p)
        self.fc1 = nn.Linear(d, 128)
        self.fc2 = nn.Linear(128, c)

    def forward(self, x, lengths=None):  # x: (B, L, D)
        weights = torch.softmax(self.attn(x).squeeze(-1), dim=1)  # (B, L)
        h = (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        h = self.drop(h)
        h = self.drop(torch.relu(self.fc1(h)))
        return self.fc2(h)


class GRUNet(nn.Module):
    """GRU MONO-direzionale (non bidirezionale come BiGRU) su una sequenza
    di embedding per token. Piu' semplice e leggera di BiGRU/BiLSTM (vede
    la frase in un solo verso, meta' dei parametri ricorrenti) -- un buon
    riferimento per misurare quanto la bidirezionalita' incida davvero
    sui risultati, a parita' di tutto il resto."""

    def __init__(self, d: int, c: int = 3, h: int = 128, p: float = 0.4):
        super().__init__()
        self.gru = nn.GRU(d, h, batch_first=True, bidirectional=False)
        self.drop = nn.Dropout(p)
        self.fc1 = nn.Linear(h, 128)
        self.fc2 = nn.Linear(128, c)

    def forward(self, x, lengths=None):  # x: (B, L, D)
        _, hn = self.gru(x)  # hn: (1, B, h) -- una sola direzione
        h = self.drop(hn[0])
        h = self.drop(torch.relu(self.fc1(h)))
        return self.fc2(h)


class ResidualConvBlock(nn.Module):
    """Un blocco Conv1D con connessione residua (skip connection) e
    normalizzazione, lo stesso principio delle reti ResNet applicato a
    sequenze di testo: l'input del blocco viene sommato al suo output,
    cosi' il gradiente puo' "saltare" il blocco durante il training --
    piu' facile da ottimizzare quando si impilano piu' blocchi in
    sequenza rispetto a una pila di Conv1D semplici."""

    def __init__(self, channels: int, kernel_size: int = 3, p: float = 0.2):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(p)

    def forward(self, x):  # x: (B, C, L)
        residual = x
        h = torch.relu(self.norm1(self.conv1(x)))
        h = self.drop(h)
        h = self.norm2(self.conv2(h))
        return torch.relu(h + residual)


class ResCNN(nn.Module):
    """CNN profonda con connessioni residue (piu' blocchi impilati),
    variante piu' moderna di :class:`ConvNet`: quest'ultima ha un solo
    strato di convoluzioni multi-kernel in parallelo, ResCNN impila piu'
    blocchi in profondita' con skip connection tra un blocco e l'altro."""

    def __init__(self, d: int, c: int = 3, n_blocks: int = 3, channels: int = 128,
                p: float = 0.4, proj_dim: int = 128):
        super().__init__()
        self.proj_dim = min(proj_dim, d)
        self.proj = nn.Linear(d, self.proj_dim) if self.proj_dim < d else None
        in_ch = self.proj_dim
        self.entry = nn.Conv1d(in_ch, channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualConvBlock(channels, p=p) for _ in range(n_blocks)])
        self.drop = nn.Dropout(p)
        self.fc1 = nn.Linear(channels, 128)
        self.fc2 = nn.Linear(128, c)

    def forward(self, x, lengths=None):  # x: (B, L, D)
        if self.proj is not None:
            x = self.proj(x)
        x = x.transpose(1, 2)  # (B, D, L)
        h = torch.relu(self.entry(x))
        for block in self.blocks:
            h = block(h)
        h = h.max(dim=2).values  # global max pooling sulla lunghezza
        h = self.drop(h)
        h = self.drop(torch.relu(self.fc1(h)))
        return self.fc2(h)


class MLPMixerNet(nn.Module):
    """MLP-Mixer applicato a sequenze di testo: niente attenzione (come nel
    Transformer) ne' ricorrenza (come nelle RNN) ne' convoluzione locale --
    solo strati di percettroni multi-livello (MLP) alternati, uno che
    "mescola" informazioni TRA i token della sequenza (token-mixing) e uno
    che mescola le feature ALL'INTERNO di ciascun token (channel-mixing).
    Architettura del 2021 (Tolstikhin et al.), originariamente per immagini,
    qui adattata a sequenze di embedding: piu' semplice ed economica di un
    Transformer completo, pur mantenendo un meccanismo di mescolamento
    globale tra token (a differenza di Conv1D, che resta locale)."""

    def __init__(self, d: int, c: int = 3, seq_len: int = 64, n_layers: int = 2,
                token_hidden: int = 64, channel_hidden: int = 256, p: float = 0.4):
        super().__init__()
        self.seq_len = seq_len
        self.token_mix = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(seq_len, token_hidden), nn.GELU(), nn.Dropout(p),
                nn.Linear(token_hidden, seq_len),
            ) for _ in range(n_layers)
        ])
        self.channel_mix = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, channel_hidden), nn.GELU(), nn.Dropout(p),
                nn.Linear(channel_hidden, d),
            ) for _ in range(n_layers)
        ])
        self.drop = nn.Dropout(p)
        self.fc1 = nn.Linear(d, 128)
        self.fc2 = nn.Linear(128, c)

    def forward(self, x, lengths=None):  # x: (B, L, D)
        b, seq_len, d = x.shape
        if seq_len < self.seq_len:
            x = nn.functional.pad(x, (0, 0, 0, self.seq_len - seq_len))
        elif seq_len > self.seq_len:
            x = x[:, : self.seq_len]

        for token_layer, channel_layer in zip(self.token_mix, self.channel_mix):
            # token-mixing: mescola lungo la dimensione della sequenza (L)
            y = token_layer[0](x).transpose(1, 2)  # (B, D, L) dopo LayerNorm
            y = token_layer[1:](y).transpose(1, 2)  # torna a (B, L, D)
            x = x + y
            # channel-mixing: mescola lungo la dimensione delle feature (D)
            x = x + channel_layer(x)

        h = x.mean(dim=1)  # mean pooling sulla sequenza
        h = self.drop(h)
        h = self.drop(torch.relu(self.fc1(h)))
        return self.fc2(h)


class RCNN(nn.Module):
    """Recurrent Convolutional Neural Network (Lai et al., 2015).

    Per ogni token, una BiLSTM calcola il contesto sinistro e destro;
    questi vengono concatenati con l'embedding del token stesso e passati
    attraverso un layer lineare + tanh (il "pseudo-kernel convoluzionale"
    dell'architettura originale), poi max-pooling sull'intera sequenza.
    L'idea: la ricorrenza cattura il contesto, il pooling successivo
    seleziona le feature piu' salienti -- un ibrido tra RNN e CNN, non
    una semplice somma delle due."""

    def __init__(self, d: int, c: int = 3, h: int = 128, conv_dim: int = 128, p: float = 0.4):
        super().__init__()
        self.lstm = nn.LSTM(d, h, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p)
        self.conv = nn.Linear(d + 2 * h, conv_dim)
        self.fc1 = nn.Linear(conv_dim, 128)
        self.fc2 = nn.Linear(128, c)

    def forward(self, x, lengths=None):  # x: (B, L, D)
        ctx, _ = self.lstm(x)  # (B, L, 2h): contesto sx+dx per ciascun token
        combined = torch.cat([x, ctx], dim=-1)  # (B, L, D + 2h)
        h = torch.tanh(self.conv(combined))  # (B, L, conv_dim)
        h = h.max(dim=1).values  # max-pooling sull'intera sequenza -> (B, conv_dim)
        h = self.drop(h)
        h = self.drop(torch.relu(self.fc1(h)))
        return self.fc2(h)


class DPCNN(nn.Module):
    """Deep Pyramid CNN (Johnson & Zhang, 2017).

    A differenza di :class:`ConvNet` (un solo strato di convoluzioni
    multi-kernel in parallelo) o di :class:`ResCNN` (piu' blocchi residui
    alla STESSA risoluzione), DPCNN dimezza la lunghezza della sequenza a
    ogni blocco successivo (pooling con stride 2) mentre approfondisce la
    rete -- una "piramide": pochi parametri per livello, ma un campo
    ricettivo che cresce esponenzialmente con la profondita', catturando
    dipendenze a lungo raggio senza il costo di un Transformer completo.
    Si ferma automaticamente quando la sequenza diventa troppo corta per
    un ulteriore dimezzamento (robusto a frasi brevi)."""

    def __init__(self, d: int, c: int = 3, channels: int = 128, max_blocks: int = 4,
                p: float = 0.4, proj_dim: int = 128):
        super().__init__()
        self.proj_dim = min(proj_dim, d)
        self.proj = nn.Linear(d, self.proj_dim) if self.proj_dim < d else None
        in_ch = self.proj_dim
        self.region_conv = nn.Conv1d(in_ch, channels, kernel_size=3, padding=1)
        self.max_blocks = max_blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(), nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(), nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            ) for _ in range(max_blocks)
        ])
        self.drop = nn.Dropout(p)
        self.fc1 = nn.Linear(channels, 128)
        self.fc2 = nn.Linear(128, c)

    def forward(self, x, lengths=None):  # x: (B, L, D)
        if self.proj is not None:
            x = self.proj(x)
        x = x.transpose(1, 2)  # (B, D, L)
        h = self.region_conv(x)  # embedding di regione iniziale

        for block in self.blocks:
            if h.size(-1) < 2:
                break  # sequenza troppo corta per un altro dimezzamento: si ferma qui
            residual = h
            h = block(h) + residual  # connessione residua, come nel paper originale
            h = nn.functional.max_pool1d(h, kernel_size=3, stride=2, padding=1)  # dimezza L

        h = h.max(dim=2).values  # pooling finale sulla lunghezza residua
        h = self.drop(h)
        h = self.drop(torch.relu(self.fc1(h)))
        return self.fc2(h)


NET_BUILDERS = {
    "DNN": DNN,
    "Conv1D": ConvNet,
    "BiLSTM": BiLSTM,
    "BiGRU": BiGRU,
    "Transformer": TransformerEncoderNet,
    "AttentionPool": AttentionPoolNet,
    "GRU": GRUNet,
    "ResCNN": ResCNN,
    "MLPMixer": MLPMixerNet,
    "RCNN": RCNN,
    "DPCNN": DPCNN,
}
