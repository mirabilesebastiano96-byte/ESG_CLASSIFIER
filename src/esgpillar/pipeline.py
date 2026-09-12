"""API di alto livello: :class:`ESGPillarClassifier`.

Incapsula tokenizzazione, embedding e rete in un unico oggetto con
``fit``/``predict``/``save``/``load``, per chi vuole usare la libreria
senza rientrare nei dettagli dei singoli moduli.

Tre rappresentazioni disponibili: ``"glove"`` e ``"fasttext"`` (statiche,
un vettore fisso per parola) ed ``"esgbert"`` (contestuale, FinBERT-ESG
frozen -- piu' lenta ma potenzialmente piu' espressiva, richiede l'extra
``transformers``).

Per replicare il disegno sperimentale completo (split per azienda,
confronto fra piu' embedding e architetture) vedi ``examples/`` nel
repository: quello resta un compito di ricerca, non un'unica chiamata
di libreria -- le singole fasi (:mod:`esgpillar.extraction`,
:mod:`esgpillar.cleaning`, :mod:`esgpillar.labeling`) restano comunque
disponibili separatamente per costruirselo su misura.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from .config import Config
from .tokenization import Tokenizer

EmbeddingName = Literal["glove", "fasttext", "esgbert"]
NetworkName = Literal["DNN", "Conv1D", "BiLSTM", "BiGRU"]

_STATIC_EMBEDDINGS = {"glove", "fasttext"}
_CONTEXTUAL_EMBEDDINGS = {"esgbert"}


class ESGPillarClassifier:
    """Classificatore di frasi ESG nei pilastri Environmental/Social/Governance.

    Esempio
    -------
    >>> from esgpillar import Config, ESGPillarClassifier
    >>> clf = ESGPillarClassifier(embedding="glove", network="BiLSTM")
    >>> clf.fit(df, text_col="sentence", company_col="company", label_col="label")
    >>> clf.predict(["We reduced carbon emissions by 30% in 2023."])
    ['Environmental']

    Con ``embedding="esgbert"`` si usa FinBERT-ESG come estrattore di
    feature contestuale, frozen (nessun fine-tuning):

    >>> clf = ESGPillarClassifier(embedding="esgbert", network="BiLSTM")
    """

    def __init__(self, config: Config | None = None,
                embedding: EmbeddingName = "glove", network: NetworkName = "BiLSTM"):
        self.config = config or Config()
        self.embedding_name = embedding
        self.network_name = network
        self._model = None
        self._encoder = None
        self._tokenizer: Tokenizer | None = None
        self._is_fitted = False

    # ------------------------------------------------------------------ #
    def _build_encoder(self, companies=(), dim: int | None = None, device: str = "cpu"):
        """Costruisce l'encoder giusto per ``self.embedding_name``.

        Le rappresentazioni statiche (glove/fasttext) usano il tokenizzatore
        condiviso a livello di parola; ESGBERT usa la propria tokenizzazione
        (WordPiece) e non ha bisogno del tokenizzatore condiviso.
        """
        from . import embeddings as _emb

        cfg = self.config
        if self.embedding_name in _STATIC_EMBEDDINGS:
            self._tokenizer = Tokenizer.from_companies(companies, generic_terms=cfg.generic_esg_terms)
            loader = _emb.load_glove if self.embedding_name == "glove" else _emb.load_fasttext_pretrained
            dim = dim or (cfg.glove_dim if self.embedding_name == "glove" else cfg.fasttext_pretrained_dim)
            kv = loader(cfg)
            return _emb.EmbeddingEncoder(kv=kv, dim=dim, tokenizer=self._tokenizer, max_len=cfg.max_len)

        if self.embedding_name in _CONTEXTUAL_EMBEDDINGS:
            self._tokenizer = None  # ESGBERT usa la propria tokenizzazione
            tok, model = _emb.load_esgbert(cfg)
            dim = dim or model.config.hidden_size
            return _emb.ContextualEmbeddingEncoder(
                tokenizer=tok, model=model, dim=dim, max_len=cfg.max_len,
                batch_size=cfg.esgbert_batch, device=device)

        raise ValueError(f"Rappresentazione non riconosciuta: {self.embedding_name!r}")

    def fit(self, df: pd.DataFrame, text_col: str = "sentence",
           company_col: str = "company", label_col: str = "label",
           val_size: float | None = None) -> "ESGPillarClassifier":
        """Addestra il classificatore.

        ``df`` deve avere le colonne di testo/azienda/etichetta indicate.
        Lo split di validazione interno e' fatto **per azienda** (mai la
        stessa azienda sia in fit sia in validation), coerente con il
        resto della libreria.
        """
        from .models import NET_BUILDERS
        from .training import get_device, train_model

        cfg = self.config
        val_size = val_size if val_size is not None else cfg.val_size
        device = get_device()

        self._encoder = self._build_encoder(df[company_col].dropna().unique(), device=device)
        dim = self._encoder.dim

        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=cfg.seed)
        fit_idx, val_idx = next(gss.split(df, df[label_col], groups=df[company_col]))
        fit_df = df.iloc[fit_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # se label_dataframe() ha applicato la mascheratura anti-leakage, usa
        # 'sentence_model' (testo mascherato) invece del testo grezzo per
        # costruire le rappresentazioni; altrimenti ripiega su text_col
        encode_col = "sentence_model" if "sentence_model" in df.columns else text_col
        self._encode_col_used = encode_col
        seq_f, pool_f, len_f = self._encoder(fit_df[encode_col].tolist())
        seq_v, pool_v, len_v = self._encoder(val_df[encode_col].tolist())
        y_f, y_v = fit_df[label_col].values, val_df[label_col].values

        NetClass = NET_BUILDERS[self.network_name]
        if self.network_name == "DNN":
            tri, vai = (self._to_t(pool_f),), (self._to_t(pool_v),)
        else:
            tri = (self._to_t(seq_f), self._to_t(len_f, long=True))
            vai = (self._to_t(seq_v), self._to_t(len_v, long=True))

        class_weights = compute_class_weight("balanced", classes=np.arange(len(cfg.class_names)), y=y_f)
        model, history = train_model(NetClass(d=dim), tri, y_f, vai, y_v, class_weights, cfg, device)
        self._model = model
        self._device = device
        self.history_ = history
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------ #
    def _to_t(self, arr, long: bool = False):
        import torch
        return torch.as_tensor(arr, dtype=torch.long if long else torch.float32)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Il classificatore non e' stato addestrato: chiama .fit(...) o .load(...) prima.")

    def predict_proba(self, sentences: list[str]) -> np.ndarray:
        """Ritorna le probabilita' per classe, shape (N, 3).

        Se il modello e' stato addestrato con mascheratura anti-leakage
        (comportamento di default, vedi ``Config.apply_leakage_mask``),
        la stessa mascheratura viene applicata qui alle frasi nuove, per
        coerenza tra training e inferenza.
        """
        from .training import predict_proba as _predict_proba

        self._check_fitted()
        if getattr(self, "_encode_col_used", None) == "sentence_model":
            from .labeling import mask_lexicon
            sentences = [mask_lexicon(s) for s in sentences]
        seq, pooled, lengths = self._encoder(sentences)
        inputs = (self._to_t(pooled),) if self.network_name == "DNN" else \
            (self._to_t(seq), self._to_t(lengths, long=True))
        return _predict_proba(self._model, inputs, self._device)

    def predict(self, sentences: list[str]) -> list[str]:
        """Ritorna il pilastro predetto (nome esteso) per ciascuna frase."""
        proba = self.predict_proba(sentences)
        idx = proba.argmax(axis=1)
        return [self.config.class_names[i] for i in idx]

    def evaluate(self, sentences: list[str], labels) -> dict:
        """Macro-F1 su un set etichettato (es. il test per azienda).

        Solo F1 (aggregata e per pilastro): e' la metrica che conta per un
        problema multi-classe sbilanciato come questo — accuracy/precision/
        recall aggregate erano state tolte anche dal notebook di riferimento.
        """
        proba = self.predict_proba(sentences)
        pred = proba.argmax(axis=1)
        f1_per_class = f1_score(labels, pred, average=None, zero_division=0)
        return {
            "macro_f1": f1_score(labels, pred, average="macro"),
            "f1_per_class": dict(zip(self.config.class_names, f1_per_class)),
        }

    def plot_history(self, title: str | None = None):
        """Grafico delle curve di training (loss + metrica di validazione).

        Disponibile solo dopo ``fit()`` (non dopo ``load()``, che non
        conserva la storia di un training gia' avvenuto). Richiede
        l'extra ``viz``.
        """
        if not hasattr(self, "history_"):
            raise RuntimeError(
                "Nessuna storia di training disponibile: chiama .fit(...) prima "
                "(un modello .load()-ato non conserva la storia dell'addestramento originale)."
            )
        from .plotting import plot_training_history
        title = title or f"Curva di training — {self.embedding_name} + {self.network_name}"
        return plot_training_history(self.history_, title=title)

    def plot_confusion_matrix(self, sentences: list[str], labels, normalize: bool = True):
        """Confusion matrix su un set etichettato. Richiede l'extra ``viz``."""
        from .plotting import plot_confusion_matrix
        proba = self.predict_proba(sentences)
        pred = proba.argmax(axis=1)
        return plot_confusion_matrix(labels, pred, class_names=self.config.class_names, normalize=normalize)

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        """Salva modello + metadati in un'unica cartella.

        L'embedding pre-addestrato NON viene risalvato (centinaia di MB, a
        volte oltre 1GB per ESGBERT): al ``load()`` viene ricaricato dalla
        cache locale (gensim o HuggingFace), gia' presente dopo il primo
        download.
        """
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path / "model.pt")
        meta = {
            "config": self.config,
            "embedding_name": self.embedding_name,
            "network_name": self.network_name,
            "dim": self._encoder.dim,
            "encode_col_used": getattr(self, "_encode_col_used", "sentence"),
        }
        if self.embedding_name in _STATIC_EMBEDDINGS:
            meta["company_tokens"] = list(self._tokenizer._company_tokens)
        with open(path / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        print(f"Salvato in {path}/ (model.pt, meta.pkl). "
              f"Nota: l'embedding pre-addestrato NON e' risalvato -- verra' "
              f"ricaricato dalla cache (gensim o HuggingFace) al momento del load().")

    @classmethod
    def load(cls, path: str | Path) -> "ESGPillarClassifier":
        """Ricarica un classificatore salvato con :meth:`save`."""
        import torch

        from .models import NET_BUILDERS
        from .training import get_device

        path = Path(path)
        with open(path / "meta.pkl", "rb") as f:
            meta = pickle.load(f)

        obj = cls(config=meta["config"], embedding=meta["embedding_name"], network=meta["network_name"])
        device = get_device()
        obj._encoder = obj._build_encoder(
            companies=meta.get("company_tokens", ()), dim=meta["dim"], device=device)
        if meta["embedding_name"] in _STATIC_EMBEDDINGS:
            obj._tokenizer = Tokenizer(company_tokens=meta["company_tokens"],
                                       generic_terms=meta["config"].generic_esg_terms)
            obj._encoder.tokenizer = obj._tokenizer

        NetClass = NET_BUILDERS[meta["network_name"]]
        model = NetClass(d=meta["dim"]).to(device)
        model.load_state_dict(torch.load(path / "model.pt", map_location=device))
        model.eval()
        obj._model = model
        obj._device = device
        obj._encode_col_used = meta.get("encode_col_used", "sentence")
        obj._is_fitted = True
        return obj
