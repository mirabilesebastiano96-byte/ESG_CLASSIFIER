"""API di alto livello: :class:`ESGPillarClassifier`.

Incapsula tokenizzazione, embedding e rete in un unico oggetto con
``fit``/``predict``/``save``/``load``, per chi vuole usare la libreria
senza rientrare nei dettagli dei singoli moduli.

Quattordici rappresentazioni disponibili. Statiche pre-addestrate (un vettore
fisso per parola, da un corpus esterno generico): ``"glove"``,
``"fasttext"``. Statiche addestrate sul TUO corpus di training (vedi nota
sotto su save/load): ``"word2vec"`` (Skip-gram), ``"fasttext_corpus"``
(con vere subword), ``"lsa"`` (TF-IDF + SVD, per parola). Contestuali
frozen (un vettore diverso a seconda della frase, richiedono l'extra
``transformers``): ``"esgbert"`` (specifico ESG), ``"finbert"``
(finanziario generico), ``"climatebert"`` (specifico clima),
``"bert_base"`` (generico), ``"roberta"`` (generico, pre-training piu'
robusto), ``"umberto"`` (RoBERTa-based ma addestrato su corpus ITALIANI --
usarlo solo se il tuo corpus e' in italiano). Contestuale ma con la
propria libreria: ``"elmo"`` (richiede l'extra ``elmo``, che installa
TENSORFLOW oltre a PyTorch -- un costo di dipendenza reale; il modello
va anche scaricato A MANO, non automaticamente, vedi
:func:`esgpillar.embeddings.load_elmo`). A livello di frase intera:
``"doc2vec"`` (addestrato sul corpus) e ``"sbert"`` (Sentence-Transformers,
pre-addestrato apposta per frasi, richiede l'extra ``sbert``).

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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from .config import Config
from .tokenization import Tokenizer

EmbeddingName = Literal["glove", "fasttext", "word2vec", "lsa", "fasttext_corpus",
                        "esgbert", "finbert", "climatebert", "bert_base", "roberta", "umberto",
                        "doc2vec", "sbert", "elmo"]
NetworkName = Literal[
    # reti neurali (PyTorch, esgpillar.models)
    "DNN", "Conv1D", "BiLSTM", "BiGRU", "Transformer", "AttentionPool", "GRU", "ResCNN", "MLPMixer",
    "RCNN", "DPCNN",
    # modelli classici (scikit-learn/xgboost/lightgbm, esgpillar.classical) -- lavorano solo su vettori pooled
    "LogisticRegression", "LinearSVM", "SVM_RBF", "SGD", "Ridge", "Perceptron",
    "GaussianNB", "MultinomialNB", "ComplementNB", "BernoulliNB",
    "DecisionTree", "RandomForest", "ExtraTrees", "GradientBoosting", "AdaBoost", "Bagging",
    "XGBoost", "LightGBM", "KNN", "LDA", "QDA", "MLP_sklearn", "Voting", "Stacking",
]
# oltre ai nomi sopra, 'network' accetta anche un'architettura costruita a mano:
# una CLASSE torch.nn.Module (rete personalizzata, vedi ESGPillarClassifier.__init__)
# o un'ISTANZA scikit-learn-compatibile (modello classico personalizzato).

_PRETRAINED_STATIC_EMBEDDINGS = {"glove", "fasttext"}
_CORPUS_STATIC_EMBEDDINGS = {"word2vec", "lsa", "fasttext_corpus"}  # addestrate sul corpus, parola per parola
_STATIC_EMBEDDINGS = _PRETRAINED_STATIC_EMBEDDINGS | _CORPUS_STATIC_EMBEDDINGS
_CONTEXTUAL_EMBEDDINGS = {"esgbert", "finbert", "climatebert", "bert_base", "roberta", "umberto"}
_CONTEXTUAL_LOADERS = {
    "esgbert": "load_esgbert", "finbert": "load_finbert",
    "climatebert": "load_climatebert", "bert_base": "load_bert_base",
    "roberta": "load_roberta", "umberto": "load_umberto",
}
_SENTENCE_LEVEL_EMBEDDINGS = {"doc2vec", "sbert"}  # un vettore per frase, non per parola
_ELMO_EMBEDDING = {"elmo"}  # contestuale come esgbert/finbert/..., ma con la propria libreria (TensorFlow)
_REQUIRES_TRAINING_TEXTS = _CORPUS_STATIC_EMBEDDINGS | {"doc2vec"}  # nessuna versione pre-addestrata


def _classical_networks() -> set[str]:
    """Nomi dei classificatori classici (import lazy: evita di caricare
    scikit-learn/xgboost/lightgbm quando non servono)."""
    from .classical import CLASSICAL_BUILDERS
    return set(CLASSICAL_BUILDERS)


def _net_kwargs(network_name: str, dim: int, cfg: Config) -> dict:
    """Argomenti extra per costruire la rete, oltre a ``d=dim`` (comune a
    tutte). Solo MLPMixer ne ha bisogno: ``seq_len`` deve combaciare con
    ``cfg.max_len`` usato in fase di encoding."""
    if network_name == "MLPMixer":
        return {"d": dim, "seq_len": cfg.max_len}
    return {"d": dim}


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
                embedding: EmbeddingName = "glove", network: "NetworkName | type | object" = "BiLSTM",
                network_kwargs: dict | None = None,
                custom_input_type: Literal["pooled", "sequence"] | None = None):
        """
        ``network_kwargs``: iperparametri PERSONALIZZATI per la rete/il
        modello scelto in ``network``, al posto dei valori di default
        della relativa classe -- es. ``network_kwargs={"p": 0.5, "h": 256}``
        per una BiLSTM (dropout e dimensione nascosta), o
        ``network_kwargs={"n_estimators": 500, "max_depth": 10}`` per una
        RandomForest. Le chiavi devono corrispondere ai parametri accettati
        dal costruttore della classe scelta (vedi :mod:`esgpillar.models`
        per le reti neurali, :mod:`esgpillar.classical` per i modelli
        classici). Se omesso (default), si usano i valori di default della
        classe, come prima.

        ``network``: oltre ai nomi predefiniti (stringa), puo' essere
        un'architettura COSTRUITA A MANO:

        - una CLASSE PyTorch (sottoclasse di ``torch.nn.Module``), per una
          rete neurale personalizzata -- verra' istanziata al momento del
          ``fit()`` come ``TuaClasse(d=dim, **network_kwargs)``, con
          ``dim`` la dimensione dell'embedding scelto (deducibile solo a
          runtime, non prima). Deve seguire la stessa convenzione delle
          reti native: ``__init__(self, d, c=3, ...)`` (``d``=dimensione
          embedding, ``c``=numero di classi, sempre 3 per questa libreria)
          e ``forward(self, x)`` (se ``custom_input_type="pooled"``) o
          ``forward(self, x, lengths=None)`` (se ``custom_input_type="sequence"``).
          **``custom_input_type`` e' OBBLIGATORIO in questo caso**: dice
          alla libreria se passare alla rete il vettore mediato per frase
          (``"pooled"``, come la DNN nativa) o l'intera sequenza di
          token (``"sequence"``, come BiLSTM/Conv1D/... native) -- senza
          questa informazione non c'e' modo di sapere quale input la tua
          rete si aspetti.
        - un'ISTANZA gia' costruita, compatibile con l'interfaccia
          scikit-learn (``.fit(X, y)`` / ``.predict_proba(X)``), per un
          modello classico personalizzato non gia' in
          :mod:`esgpillar.classical` -- usata cosi' com'e', SEMPRE su
          vettori pooled (i modelli classici in questa libreria non
          lavorano mai su sequenze).

        ``custom_input_type``: vedi sopra -- richiesto (e usato) solo
        quando ``network`` e' una CLASSE PyTorch personalizzata; ignorato
        altrimenti.
        """
        import inspect

        self.config = config or Config()
        self.embedding_name = embedding
        self.network_name = network
        self.network_kwargs = dict(network_kwargs) if network_kwargs else {}
        self.custom_input_type = custom_input_type
        self._model = None
        self._encoder = None
        self._tokenizer: Tokenizer | None = None
        self._is_fitted = False

        if inspect.isclass(network):
            try:
                import torch.nn as nn
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "Una rete PyTorch personalizzata richiede l'extra 'torch': "
                    "pip install esg-pillar-classifier[torch]"
                ) from e
            if not issubclass(network, nn.Module):
                raise TypeError(
                    f"network={network!r} e' una classe ma non una sottoclasse di "
                    "torch.nn.Module -- per un modello classico personalizzato, passa "
                    "un'ISTANZA gia' costruita (con .fit()/.predict_proba()), non una classe."
                )
            if custom_input_type not in ("pooled", "sequence"):
                raise ValueError(
                    "network e' una classe PyTorch personalizzata: serve "
                    "custom_input_type='pooled' o custom_input_type='sequence' "
                    "(vedi la docstring di __init__ per la differenza)."
                )

    # ------------------------------------------------------------------ #
    def _is_custom_network(self) -> bool:
        return not isinstance(self.network_name, str)

    def _is_custom_neural_class(self) -> bool:
        import inspect
        return inspect.isclass(self.network_name)

    def _network_display_name(self) -> str:
        if isinstance(self.network_name, str):
            return self.network_name
        if self._is_custom_neural_class():
            return self.network_name.__name__
        return type(self.network_name).__name__

    def _uses_pooled_input(self) -> bool:
        """Vero se il modello scelto va alimentato con il vettore pooled
        (media per frase) invece che con l'intera sequenza di token --
        vale per la DNN nativa, per TUTTI i modelli classici (nativi o
        personalizzati), e per una rete personalizzata dichiarata
        ``custom_input_type="pooled"``."""
        if isinstance(self.network_name, str):
            return self.network_name == "DNN" or self.network_name in _classical_networks()
        if self._is_custom_neural_class():
            return self.custom_input_type == "pooled"
        return True  # istanza classica personalizzata: sempre pooled

    # ------------------------------------------------------------------ #
    def _build_encoder(self, companies=(), texts=None, corpus_kv=None,
                       corpus_model=None, dim: int | None = None, device: str = "cpu"):
        """Costruisce l'encoder giusto per ``self.embedding_name``.

        Le rappresentazioni statiche (glove/fasttext/word2vec/lsa/
        fasttext_corpus) usano il tokenizzatore condiviso a livello di
        parola; quelle contestuali (esgbert/finbert/...) usano la propria
        tokenizzazione; quelle a livello di frase (doc2vec/sbert) danno un
        vettore per frase intera.

        ``texts``: richiesto per le rappresentazioni addestrate sul corpus
        (word2vec/lsa/fasttext_corpus/doc2vec) in fase di ``fit()`` --
        servono per addestrare i vettori, non esiste una versione
        pre-addestrata di queste.
        ``corpus_kv`` / ``corpus_model``: usati da ``load()`` per
        ricaricare i vettori/il modello gia' addestrati (salvati con
        ``save()``), senza bisogno del corpus originale.
        """
        from . import embeddings as _emb

        cfg = self.config
        if self.embedding_name in _STATIC_EMBEDDINGS:
            self._tokenizer = Tokenizer.from_companies(companies, generic_terms=cfg.generic_esg_terms, normalization=cfg.tokenizer_normalization)

            if self.embedding_name in _CORPUS_STATIC_EMBEDDINGS:
                if corpus_kv is not None:
                    kv = corpus_kv
                elif texts is not None:
                    token_lists = [self._tokenizer(t) for t in texts]
                    loader = {
                        "word2vec": _emb.load_word2vec,
                        "lsa": _emb.load_lsa,
                        "fasttext_corpus": _emb.load_fasttext_corpus,
                    }[self.embedding_name]
                    kv = loader(token_lists, cfg)
                else:
                    raise ValueError(
                        f"embedding={self.embedding_name!r} e' addestrato sul corpus: serve "
                        "'texts' (durante fit()) o 'corpus_kv' (durante load())."
                    )
                dim = dim or {
                    "word2vec": cfg.word2vec_dim, "lsa": cfg.lsa_dim,
                    "fasttext_corpus": cfg.fasttext_corpus_dim,
                }[self.embedding_name]
                self._corpus_kv = kv  # conservato per il save()
            else:
                loader = _emb.load_glove if self.embedding_name == "glove" else _emb.load_fasttext_pretrained
                dim = dim or (cfg.glove_dim if self.embedding_name == "glove" else cfg.fasttext_pretrained_dim)
                kv = loader(cfg)

            return _emb.EmbeddingEncoder(kv=kv, dim=dim, tokenizer=self._tokenizer, max_len=cfg.max_len)

        if self.embedding_name in _CONTEXTUAL_EMBEDDINGS:
            self._tokenizer = None  # i modelli contestuali usano la propria tokenizzazione
            loader = getattr(_emb, _CONTEXTUAL_LOADERS[self.embedding_name])
            tok, model = loader(cfg)
            dim = dim or model.config.hidden_size
            return _emb.ContextualEmbeddingEncoder(
                tokenizer=tok, model=model, dim=dim, max_len=cfg.max_len,
                batch_size=cfg.esgbert_batch, device=device)

        if self.embedding_name in _ELMO_EMBEDDING:
            # ELMo usa il tokenizzatore condiviso (parole gia' divise), come le
            # rappresentazioni statiche -- ma i vettori dipendono dal contesto
            self._tokenizer = Tokenizer.from_companies(companies, generic_terms=cfg.generic_esg_terms, normalization=cfg.tokenizer_normalization)
            model = _emb.load_elmo(cfg)
            dim = dim or cfg.elmo_dim
            return _emb.ElmoEncoder(model=model, tokenizer=self._tokenizer, dim=dim, max_len=cfg.max_len)

        if self.embedding_name in _SENTENCE_LEVEL_EMBEDDINGS:
            if self.embedding_name == "doc2vec":
                self._tokenizer = Tokenizer.from_companies(companies, generic_terms=cfg.generic_esg_terms, normalization=cfg.tokenizer_normalization)
                if corpus_model is not None:
                    model = corpus_model
                elif texts is not None:
                    token_lists = [self._tokenizer(t) for t in texts]
                    model = _emb.load_doc2vec(token_lists, cfg)
                else:
                    raise ValueError(
                        "embedding='doc2vec' e' addestrato sul corpus: serve "
                        "'texts' (durante fit()) o 'corpus_model' (durante load())."
                    )
                self._corpus_model = model  # conservato per il save()
                dim = dim or cfg.doc2vec_dim
                return _emb.Doc2VecEncoder(model=model, dim=dim, tokenizer=self._tokenizer, max_len=cfg.max_len)

            # sbert: pre-addestrato, nessun bisogno del corpus di training
            self._tokenizer = None
            model = _emb.load_sentence_transformer(cfg)
            return _emb.SentenceTransformerEncoder(model=model, max_len=cfg.max_len)

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

        # se label_dataframe() ha applicato la mascheratura anti-leakage, usa
        # 'sentence_model' (testo mascherato) invece del testo grezzo per
        # costruire le rappresentazioni; altrimenti ripiega su text_col
        encode_col = "sentence_model" if "sentence_model" in df.columns else text_col
        self._encode_col_used = encode_col

        # word2vec/lsa/fasttext_corpus/doc2vec vengono addestrate qui, sull'INTERO
        # corpus di training (non solo la parte di fit interna): serve il testo
        texts_for_corpus_embedding = df[encode_col].tolist() if self.embedding_name in _REQUIRES_TRAINING_TEXTS else None
        self._encoder = self._build_encoder(
            df[company_col].dropna().unique(), texts=texts_for_corpus_embedding, device=device)
        dim = self._encoder.dim

        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=cfg.seed)
        fit_idx, val_idx = next(gss.split(df, df[label_col], groups=df[company_col]))
        fit_df = df.iloc[fit_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        seq_f, pool_f, len_f = self._encoder(fit_df[encode_col].tolist())
        seq_v, pool_v, len_v = self._encoder(val_df[encode_col].tolist())
        y_f, y_v = fit_df[label_col].values, val_df[label_col].values

        if self._is_custom_network():
            if self._is_custom_neural_class():
                # rete PyTorch personalizzata: stessa strada delle reti native,
                # solo con la classe fornita dall'utente invece che da NET_BUILDERS
                if self._uses_pooled_input():
                    tri, vai = (self._to_t(pool_f),), (self._to_t(pool_v),)
                else:
                    tri = (self._to_t(seq_f), self._to_t(len_f, long=True))
                    vai = (self._to_t(seq_v), self._to_t(len_v, long=True))
                class_weights = compute_class_weight("balanced", classes=np.arange(len(cfg.class_names)), y=y_f)
                model, history = train_model(
                    self.network_name(d=dim, **self.network_kwargs), tri, y_f, vai, y_v, class_weights, cfg, device)
                self._model = model
                self._device = device
                self.history_ = history
                self._is_classical = False
                self._is_fitted = True
                return self
            else:
                # istanza classica personalizzata (scikit-learn-compatibile): usata
                # cosi' com'e', nessuna istanziazione -- niente da fare con network_kwargs
                # (sono gia' impostati dall'utente al momento in cui ha costruito l'istanza)
                model = self.network_name
                model.fit(pool_f, y_f)
                self._model = model
                self._device = None
                self.history_ = None
                self._is_classical = True
                self._is_fitted = True
                return self

        if self.network_name in _classical_networks():
            # modelli classici (Logistic Regression, Naive Bayes, SVM, alberi,
            # ensemble, ...): lavorano solo su vettori pooled, niente sequenza,
            # niente training epoca-per-epoca -- un semplice .fit(X, y) sklearn.
            from .classical import build_classical_model

            model = build_classical_model(self.network_name, cfg, **self.network_kwargs)
            model.fit(pool_f, y_f)
            self._model = model
            self._device = None
            self.history_ = None  # nessuna curva di training per un modello classico
            self._is_classical = True
            self._is_fitted = True
            return self

        self._is_classical = False
        NetClass = NET_BUILDERS[self.network_name]
        if self._uses_pooled_input():
            tri, vai = (self._to_t(pool_f),), (self._to_t(pool_v),)
        else:
            tri = (self._to_t(seq_f), self._to_t(len_f, long=True))
            vai = (self._to_t(seq_v), self._to_t(len_v, long=True))

        class_weights = compute_class_weight("balanced", classes=np.arange(len(cfg.class_names)), y=y_f)
        model, history = train_model(
            NetClass(**{**_net_kwargs(self.network_name, dim, cfg), **self.network_kwargs}),
            tri, y_f, vai, y_v, class_weights, cfg, device)
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

        if getattr(self, "_is_classical", False):
            return self._model.predict_proba(pooled)

        inputs = (self._to_t(pooled),) if self._uses_pooled_input() else \
            (self._to_t(seq), self._to_t(lengths, long=True))
        return _predict_proba(self._model, inputs, self._device)

    def predict(self, sentences: list[str]) -> list[str]:
        """Ritorna il pilastro predetto (nome esteso) per ciascuna frase."""
        proba = self.predict_proba(sentences)
        idx = proba.argmax(axis=1)
        return [self.config.class_names[i] for i in idx]

    def evaluate(self, sentences: list[str], labels) -> dict:
        """Report di valutazione COMPLETO su un set etichettato (es. il
        test per azienda) -- sullo stile di ``sklearn.classification_report``:
        precision, recall, F1-score e support per CIASCUNA classe, piu'
        l'accuracy globale e le medie aggregate (macro e pesata).

        Ritorna un dict strutturato cosi':

        - ``report["Environmental"]``, ``report["Social"]``,
          ``report["Governance"]``: dict con ``precision``, ``recall``,
          ``f1-score``, ``support`` (n. di frasi vere di quella classe nel
          set valutato) per la singola classe.
        - ``report["accuracy"]``: accuracy globale (un singolo numero,
          non un dict).
        - ``report["macro avg"]``: media SEMPLICE delle tre classi (ogni
          classe pesa uguale, indipendentemente da quante frasi ha --
          utile per non far dominare la valutazione dalle classi piu'
          numerose).
        - ``report["weighted avg"]``: media pesata per il support di
          ciascuna classe (piu' vicina all'accuracy globale su classi
          sbilanciate).

        Esempio: ``evaluate(...)["Social"]["recall"]`` da' il recall sulla
        sola classe Social.
        """
        from sklearn.metrics import classification_report

        proba = self.predict_proba(sentences)
        pred = proba.argmax(axis=1)
        # 'labels' esplicito (non lasciato dedurre dai dati): se una classe non
        # compare ne' nelle etichette vere ne' nelle predizioni di QUESTO batch
        # (capita facilmente su un sottoinsieme piccolo, es. un test per singola
        # azienda), classification_report altrimenti solleva un ValueError invece
        # di riportare semplicemente support=0 per quella classe.
        return classification_report(
            labels, pred, labels=list(range(len(self.config.class_names))),
            target_names=self.config.class_names, output_dict=True, zero_division=0,
        )

    def plot_history(self, title: str | None = None):
        """Grafico delle curve di training (loss + metrica di validazione).

        Disponibile solo dopo ``fit()`` con una rete NEURALE (non dopo
        ``load()``, che non conserva la storia di un training gia'
        avvenuto, e non per i modelli classici in :mod:`esgpillar.classical`,
        che non hanno un training epoca-per-epoca). Richiede l'extra ``viz``.
        """
        if not hasattr(self, "history_") or self.history_ is None:
            reason = (
                "e' un modello classico (Logistic Regression, Naive Bayes, ...): "
                "nessun training epoca-per-epoca, quindi nessuna curva da mostrare."
                if getattr(self, "_is_classical", False) else
                "chiama .fit(...) prima (un modello .load()-ato non conserva "
                "la storia dell'addestramento originale)."
            )
            raise RuntimeError(f"Nessuna storia di training disponibile: {reason}")
        from .plotting import plot_training_history
        title = title or f"Curva di training — {self.embedding_name} + {self._network_display_name()}"
        return plot_training_history(self.history_, title=title)

    def plot_confusion_matrix(self, sentences: list[str], labels, normalize: bool = True):
        """Confusion matrix su un set etichettato. Richiede l'extra ``viz``."""
        from .plotting import plot_confusion_matrix
        proba = self.predict_proba(sentences)
        pred = proba.argmax(axis=1)
        return plot_confusion_matrix(labels, pred, class_names=self.config.class_names, normalize=normalize)

    def plot_embeddings(self, sentences: list[str], labels, method: str = "pca"):
        """Proietta in 2D (PCA o t-SNE) i vettori pooled prodotti dall'encoder
        gia' addestrato, colorati per pilastro. Richiede l'extra ``viz``."""
        from .plotting import plot_embeddings_2d
        _, pooled, _ = self._encoder(sentences)
        return plot_embeddings_2d(pooled, labels, class_names=self.config.class_names, method=method)

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        """Salva modello + metadati in un'unica cartella.

        L'embedding PRE-ADDESTRATO (glove/fasttext/esgbert/finbert/
        climatebert/bert_base/sbert) NON viene risalvato (centinaia di MB,
        a volte oltre 1GB): al ``load()`` viene ricaricato dalla cache
        locale (gensim o HuggingFace), gia' presente dopo il primo
        download. Le rappresentazioni ADDESTRATE SUL CORPUS (word2vec/lsa/
        fasttext_corpus/doc2vec) sono invece un'eccezione: non esiste una
        fonte esterna da cui riscaricarle, quindi vengono salvate per
        intero insieme al modello.
        """
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if getattr(self, "_is_classical", False):
            # un modello classico (sklearn) non ha state_dict(): si salva per intero con pickle
            with open(path / "classical_model.pkl", "wb") as f:
                pickle.dump(self._model, f)
        else:
            torch.save(self._model.state_dict(), path / "model.pt")
        meta = {
            "config": self.config,
            "embedding_name": self.embedding_name,
            "network_name": self.network_name,
            "network_kwargs": self.network_kwargs,
            "custom_input_type": self.custom_input_type,
            "dim": self._encoder.dim,
            "encode_col_used": getattr(self, "_encode_col_used", "sentence"),
            "is_classical": getattr(self, "_is_classical", False),
        }
        if self.embedding_name in _STATIC_EMBEDDINGS or self.embedding_name in ("doc2vec", "elmo"):
            meta["company_tokens"] = list(self._tokenizer._company_tokens)
        extra_note = ""
        if self.embedding_name in _CORPUS_STATIC_EMBEDDINGS:
            # NON usare dict(self._corpus_kv): su un vero gensim KeyedVectors
            # (word2vec) fallisce (ValueError) -- serve .key_to_index + indicizzazione,
            # l'unica interfaccia comune sia a KeyedVectors sia a _LSAKeyedVectors.
            kv_as_dict = {w: self._corpus_kv[w] for w in self._corpus_kv.key_to_index}
            with open(path / "corpus_kv.pkl", "wb") as f:
                pickle.dump(kv_as_dict, f)
            extra_note = " + corpus_kv.pkl (vettori, non riscaricabili)"
        elif self.embedding_name == "doc2vec":
            # Doc2Vec e' un modello vero e proprio (con la propria logica di
            # inferenza), non un semplice dizionario parola->vettore: usa il
            # meccanismo di salvataggio nativo di gensim, non pickle diretto.
            self._corpus_model.save(str(path / "doc2vec.model"))
            extra_note = " + doc2vec.model (non riscaricabile)"
        with open(path / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        model_file = "classical_model.pkl" if getattr(self, "_is_classical", False) else "model.pt"
        print(f"Salvato in {path}/ ({model_file}, meta.pkl{extra_note}). "
              f"Nota: gli embedding pre-addestrati (non quelli sul corpus) NON sono "
              f"risalvati -- verranno ricaricati dalla cache al momento del load().")

    @classmethod
    def load(cls, path: str | Path) -> "ESGPillarClassifier":
        """Ricarica un classificatore salvato con :meth:`save`."""
        import torch

        from .models import NET_BUILDERS
        from .training import get_device

        path = Path(path)
        with open(path / "meta.pkl", "rb") as f:
            meta = pickle.load(f)

        obj = cls(config=meta["config"], embedding=meta["embedding_name"], network=meta["network_name"],
                 network_kwargs=meta.get("network_kwargs"), custom_input_type=meta.get("custom_input_type"))
        device = get_device()

        corpus_kv = None
        corpus_model = None
        if meta["embedding_name"] in _CORPUS_STATIC_EMBEDDINGS:
            from .embeddings import _LSAKeyedVectors  # riusata come wrapper generico dict -> KeyedVectors-like

            with open(path / "corpus_kv.pkl", "rb") as f:
                corpus_kv = _LSAKeyedVectors(pickle.load(f))
        elif meta["embedding_name"] == "doc2vec":
            from gensim.models.doc2vec import Doc2Vec

            corpus_model = Doc2Vec.load(str(path / "doc2vec.model"))

        obj._encoder = obj._build_encoder(
            companies=meta.get("company_tokens", ()), corpus_kv=corpus_kv, corpus_model=corpus_model,
            dim=meta["dim"], device=device)
        if meta["embedding_name"] in _STATIC_EMBEDDINGS or meta["embedding_name"] in ("doc2vec", "elmo"):
            obj._tokenizer = Tokenizer(company_tokens=meta["company_tokens"],
                                       generic_terms=meta["config"].generic_esg_terms,
                                       normalization=meta["config"].tokenizer_normalization)
            obj._encoder.tokenizer = obj._tokenizer

        if meta.get("is_classical", False):
            with open(path / "classical_model.pkl", "rb") as f:
                obj._model = pickle.load(f)
            obj._device = None
            obj._is_classical = True
            obj._encode_col_used = meta.get("encode_col_used", "sentence")
            obj._is_fitted = True
            return obj

        NetClass = meta["network_name"] if not isinstance(meta["network_name"], str) else NET_BUILDERS[meta["network_name"]]
        net_kwargs = {**_net_kwargs(meta["network_name"], meta["dim"], meta["config"]), **obj.network_kwargs} \
            if isinstance(meta["network_name"], str) else {"d": meta["dim"], **obj.network_kwargs}
        model = NetClass(**net_kwargs).to(device)
        model.load_state_dict(torch.load(path / "model.pt", map_location=device))
        model.eval()
        obj._model = model
        obj._device = device
        obj._is_classical = False
        obj._encode_col_used = meta.get("encode_col_used", "sentence")
        obj._is_fitted = True
        return obj
