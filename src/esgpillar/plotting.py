"""Grafici diagnostici: curve di training, confronto tra modelli,
confusion matrix, curve ROC.

Tutte le funzioni ritornano la ``Figure`` di matplotlib (non chiamano
``plt.show()``): puoi visualizzarla in un notebook, salvarla con
``fig.savefig(...)``, o incorporarla in un'app.

Richiede l'extra ``viz`` (``pip install esg-pillar-classifier[viz]``).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _smooth(y, window: int = 3):
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return y
    w = min(window, len(y))
    pad = w // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(y_pad, np.ones(w) / w, mode="valid")[: len(y)]


def plot_training_history(history: Mapping[str, Sequence[float]], title: str = "Curva di training"):
    """Traccia train/validation loss (asse sinistro) e la metrica di
    validazione per epoca -- accuracy o F1, quella disponibile in
    ``history`` (asse destro).

    Parameters
    ----------
    history : dict
        Il dizionario restituito da :func:`esgpillar.training.train_model`
        come secondo elemento della tupla, o ``clf.history_`` dopo
        ``ESGPillarClassifier.fit()``. Atteso con le chiavi
        ``train_loss``, ``val_loss`` e una tra ``val_acc``/``val_f1``.
    title : str
        Titolo del grafico.
    """
    import matplotlib.pyplot as plt

    metric_key = "val_acc" if "val_acc" in history else ("val_f1" if "val_f1" in history else None)
    metric_label = "Accuracy" if metric_key == "val_acc" else "Macro-F1"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["train_loss"], color="#1565c0", lw=2, label="Train loss")
    ax.plot(history["val_loss"], color="#ef9a3d", lw=1, alpha=0.45, label="Validation (grezza)")
    ax.plot(_smooth(history["val_loss"]), color="#ef9a3d", lw=2.3, label="Validation (media mobile)")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if metric_key is not None:
        ax2 = ax.twinx()
        ax2.plot(history[metric_key], color="#2e7d32", lw=1.6, ls="--", label=f"Val {metric_label}")
        ax2.set_ylabel(metric_label, color="#2e7d32")
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis="y", colors="#2e7d32")
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2

    ax.legend(handles, labels, fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_comparison(results: Mapping[tuple[str, str], Mapping], metric: str = "test_acc",
                    metric_label: str | None = None):
    """Grafico a barre: metrica per rete, raggruppato per rappresentazione.

    Parameters
    ----------
    results : dict
        Chiavi ``(nome_rappresentazione, nome_rete)``, valori dict con
        almeno la chiave indicata da ``metric`` (es. ``test_acc`` o
        ``test_f1``) -- lo stesso formato usato nel notebook per
        ``results_glove``, ``results_esgbert``, ecc., raccolti insieme.
    metric : str
        Chiave della metrica da confrontare in ciascun dict di risultati.
    metric_label : str, opzionale
        Etichetta leggibile per l'asse Y e il titolo (default: ``metric``).
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    metric_label = metric_label or metric
    rows = [
        {"Rappresentazione": rep, "Rete": net, metric_label: round(v[metric], 3)}
        for (rep, net), v in results.items()
    ]
    if not rows:
        raise ValueError("'results' e' vuoto: nessuna combinazione da confrontare.")
    df = pd.DataFrame(rows).sort_values(metric_label, ascending=False).reset_index(drop=True)

    piv = df.pivot_table(index="Rete", columns="Rappresentazione", values=metric_label)
    fig, ax = plt.subplots(figsize=(9, 5))
    piv.plot(kind="bar", ax=ax, rot=0)
    ax.set_ylabel(metric_label)
    ax.set_title(f"Confronto rappresentazione × rete — {metric_label}")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Rappresentazione", fontsize=8)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=7)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names: Sequence[str] = ("Environmental", "Social", "Governance"),
                          normalize: bool = True, title: str = "Confusion matrix"):
    """Confusion matrix con annotazioni numeriche.

    Parameters
    ----------
    normalize : bool
        Se True (default), mostra le proporzioni per riga (quota di
        ciascuna classe vera predetta come ciascuna classe); se False,
        mostra i conteggi assoluti.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalize else cm

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_display, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predetto")
    ax.set_ylabel("Vero")
    ax.set_title(title)

    thresh = cm_display.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = f"{cm_display[i, j]:.2f}" if normalize else f"{int(cm_display[i, j])}"
            ax.text(j, i, value, ha="center", va="center",
                    color="white" if cm_display[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_roc_curves(y_true, results: Mapping[str, Mapping], n_classes: int = 3,
                    title: str = "ROC micro-average"):
    """Curve ROC micro-average per una o piu' combinazioni.

    Parameters
    ----------
    y_true : array-like
        Etichette vere (interi 0..n_classes-1) del test set.
    results : dict
        Chiavi = nome della combinazione (es. ``"GloVe-BiLSTM"``), valori
        dict con almeno la chiave ``proba`` (array (N, n_classes) di
        probabilita' predette).
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    yb = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, v in sorted(results.items(), key=lambda kv: -np.asarray(kv[1]["proba"]).max()):
        proba = np.asarray(v["proba"])
        fpr, tpr, _ = roc_curve(yb.ravel(), proba.ravel())
        ax.plot(fpr, tpr, lw=1.6, label=f"{name} (AUC={auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_embeddings_2d(vectors, labels, class_names: Sequence[str] = ("Environmental", "Social", "Governance"),
                       method: str = "pca", title: str | None = None, seed: int = 42):
    """Proietta vettori pooled (uno per frase, ad alta dimensione) in 2D,
    colorati per pilastro -- un modo visivo per capire se una
    rappresentazione separa davvero le classi, o se il confine e'
    confuso (utile per interpretare PERCHE' una rappresentazione va
    meglio/peggio di un'altra, non solo di quanto).

    ``method``: ``"pca"`` (lineare, veloce, preserva la varianza globale
    -- buon primo sguardo) o ``"tsne"`` (non lineare, piu' lento, spesso
    separa meglio i cluster locali ma le distanze tra cluster diversi non
    sono interpretabili in valore assoluto -- solo il raggruppamento conta).

    ``vectors``: array (N, D) -- tipicamente ``pooled`` restituito da un
    qualunque encoder di :mod:`esgpillar.embeddings`, o
    ``clf._encoder(sentences)[1]`` con un :class:`esgpillar.ESGPillarClassifier`
    gia' addestrato.
    ``labels``: array (N,) di interi di classe (0=Environmental, 1=Social, 2=Governance).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    vectors = np.asarray(vectors)
    labels = np.asarray(labels)

    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=seed)
        method_label = "PCA"
    elif method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30, max(5, len(vectors) // 4))
        reducer = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca")
        method_label = "t-SNE"
    else:
        raise ValueError(f"method={method!r} non riconosciuto. Usa 'pca' o 'tsne'.")

    coords = reducer.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2e7d32", "#ef9a3d", "#1565c0"]
    for i, name in enumerate(class_names):
        mask = labels == i
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1], c=colors[i % len(colors)],
                      label=name, alpha=0.7, s=30)
    ax.set_xlabel(f"{method_label} 1")
    ax.set_ylabel(f"{method_label} 2")
    ax.set_title(title or f"Proiezione {method_label} degli embedding, per pilastro")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_wordcloud_by_pillar(df, text_col: str = "sentence", label_col: str = "esg",
                             tokenizer=None, class_names: Sequence[str] = ("E", "S", "G"),
                             min_count: int = 5, min_lift: float = 1.6, max_words: int = 200,
                             shape: str = "letter"):
    """Word cloud per pilastro, con le parole DISTINTIVE (non semplicemente
    frequenti) di ciascuna classe.

    Le parole trasversali (es. "risk", "management") compaiono spesso in
    tutti i pilastri e non li caratterizzano: ogni parola qui è pesata per
    **lift** = frequenza nella classe / frequenza nelle altre classi, cosi'
    emergono i termini davvero specifici di ciascun pilastro.

    Parameters
    ----------
    df : pd.DataFrame
        Deve avere le colonne ``text_col`` (testo) e ``label_col`` (uno dei
        valori in ``class_names``, es. "E"/"S"/"G").
    tokenizer : callable, opzionale
        Funzione testo -> lista di token (es. un'istanza di
        :class:`esgpillar.tokenization.Tokenizer`, giа lemmatizza ed esclude
        nomi azienda/stopword). Se ``None``, usa una tokenizzazione minima
        (solo lowercase + split su parole) -- MENO pulita: se il tuo
        DataFrame include una colonna azienda, costruisci comunque un
        ``Tokenizer.from_companies(...)`` e passalo qui, altrimenti i nomi
        delle aziende possono comparire come parole "distintive" (lo stesso
        problema riscontrato e corretto nello sviluppo di questa libreria).
    shape : str
        ``"letter"`` (default): word cloud a forma della lettera del
        pilastro (E/S/G). ``"rectangle"``: word cloud rettangolare
        standard, piu' leggera (non richiede PIL/font).

    Richiede l'extra ``viz`` (in particolare ``wordcloud`` e, per
    ``shape="letter"``, ``pillow``).
    """
    import re as _re
    from collections import Counter

    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    if tokenizer is None:
        def tokenizer(s):
            return _re.findall(r"[a-z]{3,}", str(s).lower())

    cmap_by_class = {"E": "Greens", "S": "Oranges", "G": "Blues"}
    colors = [cmap_by_class.get(c, "Greys") for c in class_names]

    counts_by_class = {c: Counter() for c in class_names}
    for c in class_names:
        for s in df.loc[df[label_col] == c, text_col]:
            counts_by_class[c].update(tokenizer(s))

    def _distinctive(c, top=max_words):
        others = Counter()
        for o in class_names:
            if o != c:
                others.update(counts_by_class[o])
        tot_c, tot_o = sum(counts_by_class[c].values()), sum(others.values())
        out = {}
        for w, cnt in counts_by_class[c].items():
            if cnt < min_count:
                continue
            lift = (cnt / tot_c) / ((others.get(w, 0) / tot_o) + 1e-9) if tot_o else float("inf")
            if lift >= min_lift:
                out[w] = cnt * min(lift, 5)
        return dict(sorted(out.items(), key=lambda x: -x[1])[:top])

    distinctive_by_class = {c: _distinctive(c) for c in class_names}

    def _letter_mask(letter, size=(900, 1100), font_size=1000):
        import numpy as np
        import matplotlib.font_manager as fm
        from PIL import Image, ImageDraw, ImageFont

        path = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))
        img = Image.new("L", size, 255)
        d = ImageDraw.Draw(img)
        f = ImageFont.truetype(path, font_size)
        bb = d.textbbox((0, 0), letter, font=f)
        d.text(((size[0] - (bb[2] - bb[0])) / 2 - bb[0], (size[1] - (bb[3] - bb[1])) / 2 - bb[1]),
               letter, font=f, fill=0)
        return np.array(img)

    fig, axes = plt.subplots(1, len(class_names), figsize=(6 * len(class_names), 6))
    if len(class_names) == 1:
        axes = [axes]
    for ax, c, color in zip(axes, class_names, colors):
        freqs = distinctive_by_class[c]
        if not freqs:
            ax.axis("off")
            ax.set_title(f"{c} (nessuna parola distintiva)")
            continue
        kwargs = dict(background_color="white", colormap=color, max_words=max_words,
                     relative_scaling=0.4, min_font_size=6)
        if shape == "letter":
            kwargs.update(mask=_letter_mask(c), contour_width=2, contour_color="lightgrey")
        wc = WordCloud(**kwargs).generate_from_frequencies(freqs)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(c, fontsize=15)

    fig.suptitle("Parole distintive per pilastro", fontsize=16)
    fig.tight_layout()
    return fig


def plot_sentences_per_company(df, company_col: str = "company", title: str = "Frasi per azienda"):
    """Grafico a barre: numero di frasi per azienda, con il conteggio
    scritto sopra ciascuna colonna. Utile in fase di EDA per verificare
    che nessuna azienda domini sproporzionatamente il dataset (rilevante
    per lo split per azienda: un'azienda con pochissime frasi rischia di
    finire tutta in un solo insieme)."""
    import matplotlib.pyplot as plt

    cnt = df.groupby(company_col).size().sort_values()

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(cnt)), 5))
    bars = ax.bar(cnt.index.astype(str), cnt.values, color="#2e7d32", alpha=0.85)
    ax.bar_label(bars, padding=2)
    ax.set_title(title)
    ax.set_ylabel("n. frasi")
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_class_distribution(labels, class_names: Sequence[str] = ("Environmental", "Social", "Governance"),
                            title: str = "Distribuzione delle classi"):
    """Grafico a barre della distribuzione delle classi (utile in fase di EDA)."""
    import matplotlib.pyplot as plt
    import pandas as pd

    counts = pd.Series(labels).value_counts().reindex(range(len(class_names)), fill_value=0)
    colors = ["#2e7d32", "#ef9a3d", "#1565c0"][: len(class_names)]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(class_names, counts.values, color=colors)
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom")
    ax.set_ylabel("n. frasi")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- #
# EDA aggiuntiva
# --------------------------------------------------------------------- #
def text_statistics(texts: Sequence[str]) -> dict:
    """Riepilogo statistico del corpus: numero di testi, token totali,
    dimensione del vocabolario (parole distinte), TYPE-TOKEN RATIO
    (vocabolario / token totali -- una misura di DIVERSITA' LESSICALE: un
    valore vicino a 1 indica poca ripetizione, vicino a 0 indica un
    vocabolario molto ripetitivo rispetto alla lunghezza del testo -- e
    lunghezza media delle frasi in parole.

    Nota: il type-token ratio dipende dalla lunghezza del testo (corpora
    piu' lunghi tendono ad avere TTR piu' basso anche a parita' di
    "ricchezza" lessicale reale, perche' le parole comuni si ripetono
    inevitabilmente di piu' su testi lunghi) -- confrontabile solo tra
    corpora di dimensioni simili."""
    all_words = []
    lengths = []
    for t in texts:
        words = str(t).lower().split()
        all_words.extend(words)
        lengths.append(len(words))

    vocab = set(all_words)
    n_tokens = len(all_words)
    return {
        "n_texts": len(texts),
        "n_tokens": n_tokens,
        "vocab_size": len(vocab),
        "type_token_ratio": len(vocab) / n_tokens if n_tokens else 0.0,
        "avg_sentence_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "min_sentence_length": min(lengths) if lengths else 0,
        "max_sentence_length": max(lengths) if lengths else 0,
    }


def plot_top_ngrams(texts: Sequence[str], n: int = 2, method: str = "frequency", top_n: int = 15,
                    tokenizer=None, title: str | None = None):
    """Grafico a barre orizzontali degli n-grammi piu' rilevanti nel corpus
    (vedi :func:`esgpillar.text_mining.top_ngrams` per il dettaglio dei
    due metodi disponibili, ``"frequency"`` e ``"pmi"``). Richiede l'extra
    ``mining``."""
    import matplotlib.pyplot as plt

    from .text_mining import top_ngrams

    result = top_ngrams(texts, n=n, method=method, top_n=top_n, tokenizer=tokenizer)
    labels = [" ".join(ng) for ng, _ in result][::-1]  # invertito: il migliore in alto nel grafico orizzontale
    scores = [s for _, s in result][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(labels))))
    ax.barh(labels, scores, color="#1565c0", alpha=0.85)
    ax.set_xlabel("Frequenza" if method == "frequency" else "PMI")
    ax.set_title(title or f"Top {'bigrammi' if n == 2 else 'trigrammi'} ({method})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_feature_correlation_by_pillar(df, feature_cols: Sequence[str], label_col: str = "esg",
                                       class_names: Sequence[str] = ("E", "S", "G")):
    """Heatmap di correlazione (Pearson) tra le colonne numeriche indicate
    (es. punteggi di leggibilita', sentiment, numero di claim quantitativi
    per frase), UNA PER PILASTRO -- utile per vedere se, ad esempio, le
    frasi Environmental tendono ad avere leggibilita' e sentiment
    correlati in modo diverso da quelle Governance, un pattern che una
    correlazione sull'intero corpus mescolato nasconderebbe."""
    import matplotlib.pyplot as plt
    import numpy as np

    pillars = [p for p in class_names if (df[label_col] == p).any()]
    fig, axes = plt.subplots(1, len(pillars), figsize=(5 * len(pillars), 4.5))
    if len(pillars) == 1:
        axes = [axes]

    for ax, pillar in zip(axes, pillars):
        sub = df[df[label_col] == pillar][list(feature_cols)]
        corr = sub.corr().values
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(feature_cols)))
        ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(feature_cols)))
        ax.set_yticklabels(feature_cols, fontsize=8)
        ax.set_title(f"Pilastro {pillar}")
        for i in range(len(feature_cols)):
            for j in range(len(feature_cols)):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                       color="white" if abs(corr[i, j]) > 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=axes, label="correlazione di Pearson", fraction=0.03)
    return fig


def plot_zipf(texts: Sequence[str], title: str = "Distribuzione di frequenza delle parole (legge di Zipf)"):
    """Grafico log-log rango/frequenza delle parole del corpus: secondo la
    legge di Zipf, in un testo naturale la frequenza di una parola e'
    approssimativamente inversamente proporzionale al suo rango (la
    parola piu' frequente compare circa 2 volte la seconda, 3 volte la
    terza, ecc.) -- su scala log-log questo appare come una retta quasi
    diritta. Uno scostamento marcato dalla retta (es. un "gomito" netto)
    puo' segnalare un corpus molto ripetitivo/artificiale (poche frasi
    duplicate quasi identiche) o, all'opposto, un vocabolario
    insolitamente piatto."""
    from collections import Counter

    import matplotlib.pyplot as plt
    import numpy as np

    all_words = []
    for t in texts:
        all_words.extend(str(t).lower().split())
    freq = Counter(all_words)
    counts = sorted(freq.values(), reverse=True)
    ranks = np.arange(1, len(counts) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ranks, counts, marker=".", linestyle="none", color="#1565c0", alpha=0.6)
    ax.set_xlabel("Rango (log)")
    ax.set_ylabel("Frequenza (log)")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig
