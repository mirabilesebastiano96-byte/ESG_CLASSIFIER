"""Configurazione centrale della pipeline ESG Pillar Classifier.

Un'unica dataclass raccoglie tutti i parametri usati dalle varie fasi
(estrazione, pulizia, etichettatura, tokenizzazione, embedding, training),
cosi' ogni funzione della libreria riceve la STESSA configurazione invece
di leggere variabili globali sparse — il problema che affliggeva la
versione a notebook di questo lavoro (parametri duplicati, celle che si
rompevano a seconda dell'ordine di esecuzione).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    """Parametri della pipeline, con default ragionevoli.

    Esempio
    -------
    >>> from esgpillar import Config
    >>> cfg = Config(pdf_dir="./reports")
    """

    # --- riproducibilita' ---
    seed: int = 42

    # --- split per azienda (anti-leakage: mai la stessa azienda in piu' insiemi) ---
    test_size: float = 0.20
    val_size: float = 0.15

    # --- estrazione dai PDF ---
    pdf_dir: str = "./reports"
    extract_cache: str = "dataset_from_pdfs.csv"
    min_words_sent: int = 8
    max_words_sent: int = 45
    check_verb: bool = True

    # --- pulizia del testo ---
    min_words: int = 5
    max_words: int = 100
    letter_ratio: float = 0.55
    dedup_mode: str = "normalized"  # "exact" | "normalized"
    lang_filter: bool = True
    target_lang: str = "en"

    # --- etichettatura debole (topic ESRS) ---
    # nessuna soglia di confidenza minima di default: la frase resta al
    # topic vincente qualunque sia il punteggio (Other solo per parita'/nessun match)
    esrs_conf_thresh: float | None = None

    # --- anti-leakage: le parole del dizionario ESRS che hanno GENERATO
    # l'etichetta vengono rimosse dal testo usato per le rappresentazioni
    # (altrimenti il modello impara a riconoscere la parola-chiave, non il
    # concetto -> metriche artificialmente alte). Attivo di default.
    apply_leakage_mask: bool = True

    # --- tokenizzazione condivisa ---
    generic_esg_terms: tuple[str, ...] = (
        "esg", "sustainability", "sustainable", "report", "reporting", "company",
        "companies", "business", "corporate", "group", "year", "also", "including",
    )

    # --- lunghezza massima sequenza (token) per Conv1D/BiLSTM ---
    max_len: int = 64

    # --- embedding pre-addestrati (frozen) ---
    fasttext_pretrained_name: str = "fasttext-wiki-news-subwords-300"
    fasttext_pretrained_dim: int = 300
    glove_name: str = "glove-wiki-gigaword-300"
    glove_dim: int = 300

    # --- ESGBERT / FinBERT-ESG (contestuale, frozen: nessun fine-tuning) ---
    esgbert_model: str = "yiyanghkust/finbert-esg"
    esgbert_batch: int = 32

    # --- training reti ---
    epochs: int = 80
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 4e-4
    grad_clip: float = 1.0
    patience: int = 4

    class_names: tuple[str, ...] = ("Environmental", "Social", "Governance")

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)
