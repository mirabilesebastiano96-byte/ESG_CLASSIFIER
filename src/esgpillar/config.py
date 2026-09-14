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
    # "lemma" (default, POS-aware), "porter"/"snowball" (stemming), "none"
    tokenizer_normalization: str = "lemma"

    # --- lunghezza massima sequenza (token) per Conv1D/BiLSTM ---
    max_len: int = 64

    # --- embedding pre-addestrati (frozen) ---
    fasttext_pretrained_name: str = "fasttext-wiki-news-subwords-300"
    fasttext_pretrained_dim: int = 300
    glove_name: str = "glove-wiki-gigaword-300"
    glove_dim: int = 300

    # --- Word2Vec (addestrato sul corpus, non pre-addestrato) ---
    word2vec_dim: int = 300
    word2vec_window: int = 5
    word2vec_min_count: int = 1
    word2vec_epochs: int = 30

    # --- fastText addestrato sul corpus (con vere subword, a differenza
    # della variante pre-addestrata via gensim.downloader) ---
    fasttext_corpus_dim: int = 300
    fasttext_corpus_window: int = 5
    fasttext_corpus_min_count: int = 1
    fasttext_corpus_epochs: int = 30

    # --- Doc2Vec (un vettore per frase intera, non per parola) ---
    doc2vec_dim: int = 300
    doc2vec_window: int = 5
    doc2vec_min_count: int = 1
    doc2vec_epochs: int = 30

    # --- Sentence-Transformers / SBERT (frozen, un vettore per frase) ---
    sentence_transformer_model: str = "all-MiniLM-L6-v2"

    # --- LSA (statistica, per parola: TF-IDF + SVD sul corpus) ---
    lsa_dim: int = 300
    lsa_max_features: int = 20000
    lsa_min_df: int = 2

    # --- rappresentazioni contestuali (tutte frozen: nessun fine-tuning) ---
    esgbert_model: str = "yiyanghkust/finbert-esg"
    esgbert_batch: int = 32
    finbert_model: str = "ProsusAI/finbert"
    climatebert_model: str = "climatebert/distilroberta-base-climate-f"
    bert_base_model: str = "bert-base-uncased"
    roberta_model: str = "roberta-base"
    umberto_model: str = "Musixmatch/umberto-commoncrawl-cased-v1"  # italiano, non inglese

    # --- ELMo: NON si scarica automaticamente, va scaricato a mano
    # (http://vectors.nlpl.eu/repository/) e indicato qui come percorso locale ---
    elmo_model_path: str = "./elmo_model"
    elmo_dim: int = 1024

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
