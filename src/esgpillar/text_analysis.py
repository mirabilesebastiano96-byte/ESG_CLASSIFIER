"""Analisi linguistica per-frase: POS tagging, riconoscimento di entita' (NER), sentiment.

Nessuna di queste tecniche entra nella pipeline di classificazione dei
pilastri E/S/G (:mod:`esgpillar.pipeline`) -- sono strumenti di analisi
descrittiva/esplorativa, utili per arricchire un DataFrame con colonne
aggiuntive: es. per vedere se un pilastro tende ad avere un tono piu'
negativo, o quali entita' (aziende, persone, luoghi) ricorrono di piu' in
un pilastro rispetto agli altri.

POS tagging: nltk (nessuna dipendenza extra oltre al core).

NER: due backend intercambiabili.
- ``"nltk"`` (default): nessuna dipendenza extra, ma il chunker classico
  di nltk e' impreciso su testo reale -- es. puo' confondere un nome
  azienda con una persona.
- ``"spacy"`` (extra ``nlp``): sensibilmente piu' accurato, ma richiede
  un modello scaricato a parte (``python -m spacy download en_core_web_sm``).

Sentiment: tre tecniche.
- ``"vader"`` (default): lessico di polarita' (nltk), pensato per testo
  breve e informale -- veloce, nessun modello da scaricare.
- ``"textblob"``: altro lessico di polarita', da' anche la soggettivita'
  (quanto la frase e' opinione vs fatto), non solo la polarita'.
- ``"finbert"``: modello (non lessico), FinBERT con la sua testa di
  classificazione fine-tunata per il sentiment finanziario -- piu' lento,
  richiede l'extra ``transformers``, ma pensato apposta per testo
  economico-finanziario (a differenza di VADER/TextBlob, generici).
"""
from __future__ import annotations

from collections.abc import Sequence

from .config import Config


def ensure_text_analysis_nltk_data(quiet: bool = True) -> None:
    """Scarica i corpora nltk aggiuntivi per NER e sentiment (oltre a
    quelli gia' richiesti da :func:`esgpillar.tokenization.ensure_nltk_data`)."""
    import nltk
    for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
               "maxent_ne_chunker", "maxent_ne_chunker_tab", "words", "vader_lexicon"]:
        try:
            nltk.download(pkg, quiet=quiet)
        except Exception:
            pass


# --------------------------------------------------------------------- #
# POS tagging
# --------------------------------------------------------------------- #
def pos_tag_sentence(text: str) -> list[tuple[str, str]]:
    """Ritorna la lista ``(parola, tag_POS)`` per la frase (tagset di Penn Treebank,
    es. ``NN``=nome singolare, ``VBD``=verbo al passato, ``JJ``=aggettivo)."""
    from nltk import pos_tag, word_tokenize
    ensure_text_analysis_nltk_data()
    return pos_tag(word_tokenize(str(text)))


def add_pos_column(df, text_col: str = "sentence", out_col: str = "pos_tags"):
    """Aggiunge una colonna con i tag POS di ciascuna frase del DataFrame."""
    df = df.copy()
    df[out_col] = df[text_col].apply(pos_tag_sentence)
    return df


# --------------------------------------------------------------------- #
# NER (Named Entity Recognition)
# --------------------------------------------------------------------- #
def extract_entities(text: str, backend: str = "nltk") -> list[tuple[str, str]]:
    """Estrae le entita' nominate dalla frase come lista ``(testo, tipo)``,
    es. ``[("Amazon", "ORG"), ("New York", "GPE")]``.

    ``backend``: ``"nltk"`` (default, nessuna dipendenza extra, meno
    accurato) o ``"spacy"`` (extra ``nlp``, piu' accurato, richiede un
    modello scaricato a parte).
    """
    if backend == "nltk":
        from nltk import ne_chunk, pos_tag, word_tokenize
        ensure_text_analysis_nltk_data()
        tree = ne_chunk(pos_tag(word_tokenize(str(text))))
        entities = []
        for chunk in tree:
            if hasattr(chunk, "label"):
                entities.append((" ".join(c[0] for c in chunk), chunk.label()))
        return entities

    if backend == "spacy":
        import spacy
        try:
            nlp = extract_entities._spacy_model
        except AttributeError:
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError as e:
                raise RuntimeError(
                    "Modello spaCy non trovato. Scaricalo con: "
                    "python -m spacy download en_core_web_sm"
                ) from e
            extract_entities._spacy_model = nlp
        doc = nlp(str(text))
        return [(ent.text, ent.label_) for ent in doc.ents]

    raise ValueError(f"backend={backend!r} non riconosciuto. Usa 'nltk' o 'spacy'.")


def add_entities_column(df, text_col: str = "sentence", out_col: str = "entities", backend: str = "nltk"):
    """Aggiunge una colonna con le entita' nominate di ciascuna frase del DataFrame."""
    df = df.copy()
    df[out_col] = df[text_col].apply(lambda t: extract_entities(t, backend=backend))
    return df


# --------------------------------------------------------------------- #
# Sentiment
# --------------------------------------------------------------------- #
def sentiment_vader(text: str) -> dict:
    """Sentiment con VADER (lessico, nltk): ritorna i punteggi
    ``neg``/``neu``/``pos``/``compound`` (il ``compound`` va da -1 a 1) e
    un'etichetta discreta (``positive``/``neutral``/``negative``, soglie
    standard di VADER: |compound| < 0.05 -> neutral).

    LIMITE DA CONOSCERE: come ogni lessico generico, VADER non capisce il
    contenuto semantico specifico del dominio ESG -- una frase come
    "Employee safety incidents increased dramatically this year" (un
    fatto negativo) puo' essere classificata erroneamente come positiva,
    perche' il lessico non associa "incidents increased" a un evento
    negativo nel contesto della sicurezza sul lavoro. Per testo
    economico-finanziario, :func:`sentiment_finbert` (un modello, non un
    lessico) e' generalmente piu' affidabile."""
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    ensure_text_analysis_nltk_data()
    try:
        sia = sentiment_vader._analyzer
    except AttributeError:
        sia = SentimentIntensityAnalyzer()
        sentiment_vader._analyzer = sia
    scores = sia.polarity_scores(str(text))
    if scores["compound"] >= 0.05:
        label = "positive"
    elif scores["compound"] <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {**scores, "label": label}


def sentiment_textblob(text: str) -> dict:
    """Sentiment con TextBlob (lessico): ritorna ``polarity`` (-1 a 1) e
    ``subjectivity`` (0 a 1, quanto la frase e' opinione vs fatto oggettivo
    -- una dimensione che VADER non da'), piu' un'etichetta discreta."""
    from textblob import TextBlob
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    if polarity > 0.05:
        label = "positive"
    elif polarity < -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {"polarity": polarity, "subjectivity": blob.sentiment.subjectivity, "label": label}


def sentiment_finbert(sentences: Sequence[str], config: Config | None = None,
                      batch_size: int = 32) -> list[dict]:
    """Sentiment con FinBERT (MODELLO fine-tunato, non un lessico) --
    pensato apposta per testo economico-finanziario, a differenza di
    VADER/TextBlob (generici). Usa la testa di classificazione originale
    del modello (positive/negative/neutral), non solo l'encoder come fa
    :func:`esgpillar.embeddings.load_finbert` (che carica FinBERT come
    sorgente di EMBEDDING, senza testa di classificazione).

    Ritorna una lista di dict ``{"label": ..., "score": ...}``, uno per frase.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    config = config or Config()
    tok = AutoTokenizer.from_pretrained(config.finbert_model)
    model = AutoModelForSequenceClassification.from_pretrained(config.finbert_model).eval()
    id2label = model.config.id2label

    results = []
    sentences = list(sentences)
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start + batch_size]
        enc = tok(batch, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        proba = torch.softmax(logits, dim=1)
        for i in range(len(batch)):
            idx = int(proba[i].argmax())
            results.append({"label": id2label[idx].lower(), "score": float(proba[i, idx])})
    return results


def add_sentiment_column(df, text_col: str = "sentence", out_col: str = "sentiment", method: str = "vader",
                         config: Config | None = None):
    """Aggiunge colonne di sentiment al DataFrame: ``{out_col}_label`` e
    ``{out_col}_score`` (compound per vader, polarity per textblob, score
    di confidenza per finbert).

    ``method``: ``"vader"`` (default), ``"textblob"`` o ``"finbert"``.
    """
    df = df.copy()
    if method == "vader":
        scores = df[text_col].apply(sentiment_vader)
        df[f"{out_col}_label"] = scores.apply(lambda s: s["label"])
        df[f"{out_col}_score"] = scores.apply(lambda s: s["compound"])
    elif method == "textblob":
        scores = df[text_col].apply(sentiment_textblob)
        df[f"{out_col}_label"] = scores.apply(lambda s: s["label"])
        df[f"{out_col}_score"] = scores.apply(lambda s: s["polarity"])
    elif method == "finbert":
        results = sentiment_finbert(df[text_col].tolist(), config=config)
        df[f"{out_col}_label"] = [r["label"] for r in results]
        df[f"{out_col}_score"] = [r["score"] for r in results]
    else:
        raise ValueError(f"method={method!r} non riconosciuto. Usa 'vader', 'textblob' o 'finbert'.")
    return df
