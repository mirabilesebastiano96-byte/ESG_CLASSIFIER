from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from esgpillar.text_mining import (
    fit_topic_model,
    find_similar_sentences,
    keywords_rake,
    keywords_textrank,
    keywords_tfidf,
    similarity_embeddings,
    similarity_tfidf,
    summarize_textrank,
)

_CORPUS = [
    "We reduced carbon emissions significantly through renewable energy investments.",
    "Water consumption decreased across our manufacturing facilities this year.",
    "Employee training programs on workplace safety were expanded to all offices.",
    "The board of directors approved a new anti corruption compliance policy.",
    "Audit committee strengthened oversight of corporate governance practices.",
    "Board diversity increased following several new independent director appointments.",
]


class _FakeEncoder:
    """Encoder finto compatibile con l'interfaccia (seq, pooled, lengths).
    Deterministico per testo (stesso testo -> stesso vettore), come farebbe
    un vero encoder -- necessario per testare correttamente la similarita'."""

    def __init__(self, dim=10, seed=0):
        self.dim = dim
        self.seed = seed

    def __call__(self, sentences):
        pooled = np.stack([
            np.random.default_rng(abs(hash(s)) % (2**32) + self.seed).normal(size=self.dim).astype("float32")
            for s in sentences
        ])
        return None, pooled, None


# --------------------------------------------------------------------- #
# Estrazione keyword
# --------------------------------------------------------------------- #
def test_keywords_tfidf_returns_scored_words():
    result = keywords_tfidf(_CORPUS, top_n=5)
    assert len(result) == 5
    assert all(isinstance(w, str) and isinstance(s, float) for w, s in result)
    # ordinato per punteggio decrescente
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)


def test_keywords_rake_returns_phrases():
    pytest.importorskip("rake_nltk")
    result = keywords_rake(_CORPUS[0], top_n=3)
    assert len(result) <= 3
    assert all(isinstance(phrase, str) for phrase, _ in result)


def test_keywords_textrank_returns_ranked_words():
    pytest.importorskip("networkx")
    result = keywords_textrank(" ".join(_CORPUS), top_n=5)
    assert len(result) == 5
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)


def test_keywords_textrank_handles_short_text():
    pytest.importorskip("networkx")
    result = keywords_textrank("hello", top_n=5)
    assert result == [("hello", 1.0)]


def test_keywords_textrank_uses_tokenizer_when_given():
    pytest.importorskip("networkx")
    from esgpillar.tokenization import Tokenizer
    tok = Tokenizer.from_companies([])
    result = keywords_textrank(_CORPUS[0], top_n=5, tokenizer=tok)
    words = [w for w, _ in result]
    assert "the" not in words  # stopword, dovrebbe essere gia' rimossa dal tokenizzatore


# --------------------------------------------------------------------- #
# Topic modeling
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["lda", "nmf"])
def test_fit_topic_model(method):
    tm = fit_topic_model(_CORPUS, n_topics=2, method=method, seed=0)
    topics = tm.top_words(n_words=4)
    assert len(topics) == 2
    assert all(len(t) == 4 for t in topics)

    dist = tm.transform(_CORPUS)
    assert dist.shape == (len(_CORPUS), 2)


def test_fit_topic_model_unknown_method_raises():
    with pytest.raises(ValueError, match="non riconosciuto"):
        fit_topic_model(_CORPUS, method="invalid")


# --------------------------------------------------------------------- #
# Riassunto estrattivo
# --------------------------------------------------------------------- #
def test_summarize_textrank_returns_fewer_sentences():
    pytest.importorskip("networkx")
    summary = summarize_textrank(_CORPUS, n_sentences=2)
    assert len(summary) == 2
    assert all(s in _CORPUS for s in summary)


def test_summarize_textrank_preserves_original_order():
    pytest.importorskip("networkx")
    summary = summarize_textrank(_CORPUS, n_sentences=3)
    indices = [_CORPUS.index(s) for s in summary]
    assert indices == sorted(indices)


def test_summarize_textrank_returns_all_if_fewer_than_requested():
    pytest.importorskip("networkx")
    short_corpus = _CORPUS[:2]
    summary = summarize_textrank(short_corpus, n_sentences=5)
    assert summary == short_corpus


# --------------------------------------------------------------------- #
# Similarita'
# --------------------------------------------------------------------- #
def test_similarity_tfidf_identical_sentences():
    sim = similarity_tfidf(_CORPUS[0], _CORPUS[0])
    assert sim == pytest.approx(1.0, abs=1e-6)


def test_similarity_tfidf_unrelated_sentences():
    sim = similarity_tfidf("We reduced carbon emissions.", "The cat sat on the mat.")
    assert sim == pytest.approx(0.0, abs=1e-6)


def test_similarity_embeddings_with_fake_encoder():
    enc = _FakeEncoder()
    sim = similarity_embeddings(_CORPUS[0], _CORPUS[1], enc)
    assert -1.0 <= sim <= 1.0


def test_similarity_embeddings_identical_text_gives_similarity_one():
    enc = _FakeEncoder()
    sim = similarity_embeddings(_CORPUS[0], _CORPUS[0], enc)
    assert sim == pytest.approx(1.0, abs=1e-4)


def test_find_similar_sentences_returns_top_k():
    enc = _FakeEncoder()
    result = find_similar_sentences(_CORPUS[0], _CORPUS[1:], enc, top_k=3)
    assert len(result) == 3
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)
    assert all(text in _CORPUS[1:] for text, _ in result)


# --------------------------------------------------------------------- #
# Leggibilita'
# --------------------------------------------------------------------- #
def test_readability_scores_returns_three_indices():
    pytest.importorskip("textstat")
    from esgpillar.text_mining import readability_scores
    result = readability_scores("We cut waste. We saved water. We helped workers.")
    assert set(result) == {"flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog"}
    assert all(isinstance(v, (int, float)) for v in result.values())


def test_readability_simple_text_easier_than_complex_text():
    pytest.importorskip("textstat")
    from esgpillar.text_mining import readability_scores
    simple = readability_scores("We cut waste. We saved water. We helped workers.")
    complex_ = readability_scores(
        "The comprehensive implementation of multifaceted sustainability infrastructure "
        "necessitated substantial capital reallocation across numerous operational divisions."
    )
    assert simple["flesch_reading_ease"] > complex_["flesch_reading_ease"]
    assert simple["flesch_kincaid_grade"] < complex_["flesch_kincaid_grade"]


def test_add_readability_columns():
    pytest.importorskip("textstat")
    from esgpillar.text_mining import add_readability_columns
    df = pd.DataFrame({"sentence": ["We cut waste.", "Comprehensive multifaceted infrastructure implementation."]})
    result = add_readability_columns(df)
    assert {"flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog"}.issubset(result.columns)


# --------------------------------------------------------------------- #
# Claim quantitativi
# --------------------------------------------------------------------- #
def test_extract_quantitative_claims_finds_all_values_without_swallowing():
    from esgpillar.text_mining import extract_quantitative_claims
    text = (
        "We reduced our carbon emissions by 25% since 2019, investing $10 million in "
        "renewable projects and cutting water use by 15 percent."
    )
    claims = extract_quantitative_claims(text)
    values = [c["value"] for c in claims]
    assert len(claims) == 3, f"attesi 3 claim distinti, trovati {len(claims)}: {values}"
    assert "25%" in values
    assert "15 percent" in values
    assert any("10 million" in v for v in values)


def test_extract_quantitative_claims_includes_context():
    from esgpillar.text_mining import extract_quantitative_claims
    claims = extract_quantitative_claims("Revenue grew by 12% this year thanks to strong demand.")
    assert len(claims) == 1
    assert "revenue" in claims[0]["context_before"].lower() or "grew" in claims[0]["context_before"].lower()
    assert claims[0]["start"] < claims[0]["end"]


def test_extract_quantitative_claims_empty_when_no_numbers():
    from esgpillar.text_mining import extract_quantitative_claims
    claims = extract_quantitative_claims("We are committed to sustainability and transparency.")
    assert claims == []


def test_add_quantitative_claims_column():
    from esgpillar.text_mining import add_quantitative_claims_column
    df = pd.DataFrame({"sentence": ["We reduced emissions by 25%.", "No numbers here."]})
    result = add_quantitative_claims_column(df)
    assert "quantitative_claims" in result.columns
    assert len(result["quantitative_claims"].iloc[0]) == 1
    assert result["quantitative_claims"].iloc[1] == []


# --------------------------------------------------------------------- #
# N-grammi / collocazioni
# --------------------------------------------------------------------- #
def test_top_ngrams_frequency():
    from esgpillar.text_mining import top_ngrams
    corpus = [
        "We reduced greenhouse gas emissions.",
        "Greenhouse gas emissions decreased significantly this year.",
    ]
    result = top_ngrams(corpus, n=2, method="frequency", top_n=3)
    ngrams = [ng for ng, _ in result]
    assert ("gas", "emissions") in ngrams


def test_top_ngrams_pmi():
    from esgpillar.text_mining import top_ngrams
    corpus = [
        "We reduced greenhouse gas emissions.",
        "Greenhouse gas emissions decreased significantly this year.",
        "The board met to discuss quarterly results.",
    ]
    result = top_ngrams(corpus, n=2, method="pmi", top_n=5)
    assert len(result) > 0
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)


def test_top_ngrams_trigrams():
    from esgpillar.text_mining import top_ngrams
    corpus = ["We reduced greenhouse gas emissions significantly this year."]
    result = top_ngrams(corpus, n=3, method="frequency", top_n=3)
    assert all(len(ng) == 3 for ng, _ in result)


def test_top_ngrams_invalid_n_raises():
    from esgpillar.text_mining import top_ngrams
    with pytest.raises(ValueError, match="non supportato"):
        top_ngrams(["some text"], n=4)


def test_top_ngrams_invalid_method_raises():
    from esgpillar.text_mining import top_ngrams
    with pytest.raises(ValueError, match="non riconosciuto"):
        top_ngrams(["some text"], method="invalid")


# --------------------------------------------------------------------- #
# Classificazione zero-shot (pipeline mockata)
# --------------------------------------------------------------------- #
def test_zero_shot_classify_with_mocked_pipeline():
    import esgpillar.text_mining as tm_module

    fake_pipeline = MagicMock(
        return_value={"labels": ["Environmental", "Social", "Governance"], "scores": [0.85, 0.10, 0.05]}
    )
    tm_module.zero_shot_classify._pipeline_cache.clear()
    with patch("esgpillar.text_mining._load_zero_shot_pipeline", return_value=fake_pipeline):
        result = tm_module.zero_shot_classify(
            "We reduced carbon emissions.", ["Environmental", "Social", "Governance"]
        )
    assert result["labels"][0] == "Environmental"
    assert result["scores"][0] == 0.85


def test_zero_shot_classify_caches_pipeline_per_model():
    import esgpillar.text_mining as tm_module

    fake_pipeline = MagicMock(return_value={"labels": ["A"], "scores": [0.9]})
    tm_module.zero_shot_classify._pipeline_cache.clear()
    with patch("esgpillar.text_mining._load_zero_shot_pipeline", return_value=fake_pipeline) as mock_load:
        tm_module.zero_shot_classify("text one", ["A"], model="fake-model")
        tm_module.zero_shot_classify("text two", ["A"], model="fake-model")
    mock_load.assert_called_once()  # la seconda chiamata riusa la pipeline in cache


def test_add_zero_shot_column():
    import esgpillar.text_mining as tm_module

    fake_pipeline = MagicMock(return_value={"labels": ["Environmental", "Social"], "scores": [0.7, 0.3]})
    tm_module.zero_shot_classify._pipeline_cache.clear()
    df = pd.DataFrame({"sentence": ["We reduced carbon emissions."]})
    with patch("esgpillar.text_mining._load_zero_shot_pipeline", return_value=fake_pipeline):
        result = tm_module.add_zero_shot_column(df, candidate_labels=["Environmental", "Social"])
    assert result["zero_shot_label"].iloc[0] == "Environmental"
    assert result["zero_shot_label_score"].iloc[0] == 0.7
