import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from esgpillar.plotting import (
    plot_feature_correlation_by_pillar,
    plot_top_ngrams,
    plot_zipf,
    text_statistics,
)

_CORPUS = [
    "We reduced greenhouse gas emissions significantly this year.",
    "Greenhouse gas emissions decreased across all our facilities.",
    "The board approved a new governance compliance policy today.",
    "Employee training programs expanded workplace safety significantly.",
]


# --------------------------------------------------------------------- #
# Statistiche testuali
# --------------------------------------------------------------------- #
def test_text_statistics_basic_fields():
    stats = text_statistics(_CORPUS)
    assert stats["n_texts"] == len(_CORPUS)
    assert stats["vocab_size"] > 0
    assert stats["n_tokens"] > stats["vocab_size"]  # ci sono ripetizioni tra le frasi
    assert 0 < stats["type_token_ratio"] <= 1
    assert stats["min_sentence_length"] <= stats["avg_sentence_length"] <= stats["max_sentence_length"]


def test_text_statistics_empty_corpus():
    stats = text_statistics([])
    assert stats["n_texts"] == 0
    assert stats["vocab_size"] == 0
    assert stats["type_token_ratio"] == 0.0


def test_text_statistics_type_token_ratio_lower_with_repetition():
    repetitive = ["the cat the cat the cat"] * 5
    diverse = ["one two three four five six seven eight nine ten"]
    stats_rep = text_statistics(repetitive)
    stats_div = text_statistics(diverse)
    assert stats_rep["type_token_ratio"] < stats_div["type_token_ratio"]


# --------------------------------------------------------------------- #
# Grafico n-gram
# --------------------------------------------------------------------- #
def test_plot_top_ngrams_frequency():
    pytest.importorskip("networkx")
    fig = plot_top_ngrams(_CORPUS, n=2, method="frequency", top_n=5)
    assert fig is not None
    assert "bigrammi" in fig.axes[0].get_title()


def test_plot_top_ngrams_pmi():
    pytest.importorskip("networkx")
    fig = plot_top_ngrams(_CORPUS, n=2, method="pmi", top_n=5)
    assert fig is not None


# --------------------------------------------------------------------- #
# Grafico Zipf
# --------------------------------------------------------------------- #
def test_plot_zipf():
    fig = plot_zipf(_CORPUS)
    assert fig is not None
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"


# --------------------------------------------------------------------- #
# Heatmap di correlazione per pilastro
# --------------------------------------------------------------------- #
def test_plot_feature_correlation_by_pillar():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "esg": ["E", "E", "S", "S", "G", "G"] * 3,
        "readability": rng.normal(size=18),
        "sentiment": rng.normal(size=18),
        "n_claims": rng.integers(0, 3, 18),
    })
    fig = plot_feature_correlation_by_pillar(df, ["readability", "sentiment", "n_claims"])
    assert fig is not None
    # 3 pilastri presenti -> 3 sotto-grafici (oltre alla colorbar, che non e' un axes standard qui)
    assert len(fig.axes) >= 3


def test_plot_feature_correlation_by_pillar_handles_missing_pillar():
    """Se un pilastro non compare affatto nel DataFrame, non deve crashare."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "esg": ["E", "E", "S", "S"],  # nessuna riga "G"
        "readability": rng.normal(size=4),
        "sentiment": rng.normal(size=4),
    })
    fig = plot_feature_correlation_by_pillar(df, ["readability", "sentiment"])
    assert fig is not None
