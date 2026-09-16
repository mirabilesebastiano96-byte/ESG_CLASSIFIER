from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

from esgpillar.config import Config
from esgpillar.text_analysis import (
    add_entities_column,
    add_pos_column,
    add_sentiment_column,
    extract_entities,
    pos_tag_sentence,
    sentiment_finbert,
    sentiment_textblob,
    sentiment_vader,
)


# --------------------------------------------------------------------- #
# POS tagging
# --------------------------------------------------------------------- #
def test_pos_tag_sentence_returns_word_tag_pairs():
    result = pos_tag_sentence("The board approved a new policy.")
    assert isinstance(result, list)
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in result)
    words = [w for w, _ in result]
    assert "board" in words


def test_add_pos_column():
    df = pd.DataFrame({"sentence": ["We reduced emissions.", "The board met today."]})
    result = add_pos_column(df)
    assert "pos_tags" in result.columns
    assert len(result) == len(df)


# --------------------------------------------------------------------- #
# NER
# --------------------------------------------------------------------- #
def test_extract_entities_nltk_backend():
    entities = extract_entities("The meeting took place in New York.", backend="nltk")
    assert isinstance(entities, list)
    # "New York" e' un'entita' ben nota, dovrebbe essere riconosciuta da nltk
    texts = [e[0] for e in entities]
    assert any("New York" in t or "York" in t for t in texts)


def test_extract_entities_unknown_backend_raises():
    with pytest.raises(ValueError, match="non riconosciuto"):
        extract_entities("Some text.", backend="invalid")


def test_extract_entities_spacy_without_model_raises_clear_error():
    """Se spaCy e' installato ma il modello non e' scaricato, l'errore deve
    essere chiaro (istruzioni), non un traceback oscuro."""
    pytest.importorskip("spacy")
    import esgpillar.text_analysis as ta
    if hasattr(ta.extract_entities, "_spacy_model"):
        del ta.extract_entities._spacy_model
    try:
        ta.extract_entities("Some text.", backend="spacy")
    except RuntimeError as e:
        assert "spacy download" in str(e)
    except OSError:
        pytest.skip("comportamento diverso a seconda della versione di spaCy installata")


def test_add_entities_column():
    df = pd.DataFrame({"sentence": ["The meeting took place in New York."]})
    result = add_entities_column(df)
    assert "entities" in result.columns


# --------------------------------------------------------------------- #
# Sentiment: VADER
# --------------------------------------------------------------------- #
def test_sentiment_vader_positive():
    result = sentiment_vader("We are proud of our excellent progress on sustainability.")
    assert result["label"] == "positive"
    assert result["compound"] > 0.05
    assert set(result) == {"neg", "neu", "pos", "compound", "label"}


def test_sentiment_vader_negative():
    result = sentiment_vader("Our safety record was terrible and workers suffered greatly.")
    assert result["label"] == "negative"
    assert result["compound"] < -0.05


# --------------------------------------------------------------------- #
# Sentiment: TextBlob
# --------------------------------------------------------------------- #
def test_sentiment_textblob_positive():
    pytest.importorskip("textblob")
    result = sentiment_textblob("We are proud of our excellent progress.")
    assert result["label"] == "positive"
    assert -1 <= result["polarity"] <= 1
    assert 0 <= result["subjectivity"] <= 1


# --------------------------------------------------------------------- #
# Sentiment: FinBERT (modello, mockato)
# --------------------------------------------------------------------- #
class _FakeEncoding(dict):
    def to(self, device):
        return self


def _make_fake_finbert(forced_label_idx=1):
    fake_tok = MagicMock()
    fake_tok.side_effect = lambda batch, truncation=True, padding=True, return_tensors="pt": _FakeEncoding(
        input_ids=torch.zeros(len(batch), 5, dtype=torch.long),
        attention_mask=torch.ones(len(batch), 5, dtype=torch.long),
    )
    fake_model = MagicMock()
    fake_model.eval.return_value = fake_model
    fake_model.config.id2label = {0: "positive", 1: "negative", 2: "neutral"}

    def fake_call(**kwargs):
        n = kwargs["input_ids"].shape[0]
        logits = torch.zeros(n, 3)
        logits[:, forced_label_idx] = 5.0
        return MagicMock(logits=logits)

    fake_model.side_effect = fake_call
    return fake_tok, fake_model


def test_sentiment_finbert_with_mocked_model():
    fake_tok, fake_model = _make_fake_finbert(forced_label_idx=1)  # "negative"
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok), \
        patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=fake_model):
        results = sentiment_finbert(["Some financial sentence.", "Another one."], Config())

    assert len(results) == 2
    assert all(r["label"] == "negative" for r in results)
    assert all(r["score"] > 0.9 for r in results)


def test_add_sentiment_column_vader():
    df = pd.DataFrame({"sentence": ["Great progress this year.", "Terrible safety record."]})
    result = add_sentiment_column(df, method="vader")
    assert "sentiment_label" in result.columns
    assert "sentiment_score" in result.columns
    assert result["sentiment_label"].iloc[0] == "positive"
    assert result["sentiment_label"].iloc[1] == "negative"


def test_add_sentiment_column_finbert():
    fake_tok, fake_model = _make_fake_finbert(forced_label_idx=0)  # "positive"
    df = pd.DataFrame({"sentence": ["Some financial sentence."]})
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok), \
        patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=fake_model):
        result = add_sentiment_column(df, method="finbert")
    assert result["sentiment_label"].iloc[0] == "positive"


def test_add_sentiment_column_unknown_method_raises():
    df = pd.DataFrame({"sentence": ["Some sentence."]})
    with pytest.raises(ValueError, match="non riconosciuto"):
        add_sentiment_column(df, method="invalid")
