from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("torch")

from esgpillar.config import Config  # noqa: E402
from esgpillar.embeddings import (  # noqa: E402
    load_bert_base,
    load_climatebert,
    load_contextual_model,
    load_finbert,
)


def _fake_auto_model_tokenizer():
    fake_tok = MagicMock(name="tokenizer")
    fake_model = MagicMock(name="model")
    return fake_tok, fake_model


@patch("transformers.AutoModel.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_load_contextual_model_uses_given_name(mock_tok_from_pretrained, mock_model_from_pretrained):
    fake_tok, fake_model = _fake_auto_model_tokenizer()
    mock_tok_from_pretrained.return_value = fake_tok
    mock_model_from_pretrained.return_value = fake_model

    tok, model = load_contextual_model("some/model-name")

    mock_tok_from_pretrained.assert_called_once_with("some/model-name")
    mock_model_from_pretrained.assert_called_once_with("some/model-name")
    assert tok is fake_tok
    assert model is fake_model


@patch("esgpillar.embeddings.load_contextual_model")
def test_load_finbert_uses_config_model_name(mock_load):
    mock_load.return_value = ("tok", "model")
    cfg = Config(finbert_model="ProsusAI/finbert")
    load_finbert(cfg)
    mock_load.assert_called_once_with("ProsusAI/finbert")


@patch("esgpillar.embeddings.load_contextual_model")
def test_load_climatebert_uses_config_model_name(mock_load):
    mock_load.return_value = ("tok", "model")
    cfg = Config(climatebert_model="climatebert/distilroberta-base-climate-f")
    load_climatebert(cfg)
    mock_load.assert_called_once_with("climatebert/distilroberta-base-climate-f")


@patch("esgpillar.embeddings.load_contextual_model")
def test_load_bert_base_uses_config_model_name(mock_load):
    mock_load.return_value = ("tok", "model")
    cfg = Config(bert_base_model="bert-base-uncased")
    load_bert_base(cfg)
    mock_load.assert_called_once_with("bert-base-uncased")


def test_all_contextual_loaders_use_default_config_when_none_given():
    with patch("esgpillar.embeddings.load_contextual_model") as mock_load:
        mock_load.return_value = ("tok", "model")
        load_finbert()
        load_climatebert()
        load_bert_base()
        assert mock_load.call_count == 3
