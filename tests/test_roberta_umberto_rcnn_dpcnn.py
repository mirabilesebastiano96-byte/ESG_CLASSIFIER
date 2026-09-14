from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from esgpillar import Config, ESGPillarClassifier  # noqa: E402
from esgpillar.models import DPCNN, RCNN  # noqa: E402

_ROWS = [
    {"company": "Acme", "sentence": "We reduced carbon emissions through renewable energy programs.", "label": 0},
    {"company": "Acme", "sentence": "Water consumption decreased across our manufacturing facilities.", "label": 0},
    {"company": "Acme", "sentence": "Employee training programs on workplace safety were expanded.", "label": 1},
    {"company": "Beta", "sentence": "Diversity and inclusion initiatives improved representation.", "label": 1},
    {"company": "Beta", "sentence": "The board approved a new anti corruption compliance policy.", "label": 2},
    {"company": "Beta", "sentence": "Audit committee strengthened oversight of governance procedures.", "label": 2},
    {"company": "Gamma", "sentence": "Biodiversity protection measures were implemented near our sites.", "label": 0},
    {"company": "Gamma", "sentence": "Worker health and safety incidents declined after training.", "label": 1},
    {"company": "Gamma", "sentence": "Governance structure was revised to increase board independence.", "label": 2},
]


# --------------------------------------------------------------------- #
# RoBERTa / UmBERTo: routing corretto (nessun download reale)
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("embedding,model_field", [("roberta", "roberta_model"), ("umberto", "umberto_model")])
def test_roberta_umberto_route_through_contextual_encoder(embedding, model_field):
    with patch("esgpillar.embeddings.load_contextual_model") as mock_load:
        fake_model = MagicMock()
        fake_model.config.hidden_size = 12
        fake_model.to.return_value = fake_model
        fake_model.eval.return_value = fake_model
        mock_load.return_value = (MagicMock(), fake_model)

        cfg = Config()
        clf = ESGPillarClassifier(config=cfg, embedding=embedding, network="DNN")
        encoder = clf._build_encoder(companies=[], device="cpu")

        mock_load.assert_called_once_with(getattr(cfg, model_field))
        assert encoder.dim == 12
        assert clf._tokenizer is None


# --------------------------------------------------------------------- #
# RCNN / DPCNN: forward pass diretto
# --------------------------------------------------------------------- #
def test_rcnn_forward_pass():
    model = RCNN(d=10, h=16, conv_dim=24)
    x = torch.randn(4, 15, 10)
    out = model(x)
    assert out.shape == (4, 3)


@pytest.mark.parametrize("seq_len", [1, 2, 3, 5, 64])
def test_dpcnn_forward_pass_various_lengths(seq_len):
    """DPCNN dimezza la sequenza a ogni blocco: deve fermarsi da sola
    invece di crashare quando la sequenza diventa troppo corta."""
    model = DPCNN(d=10, max_blocks=4)
    x = torch.randn(3, seq_len, 10)
    out = model(x)
    assert out.shape == (3, 3)


def test_dpcnn_uses_projection_for_high_dim_input():
    model = DPCNN(d=768, proj_dim=128)
    assert model.proj is not None
    out = model(torch.randn(2, 30, 768))
    assert out.shape == (2, 3)


# --------------------------------------------------------------------- #
# Attraverso il pipeline completo (con un embedding leggero, word2vec)
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("network", ["RCNN", "DPCNN"])
def test_new_networks_through_full_pipeline(network):
    pytest.importorskip("gensim")
    df = pd.DataFrame(_ROWS)
    cfg = Config(word2vec_dim=8, val_size=0.34, epochs=2, patience=1, max_len=16)

    clf = ESGPillarClassifier(config=cfg, embedding="word2vec", network=network)
    clf.fit(df, text_col="sentence", company_col="company", label_col="label")

    preds = clf.predict(["We invested in solar panels to cut emissions."])
    assert len(preds) == 1
    assert preds[0] in cfg.class_names
