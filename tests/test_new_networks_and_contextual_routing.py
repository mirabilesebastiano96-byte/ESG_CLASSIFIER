from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

pytest.importorskip("torch")

from esgpillar import Config, ESGPillarClassifier  # noqa: E402

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


@pytest.mark.parametrize("network", ["Transformer", "AttentionPool"])
def test_new_network_architectures_through_full_pipeline(network):
    df = pd.DataFrame(_ROWS)
    cfg = Config(glove_dim=8, val_size=0.34, epochs=2, patience=1)

    # embedding statico "glove" e' pesante da scaricare per un test: qui verifichiamo
    # solo che i due nuovi tipi di rete si innestino correttamente nel resto del
    # pipeline (split, training loop, predict) usando l'embedding word2vec, che si
    # addestra all'istante sul piccolo corpus del test senza bisogno di download.
    cfg.word2vec_dim = 8
    clf = ESGPillarClassifier(config=cfg, embedding="word2vec", network=network)
    clf.fit(df, text_col="sentence", company_col="company", label_col="label")

    preds = clf.predict(["We invested in solar panels to cut emissions."])
    assert len(preds) == 1
    assert preds[0] in cfg.class_names


@patch("esgpillar.embeddings.load_contextual_model")
def test_finbert_embedding_routes_through_contextual_encoder(mock_load):
    """Verifica che embedding='finbert' chiami il loader giusto e costruisca
    un ContextualEmbeddingEncoder (senza scaricare nulla da HuggingFace)."""
    fake_tok = MagicMock()
    fake_tok.return_value = {
        "input_ids": MagicMock(to=lambda d: MagicMock()),
        "attention_mask": MagicMock(to=lambda d: MagicMock(cpu=lambda: MagicMock(numpy=lambda: __import__("numpy").ones((1, 4))))),
    }
    fake_model = MagicMock()
    fake_model.config.hidden_size = 16
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model
    mock_load.return_value = (fake_tok, fake_model)

    cfg = Config()
    clf = ESGPillarClassifier(config=cfg, embedding="finbert", network="DNN")
    encoder = clf._build_encoder(companies=[], device="cpu")

    mock_load.assert_called_once_with(cfg.finbert_model)
    assert encoder.dim == 16
    assert clf._tokenizer is None  # i modelli contestuali non usano il tokenizzatore condiviso
