from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("gensim")

import torch  # noqa: E402

from esgpillar import Config, ESGPillarClassifier  # noqa: E402
from esgpillar.embeddings import (  # noqa: E402
    doc2vec_encode,
    load_doc2vec,
    load_fasttext_corpus,
    pretrained_encode,
    sentence_transformer_encode,
)
from esgpillar.models import GRUNet, MLPMixerNet, ResCNN  # noqa: E402

_TOKEN_LISTS = [
    ["carbon", "emission", "reduce", "climate", "energy"],
    ["board", "audit", "governance", "compliance", "policy"],
    ["employee", "safety", "training", "diversity", "workforce"],
] * 4

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
# fastText addestrato sul corpus: il vantaggio vero e' la gestione OOV via subword
# --------------------------------------------------------------------- #
def test_load_fasttext_corpus_handles_oov_via_subwords():
    cfg = Config(fasttext_corpus_dim=16, fasttext_corpus_epochs=5, seed=0)
    wv = load_fasttext_corpus(_TOKEN_LISTS, cfg)

    assert "carbon" in wv.key_to_index
    # "carbonizing" non e' mai comparsa nel training, ma fastText deve
    # comunque produrle un vettore tramite le subword (i suoi n-grammi)
    assert "carbonizing" not in wv.key_to_index
    seq, pooled, lengths = pretrained_encode([["carbonizing", "emission"]], wv, dim=16, max_len=5)
    assert lengths[0] == 2, "la parola OOV avrebbe dovuto essere gestita via subword, non scartata"


# --------------------------------------------------------------------- #
# Doc2Vec: un vettore per frase, non per parola
# --------------------------------------------------------------------- #
def test_load_doc2vec_and_encode():
    cfg = Config(doc2vec_dim=10, doc2vec_epochs=5, seed=0)
    sentences = [t for t in _TOKEN_LISTS]
    model = load_doc2vec(sentences, cfg)

    seq, pooled, lengths = doc2vec_encode(_TOKEN_LISTS[:3], model, dim=10, max_len=8)
    assert pooled.shape == (3, 10)
    assert seq.shape == (3, 8, 10)
    # un solo "token" reale (il vettore di frase), il resto e' padding nullo
    assert (seq[:, 1:] == 0).all()
    assert np.allclose(seq[:, 0], pooled)


def test_doc2vec_full_pipeline_fit_predict_save_load_roundtrip(tmp_path):
    df = pd.DataFrame(_ROWS)
    cfg = Config(doc2vec_dim=10, val_size=0.34, epochs=2, patience=1, doc2vec_epochs=5)

    clf = ESGPillarClassifier(config=cfg, embedding="doc2vec", network="DNN")
    clf.fit(df, text_col="sentence", company_col="company", label_col="label")

    test_sentences = ["We invested in solar panels to cut emissions."]
    pred_before = clf.predict(test_sentences)

    save_path = tmp_path / "model_doc2vec"
    clf.save(save_path)
    assert (save_path / "doc2vec.model").exists()

    reloaded = ESGPillarClassifier.load(save_path)
    pred_after = reloaded.predict(test_sentences)
    assert pred_after == pred_before


# --------------------------------------------------------------------- #
# Sentence-Transformers: stessa idea di Doc2Vec, ma pre-addestrato (mockato)
# --------------------------------------------------------------------- #
def test_sentence_transformer_encode_with_mocked_model():
    fake_model = MagicMock()
    fake_model.get_sentence_embedding_dimension.return_value = 6
    fake_model.encode.return_value = np.random.randn(2, 6).astype("float32")

    seq, pooled, lengths = sentence_transformer_encode(["frase uno", "frase due"], fake_model, max_len=5)
    assert pooled.shape == (2, 6)
    assert seq.shape == (2, 5, 6)
    assert (seq[:, 1:] == 0).all()
    fake_model.encode.assert_called_once()


@patch("esgpillar.embeddings.load_sentence_transformer")
def test_sbert_routes_through_pipeline_build_encoder(mock_load):
    fake_model = MagicMock()
    fake_model.get_sentence_embedding_dimension.return_value = 6
    mock_load.return_value = fake_model

    cfg = Config()
    clf = ESGPillarClassifier(config=cfg, embedding="sbert", network="DNN")
    encoder = clf._build_encoder(companies=[], device="cpu")

    mock_load.assert_called_once_with(cfg)
    assert encoder.dim == 6
    assert clf._tokenizer is None  # sbert tokenizza internamente, non serve il tokenizzatore condiviso


# --------------------------------------------------------------------- #
# Nuove reti: GRU, ResCNN, MLPMixer -- attraverso il pipeline completo
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("network", ["GRU", "ResCNN", "MLPMixer"])
def test_new_networks_through_full_pipeline(network):
    df = pd.DataFrame(_ROWS)
    cfg = Config(word2vec_dim=8, val_size=0.34, epochs=2, patience=1, max_len=16)

    clf = ESGPillarClassifier(config=cfg, embedding="word2vec", network=network)
    clf.fit(df, text_col="sentence", company_col="company", label_col="label")

    preds = clf.predict(["We invested in solar panels to cut emissions."])
    assert len(preds) == 1
    assert preds[0] in cfg.class_names


def test_gru_is_unidirectional_half_the_params_of_bigru():
    from esgpillar.models import BiGRU

    gru = GRUNet(d=10, h=32)
    bigru = BiGRU(d=10, h=32)
    n_gru = sum(p.numel() for p in gru.gru.parameters())
    n_bigru = sum(p.numel() for p in bigru.gru.parameters())
    assert n_gru < n_bigru


def test_rescnn_forward_pass_with_high_dim_input_uses_projection():
    model = ResCNN(d=768, proj_dim=128)
    assert model.proj is not None
    x = torch.randn(3, 20, 768)
    out = model(x)
    assert out.shape == (3, 3)
