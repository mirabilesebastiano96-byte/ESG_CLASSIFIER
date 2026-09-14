from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from esgpillar import Config, ESGPillarClassifier
from esgpillar.embeddings import ElmoEncoder
from esgpillar.tokenization import Tokenizer


def test_load_elmo_raises_clear_error_when_path_missing(tmp_path):
    from esgpillar.embeddings import load_elmo

    cfg = Config(elmo_model_path=str(tmp_path / "does_not_exist"))
    with pytest.raises(FileNotFoundError, match="non trovato"):
        load_elmo(cfg)


def test_elmo_encoder_produces_correct_shapes():
    """ElmoEncoder adatta get_elmo_vectors/get_elmo_vector_average
    all'interfaccia comune (seq, pooled, lengths), senza bisogno di un
    vero simple_elmo/TensorFlow installato: il modello e' mockato."""
    dim = 16
    fake_model = MagicMock()
    # get_elmo_vectors: (N, L_batch, dim) -- L_batch = lunghezza della frase piu' lunga nel batch
    fake_model.get_elmo_vectors.return_value = np.random.randn(2, 4, dim).astype("float32")
    fake_model.get_elmo_vector_average.return_value = np.random.randn(2, dim).astype("float32")

    tok = Tokenizer.from_companies(["Acme"])
    encoder = ElmoEncoder(model=fake_model, tokenizer=tok, dim=dim, max_len=10)

    seq, pooled, lengths = encoder(["We reduced carbon emissions.", "The board approved a new policy."])
    assert seq.shape == (2, 10, dim)
    assert pooled.shape == (2, dim)
    assert lengths.shape == (2,)
    fake_model.get_elmo_vectors.assert_called_once()
    fake_model.get_elmo_vector_average.assert_called_once()


def test_elmo_encoder_handles_empty_tokenization_gracefully():
    """Una frase che il tokenizzatore condiviso riduce a lista vuota (es.
    solo stopword/nome azienda) non deve far crashare ElmoModel."""
    dim = 8
    fake_model = MagicMock()
    fake_model.get_elmo_vectors.return_value = np.zeros((1, 1, dim), dtype="float32")
    fake_model.get_elmo_vector_average.return_value = np.zeros((1, dim), dtype="float32")

    tok = Tokenizer.from_companies(["Acme"])
    encoder = ElmoEncoder(model=fake_model, tokenizer=tok, dim=dim, max_len=5)
    seq, pooled, lengths = encoder(["Acme Acme Acme"])  # dopo tokenizzazione, verosimilmente vuota
    assert seq.shape == (1, 5, dim)


@patch("esgpillar.embeddings.load_elmo")
def test_elmo_routes_through_pipeline_build_encoder(mock_load):
    fake_model = MagicMock()
    mock_load.return_value = fake_model

    cfg = Config(elmo_dim=16)
    clf = ESGPillarClassifier(config=cfg, embedding="elmo", network="DNN")
    encoder = clf._build_encoder(companies=["Acme"], device="cpu")

    mock_load.assert_called_once_with(cfg)
    assert encoder.dim == 16
    assert clf._tokenizer is not None  # elmo usa il tokenizzatore condiviso, a differenza di BERT-style
