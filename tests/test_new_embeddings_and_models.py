import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

import torch  # noqa: E402

from esgpillar.config import Config  # noqa: E402
from esgpillar.embeddings import load_lsa, load_word2vec, pretrained_encode  # noqa: E402
from esgpillar.models import (  # noqa: E402
    NET_BUILDERS,
    AttentionPoolNet,
    TransformerEncoderNet,
)

_TOKEN_LISTS = [
    ["carbon", "emission", "reduce", "climate"],
    ["board", "audit", "governance", "compliance"],
    ["employee", "safety", "training", "diversity"],
] * 4


def test_load_word2vec_produces_usable_vectors():
    cfg = Config(word2vec_dim=16, word2vec_epochs=5, seed=0)
    wv = load_word2vec(_TOKEN_LISTS, cfg)
    assert "carbon" in wv.key_to_index
    assert wv["carbon"].shape == (16,)
    seq, pooled, lengths = pretrained_encode(_TOKEN_LISTS, wv, dim=16, max_len=6)
    assert pooled.shape == (len(_TOKEN_LISTS), 16)
    assert seq.shape == (len(_TOKEN_LISTS), 6, 16)


def test_load_lsa_produces_word_level_vectors():
    cfg = Config(lsa_dim=8, lsa_min_df=1, seed=0)
    kv = load_lsa(_TOKEN_LISTS, cfg)
    assert "carbon" in kv.key_to_index
    assert kv["carbon"].shape == (8,)
    seq, pooled, lengths = pretrained_encode(_TOKEN_LISTS, kv, dim=8, max_len=6)
    assert pooled.shape == (len(_TOKEN_LISTS), 8)


def test_net_builders_includes_new_architectures():
    assert "Transformer" in NET_BUILDERS
    assert "AttentionPool" in NET_BUILDERS
    assert {"DNN", "Conv1D", "BiLSTM", "BiGRU", "Transformer", "AttentionPool"}.issubset(NET_BUILDERS)


def test_transformer_encoder_forward_pass():
    d = 8
    model = TransformerEncoderNet(d=d, c=3, n_heads=2, n_layers=1)
    x = torch.randn(4, 10, d)
    out = model(x)
    assert out.shape == (4, 3)


def test_transformer_encoder_requires_divisible_heads():
    with pytest.raises(AssertionError):
        TransformerEncoderNet(d=7, n_heads=2)


def test_attention_pool_forward_pass():
    d = 12
    model = AttentionPoolNet(d=d, c=3)
    x = torch.randn(5, 7, d)
    out = model(x)
    assert out.shape == (5, 3)


def test_attention_pool_weights_sum_to_one():
    d = 6
    model = AttentionPoolNet(d=d, c=3)
    x = torch.randn(2, 4, d)
    weights = torch.softmax(model.attn(x).squeeze(-1), dim=1)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-5)
