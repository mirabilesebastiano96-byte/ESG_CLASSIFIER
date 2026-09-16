import tempfile

import pandas as pd
import pytest
import torch
import torch.nn as nn

from esgpillar import Config, ESGPillarClassifier

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


class _PooledNet(nn.Module):
    def __init__(self, d, c=3, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, c)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class _SequenceNet(nn.Module):
    def __init__(self, d, c=3, hidden=16):
        super().__init__()
        self.gru = nn.GRU(d, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, c)

    def forward(self, x, lengths=None):
        _, h = self.gru(x)
        return self.fc(h[0])


class _NotAModule:
    """Ne' torch.nn.Module ne' un vero estimatore -- per il test dell'errore."""


def _df():
    return pd.DataFrame(_ROWS)


@pytest.mark.parametrize("input_type", ["pooled", "sequence"])
def test_custom_neural_network_fit_predict(input_type):
    pytest.importorskip("gensim")
    net_cls = _PooledNet if input_type == "pooled" else _SequenceNet
    cfg = Config(word2vec_dim=8, val_size=0.25, epochs=2, patience=1, seed=1)
    clf = ESGPillarClassifier(config=cfg, embedding="word2vec", network=net_cls, custom_input_type=input_type)
    clf.fit(_df(), text_col="sentence", company_col="company", label_col="label")

    pred = clf.predict(["We invested in solar panels."])
    assert pred[0] in cfg.class_names
    assert clf._is_classical is False
    assert clf._network_display_name() == net_cls.__name__


def test_custom_neural_network_applies_network_kwargs():
    pytest.importorskip("gensim")
    cfg = Config(word2vec_dim=8, val_size=0.25, epochs=1, patience=1, seed=1)
    clf = ESGPillarClassifier(
        config=cfg, embedding="word2vec", network=_PooledNet,
        network_kwargs={"hidden": 32}, custom_input_type="pooled",
    )
    clf.fit(_df(), text_col="sentence", company_col="company", label_col="label")
    assert clf._model.fc1.out_features == 32


def test_custom_neural_network_without_input_type_raises():
    with pytest.raises(ValueError, match="custom_input_type"):
        ESGPillarClassifier(embedding="word2vec", network=_PooledNet)


def test_custom_neural_network_with_invalid_input_type_raises():
    with pytest.raises(ValueError, match="custom_input_type"):
        ESGPillarClassifier(embedding="word2vec", network=_PooledNet, custom_input_type="invalid")


def test_passing_a_non_module_class_raises_clear_type_error():
    from sklearn.ensemble import RandomForestClassifier

    with pytest.raises(TypeError, match="ISTANZA"):
        ESGPillarClassifier(embedding="word2vec", network=RandomForestClassifier)


def test_custom_classical_model_instance():
    pytest.importorskip("gensim")
    from sklearn.ensemble import RandomForestClassifier

    cfg = Config(word2vec_dim=8, val_size=0.25, seed=1)
    custom_model = RandomForestClassifier(n_estimators=17, max_depth=3, random_state=0)
    clf = ESGPillarClassifier(config=cfg, embedding="word2vec", network=custom_model)
    clf.fit(_df(), text_col="sentence", company_col="company", label_col="label")

    pred = clf.predict(["We invested in solar panels."])
    assert pred[0] in cfg.class_names
    assert clf._is_classical is True
    assert clf._model is custom_model
    assert clf._model.n_estimators == 17
    assert clf.history_ is None


def test_custom_classical_model_plot_history_raises_clear_error():
    pytest.importorskip("gensim")
    pytest.importorskip("matplotlib")
    from sklearn.linear_model import LogisticRegression

    cfg = Config(word2vec_dim=8, val_size=0.25, seed=1)
    clf = ESGPillarClassifier(config=cfg, embedding="word2vec", network=LogisticRegression(max_iter=200))
    clf.fit(_df(), text_col="sentence", company_col="company", label_col="label")
    with pytest.raises(RuntimeError, match="classico"):
        clf.plot_history()


def test_custom_neural_network_save_load_roundtrip():
    pytest.importorskip("gensim")
    cfg = Config(word2vec_dim=8, val_size=0.25, epochs=2, patience=1, seed=1)
    clf = ESGPillarClassifier(
        config=cfg, embedding="word2vec", network=_PooledNet,
        network_kwargs={"hidden": 32}, custom_input_type="pooled",
    )
    clf.fit(_df(), text_col="sentence", company_col="company", label_col="label")
    pred_before = clf.predict(["We invested in solar panels."])

    with tempfile.TemporaryDirectory() as tmpdir:
        clf.save(tmpdir)
        reloaded = ESGPillarClassifier.load(tmpdir)
        pred_after = reloaded.predict(["We invested in solar panels."])

    assert pred_after == pred_before
    assert reloaded._network_display_name() == "_PooledNet"


def test_custom_classical_instance_save_load_roundtrip():
    pytest.importorskip("gensim")
    from sklearn.linear_model import LogisticRegression

    cfg = Config(word2vec_dim=8, val_size=0.25, seed=1)
    clf = ESGPillarClassifier(config=cfg, embedding="word2vec", network=LogisticRegression(max_iter=200, C=0.5))
    clf.fit(_df(), text_col="sentence", company_col="company", label_col="label")
    pred_before = clf.predict(["We invested in solar panels."])

    with tempfile.TemporaryDirectory() as tmpdir:
        clf.save(tmpdir)
        reloaded = ESGPillarClassifier.load(tmpdir)
        pred_after = reloaded.predict(["We invested in solar panels."])

    assert pred_after == pred_before
    assert reloaded._is_classical is True
