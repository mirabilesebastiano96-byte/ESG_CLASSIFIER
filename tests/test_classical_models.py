import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from esgpillar import Config, ESGPillarClassifier  # noqa: E402
from esgpillar.classical import CLASSICAL_BUILDERS, NON_NEGATIVE_ONLY, build_classical_model  # noqa: E402

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
# Ogni modello classico: costruzione, fit, predict_proba (dati sintetici)
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(CLASSICAL_BUILDERS))
def test_every_classical_model_fits_and_predicts(name):
    if name in ("XGBoost", "LightGBM"):
        pytest.importorskip(name.lower())

    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(60, 10)).astype("float32")
    y_tr = rng.integers(0, 3, 60)
    X_te = rng.normal(size=(10, 10)).astype("float32")
    if name in NON_NEGATIVE_ONLY:
        X_tr, X_te = np.abs(X_tr), np.abs(X_te)

    cfg = Config(seed=0)
    model = build_classical_model(name, cfg)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)

    assert proba.shape == (10, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4), "le probabilita' devono sommare a 1"


def test_build_classical_model_unknown_name_raises():
    with pytest.raises(ValueError, match="non riconosciuto"):
        build_classical_model("NotAModel")


# --------------------------------------------------------------------- #
# Attraverso il pipeline completo: fit -> predict -> save -> load
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("network", ["LogisticRegression", "RandomForest", "GaussianNB", "Voting"])
def test_classical_model_through_full_pipeline_roundtrip(tmp_path, network):
    pytest.importorskip("gensim")
    df = pd.DataFrame(_ROWS)
    cfg = Config(word2vec_dim=8, val_size=0.34, seed=0)

    clf = ESGPillarClassifier(config=cfg, embedding="word2vec", network=network)
    clf.fit(df, text_col="sentence", company_col="company", label_col="label")

    assert clf._is_classical is True
    assert clf.history_ is None

    test_sentences = ["We invested in solar panels to cut emissions."]
    pred_before = clf.predict(test_sentences)
    assert pred_before[0] in cfg.class_names

    save_path = tmp_path / f"model_{network}"
    clf.save(save_path)
    assert (save_path / "classical_model.pkl").exists()
    assert not (save_path / "model.pt").exists(), "un modello classico non deve produrre model.pt"

    reloaded = ESGPillarClassifier.load(save_path)
    assert reloaded._is_classical is True
    pred_after = reloaded.predict(test_sentences)
    assert pred_after == pred_before


def test_classical_model_plot_history_raises_clear_error():
    pytest.importorskip("gensim")
    pytest.importorskip("matplotlib")
    df = pd.DataFrame(_ROWS)
    cfg = Config(word2vec_dim=8, val_size=0.34, seed=0)

    clf = ESGPillarClassifier(config=cfg, embedding="word2vec", network="LogisticRegression")
    clf.fit(df, text_col="sentence", company_col="company", label_col="label")

    with pytest.raises(RuntimeError, match="modello classico"):
        clf.plot_history()
