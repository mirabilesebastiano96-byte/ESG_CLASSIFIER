import pandas as pd
import pytest

from esgpillar import Config, ESGPillarClassifier
from esgpillar.classical import build_classical_model


# --------------------------------------------------------------------- #
# classical.py: override diretti
# --------------------------------------------------------------------- #
def test_build_classical_model_default_no_override():
    model = build_classical_model("RandomForest", Config())
    assert model.n_estimators == 200  # valore di default


def test_build_classical_model_with_override():
    model = build_classical_model("RandomForest", Config(), n_estimators=500, max_depth=10)
    assert model.n_estimators == 500
    assert model.max_depth == 10


def test_build_classical_model_override_applies_to_wrapped_estimator():
    """LinearSVM e' avvolto in CalibratedClassifierCV: l'override deve
    arrivare al LinearSVC interno, non essere ignorato o applicato al wrapper."""
    model = build_classical_model("LinearSVM", Config(), C=0.5)
    assert model.estimator.C == 0.5


def test_build_classical_model_voting_rejects_overrides():
    with pytest.raises(ValueError, match="meta-ensemble"):
        build_classical_model("Voting", Config(), n_estimators=500)


def test_build_classical_model_stacking_rejects_overrides():
    with pytest.raises(ValueError, match="meta-ensemble"):
        build_classical_model("Stacking", Config(), n_estimators=500)


def test_build_classical_model_voting_without_overrides_still_works():
    model = build_classical_model("Voting", Config())
    assert type(model).__name__ == "VotingClassifier"


# --------------------------------------------------------------------- #
# Attraverso ESGPillarClassifier
# --------------------------------------------------------------------- #
_ROWS = [
    {"company": "Acme", "sentence": "We reduced carbon emissions through renewable energy programs.", "label": 0},
    {"company": "Acme", "sentence": "Water consumption decreased across our manufacturing facilities.", "label": 0},
    {"company": "Beta", "sentence": "Biodiversity protection measures were implemented near our sites.", "label": 0},
    {"company": "Acme", "sentence": "Employee training programs on workplace safety were expanded.", "label": 1},
    {"company": "Beta", "sentence": "Diversity and inclusion initiatives improved representation.", "label": 1},
    {"company": "Gamma", "sentence": "Worker health and safety incidents declined after training.", "label": 1},
    {"company": "Acme", "sentence": "The board approved a new anti corruption compliance policy.", "label": 2},
    {"company": "Beta", "sentence": "Audit committee strengthened oversight of governance procedures.", "label": 2},
    {"company": "Gamma", "sentence": "Governance structure was revised to increase board independence.", "label": 2},
]


def _fit_with_retry(clf, df, max_tries=6):
    """I dataset di test sono piccoli: lo split per azienda a volte lascia
    una classe fuori dal train. Riprova con semi diversi (non e' un difetto
    del codice sotto test, solo un artefatto dei dati sintetici minuscoli)."""
    last_error = None
    for seed in range(max_tries):
        clf.config.seed = seed
        try:
            clf.fit(df, text_col="sentence", company_col="company", label_col="label")
            return
        except ValueError as e:
            last_error = e
    raise last_error


def test_esgpillarclassifier_default_network_kwargs_is_empty():
    clf = ESGPillarClassifier(config=Config(), embedding="word2vec", network="BiLSTM")
    assert clf.network_kwargs == {}


def test_esgpillarclassifier_neural_network_kwargs_applied():
    pytest.importorskip("gensim")
    df = pd.DataFrame(_ROWS)
    cfg = Config(word2vec_dim=8, val_size=0.25, epochs=2, patience=1)
    clf = ESGPillarClassifier(config=cfg, embedding="word2vec", network="BiLSTM",
                              network_kwargs={"h": 256, "p": 0.6})
    _fit_with_retry(clf, df)
    assert clf._model.lstm.hidden_size == 256
    assert clf._model.drop.p == 0.6


def test_esgpillarclassifier_classical_network_kwargs_applied():
    pytest.importorskip("gensim")
    df = pd.DataFrame(_ROWS)
    cfg = Config(word2vec_dim=8)
    clf = ESGPillarClassifier(config=cfg, embedding="word2vec", network="RandomForest",
                              network_kwargs={"n_estimators": 50, "max_depth": 5})
    clf.fit(df, text_col="sentence", company_col="company", label_col="label")
    assert clf._model.n_estimators == 50
    assert clf._model.max_depth == 5


def test_esgpillarclassifier_network_kwargs_survive_save_load(tmp_path):
    pytest.importorskip("gensim")
    df = pd.DataFrame(_ROWS)
    cfg = Config(word2vec_dim=8, val_size=0.25, epochs=2, patience=1)
    clf = ESGPillarClassifier(config=cfg, embedding="word2vec", network="BiLSTM",
                              network_kwargs={"h": 256, "p": 0.6})
    _fit_with_retry(clf, df)

    save_path = tmp_path / "model_kwargs"
    clf.save(save_path)
    reloaded = ESGPillarClassifier.load(save_path)

    assert reloaded.network_kwargs == {"h": 256, "p": 0.6}
    assert reloaded._model.lstm.hidden_size == 256
