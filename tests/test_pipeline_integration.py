"""Test di integrazione completo: fit -> predict -> save -> load.

Usa un embedding FINTO (KeyedVectors sintetico) al posto di un vero
download di GloVe/fastText, per validare la meccanica della pipeline
senza dipendere dalla rete.
"""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gensim")

from esgpillar.config import Config  # noqa: E402
from esgpillar.labeling import label_dataframe  # noqa: E402
from esgpillar.pipeline import ESGPillarClassifier  # noqa: E402


class _FakeKV(dict):
    @property
    def key_to_index(self):
        return self


@pytest.fixture
def fake_glove_loader(monkeypatch):
    rng = np.random.default_rng(0)
    vocab = ["carbon", "emission", "board", "audit", "employee", "safety",
            "water", "climate", "governance", "training", "compliance", "reduce"]
    fake_kv = _FakeKV({w: rng.normal(size=20).astype("float32") for w in vocab})

    def _fake_load_glove(config=None):
        return fake_kv

    monkeypatch.setattr("esgpillar.embeddings.load_glove", _fake_load_glove)
    monkeypatch.setattr("esgpillar.embeddings.load_fasttext_pretrained", _fake_load_glove)
    return fake_kv


def _toy_dataframe() -> pd.DataFrame:
    rows = []
    companies = ["Alpha", "Beta", "Gamma", "Delta"]
    templates = [
        ("We reduced carbon emissions through climate and water programs.", 0),
        ("Employee training and safety programs were expanded this year.", 1),
        ("The board improved audit and governance compliance procedures.", 2),
    ]
    for company in companies:
        for sentence, label in templates * 3:
            rows.append({"company": company, "sentence": sentence, "label": label})
    return pd.DataFrame(rows)


def test_fit_predict_save_load_roundtrip(tmp_path, fake_glove_loader):
    df = _toy_dataframe()
    cfg = Config(seed=0, epochs=3, batch_size=4, patience=1, glove_dim=20)

    clf = ESGPillarClassifier(config=cfg, embedding="glove", network="DNN")
    clf.fit(df, text_col="sentence", company_col="company", label_col="label")

    preds = clf.predict(["We reduced carbon emissions."])
    assert preds[0] in cfg.class_names

    proba = clf.predict_proba(["We reduced carbon emissions."])
    assert proba.shape == (1, 3)
    assert np.isclose(proba.sum(), 1.0, atol=1e-4)

    save_dir = tmp_path / "model"
    clf.save(save_dir)
    reloaded = ESGPillarClassifier.load(save_dir)

    preds_reloaded = reloaded.predict(["We reduced carbon emissions."])
    assert preds_reloaded == preds


def test_fit_uses_masked_sentences_when_available(tmp_path, fake_glove_loader):
    """label_dataframe() produce 'sentence_model' (mascherato); fit() deve
    rilevarlo e usarlo automaticamente, e predict() deve mascherare allo
    stesso modo le frasi nuove -- coerenza training/inferenza."""
    raw = pd.DataFrame({
        "company": ["Alpha", "Beta", "Gamma", "Delta"] * 3,
        "sentence": [
            "We reduced carbon emissions through climate and water programs.",
            "Employee training and safety programs were expanded this year.",
            "The board improved audit and governance compliance procedures.",
            "Employee training and safety programs were expanded this year.",
        ] * 3,
    })
    df = label_dataframe(raw)
    assert "sentence_model" in df.columns

    cfg = Config(seed=0, epochs=2, batch_size=4, patience=1, glove_dim=20)
    clf = ESGPillarClassifier(config=cfg, embedding="glove", network="DNN")
    clf.fit(df, text_col="sentence", company_col="company", label_col="label")

    assert clf._encode_col_used == "sentence_model"
    # non deve sollevare eccezioni: la mascheratura viene applicata anche qui
    preds = clf.predict(["The board approved a new anti-corruption policy."])
    assert preds[0] in cfg.class_names


def test_bigru_fit_predict_roundtrip(tmp_path, fake_glove_loader):
    """BiGRU e' sequenziale come BiLSTM: verifica che l'intero ciclo
    fit/predict/save/load funzioni anche con questa architettura."""
    df = _toy_dataframe()
    cfg = Config(seed=0, epochs=3, batch_size=4, patience=1, glove_dim=20)

    clf = ESGPillarClassifier(config=cfg, embedding="glove", network="BiGRU")
    clf.fit(df, text_col="sentence", company_col="company", label_col="label")

    preds = clf.predict(["We reduced carbon emissions."])
    assert preds[0] in cfg.class_names

    save_dir = tmp_path / "model_bigru"
    clf.save(save_dir)
    reloaded = ESGPillarClassifier.load(save_dir)
    assert reloaded.predict(["We reduced carbon emissions."]) == preds
