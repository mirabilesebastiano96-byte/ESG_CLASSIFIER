"""Test di integrazione per l'embedding contestuale (ESGBERT/FinBERT-ESG).

Usa un tokenizer/modello transformer FINTI (stessa interfaccia di
HuggingFace) al posto di un vero download, per validare la meccanica
della pipeline senza dipendere dalla rete.
"""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from esgpillar.config import Config  # noqa: E402
from esgpillar.pipeline import ESGPillarClassifier  # noqa: E402

_FAKE_DIM = 16


class _FakeConfig:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size


class _FakeModel:
    def __init__(self, dim=_FAKE_DIM):
        self.config = _FakeConfig(dim)

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        b, seq_len = input_ids.shape
        # deterministico rispetto al contenuto (non al momento della chiamata):
        # stesso input -> stesso output, indipendentemente da quanti numeri
        # casuali sono stati generati prima altrove nel processo di test.
        # (input_ids e' sempre zero in questo tokenizer finto: uso
        # attention_mask, che riflette la lunghezza reale della frase)
        seed = int(attention_mask.sum().item()) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        hidden = torch.randn(b, seq_len, self.config.hidden_size, generator=gen)
        return type("Out", (), {"last_hidden_state": hidden})()


class _FakeTokenizer:
    def __call__(self, batch, truncation=True, max_length=64, padding=True, return_tensors="pt"):
        lengths = [min(len(s.split()) + 2, max_length) for s in batch]
        seq_len = max(lengths)
        input_ids = torch.zeros(len(batch), seq_len, dtype=torch.long)
        attn = torch.zeros(len(batch), seq_len, dtype=torch.long)
        for i, length in enumerate(lengths):
            attn[i, :length] = 1

        class _Enc(dict):
            def to(self, device):
                return self

        return _Enc(input_ids=input_ids, attention_mask=attn)


@pytest.fixture
def fake_esgbert_loader(monkeypatch):
    def _fake_load_esgbert(config=None):
        return _FakeTokenizer(), _FakeModel()

    monkeypatch.setattr("esgpillar.embeddings.load_esgbert", _fake_load_esgbert)


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


def test_esgbert_fit_predict_save_load_roundtrip(tmp_path, fake_esgbert_loader):
    df = _toy_dataframe()
    cfg = Config(seed=0, epochs=3, batch_size=4, patience=1, esgbert_batch=4)

    clf = ESGPillarClassifier(config=cfg, embedding="esgbert", network="DNN")
    clf.fit(df, text_col="sentence", company_col="company", label_col="label")
    assert clf._tokenizer is None  # ESGBERT non usa il tokenizzatore condiviso

    preds = clf.predict(["We reduced carbon emissions."])
    assert preds[0] in cfg.class_names

    proba = clf.predict_proba(["We reduced carbon emissions."])
    assert proba.shape == (1, 3)
    assert np.isclose(proba.sum(), 1.0, atol=1e-4)

    save_dir = tmp_path / "model_esgbert"
    clf.save(save_dir)
    reloaded = ESGPillarClassifier.load(save_dir)
    assert reloaded.predict(["We reduced carbon emissions."]) == preds
