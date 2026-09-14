from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from esgpillar.config import Config
from esgpillar.labeling import (
    label_dataframe,
    label_dataframe_ensemble,
    label_dataframe_esgbert,
    label_dataframe_prototype,
    label_dataframe_tfidf,
    label_dataframe_zero_shot,
)

_DF = pd.DataFrame({"sentence": [
    "We reduced carbon emissions significantly through renewable energy investments.",
    "Employee training programs on workplace safety were expanded to all offices.",
    "The board of directors approved a new anti corruption compliance policy.",
    "This sentence has nothing to do with any relevant topic whatsoever today.",
]})


# --------------------------------------------------------------------- #
# 1. TF-IDF pesato
# --------------------------------------------------------------------- #
def test_label_dataframe_tfidf_matches_single_word_terms():
    result = label_dataframe_tfidf(_DF, Config(apply_leakage_mask=False))
    assert len(result) == 3  # la quarta frase non ha match, viene scartata
    assert result["esg"].tolist() == ["E", "S", "G"]


def test_label_dataframe_tfidf_matches_multiword_terms():
    """Verifica esplicita del bug trovato e corretto: un token_pattern
    troppo permissivo inghiottiva l'intera frase come un solo token,
    facendo sparire OGNI corrispondenza, anche per termini a una parola."""
    df = pd.DataFrame({"sentence": ["We rely on renewable energy across our value chain worker network."]})
    result = label_dataframe_tfidf(df, Config(apply_leakage_mask=False))
    assert len(result) == 1  # "renewable energy" (E1) e "value chain worker" (S2) devono essere trovati


def test_label_dataframe_tfidf_agrees_reasonably_with_default_method():
    result_default = label_dataframe(_DF, Config(apply_leakage_mask=False))
    result_tfidf = label_dataframe_tfidf(_DF, Config(apply_leakage_mask=False))
    # sullo stesso, piccolo corpus con frasi nette, i due metodi dovrebbero concordare
    assert result_default["esg"].tolist() == result_tfidf["esg"].tolist()


# --------------------------------------------------------------------- #
# 2. Similarita' a prototipi
# --------------------------------------------------------------------- #
class _FakeEncoder:
    def __call__(self, sentences):
        pooled = np.stack([
            np.random.default_rng(abs(hash(s)) % (2**32)).normal(size=10).astype("float32")
            for s in sentences
        ])
        return None, pooled, None


def test_label_dataframe_prototype_produces_valid_labels():
    result = label_dataframe_prototype(_DF, _FakeEncoder(), Config(apply_leakage_mask=False))
    assert set(result["esg"]).issubset({"E", "S", "G"})
    assert "esg_score" in result.columns


def test_label_dataframe_prototype_respects_threshold():
    """Con una soglia altissima, nessuna frase deve superarla -> tutto scartato."""
    result = label_dataframe_prototype(_DF, _FakeEncoder(), Config(apply_leakage_mask=False), threshold=999)
    assert len(result) == 0


# --------------------------------------------------------------------- #
# 3. Zero-shot
# --------------------------------------------------------------------- #
def test_label_dataframe_zero_shot_with_mocked_pipeline():
    import esgpillar.text_mining as tm_module

    fake_pipeline = MagicMock(
        return_value={"labels": ["Environmental", "Social", "Governance"], "scores": [0.7, 0.2, 0.1]}
    )
    tm_module.zero_shot_classify._pipeline_cache.clear()
    with patch("esgpillar.text_mining._load_zero_shot_pipeline", return_value=fake_pipeline):
        result = label_dataframe_zero_shot(_DF, Config(apply_leakage_mask=False))
    assert len(result) == len(_DF)  # zero-shot forza sempre un'etichetta, mai "Other"
    assert set(result["esg"]) == {"E"}  # il mock restituisce sempre "Environmental" come primo


# --------------------------------------------------------------------- #
# 4. ESGBERT diretto
# --------------------------------------------------------------------- #
class _FakeEncoding(dict):
    def to(self, device):
        return self


def test_label_dataframe_esgbert_with_mocked_model():
    fake_tok = MagicMock()
    fake_tok.side_effect = lambda batch, truncation=True, padding=True, return_tensors="pt": _FakeEncoding(
        input_ids=torch.zeros(len(batch), 5, dtype=torch.long),
        attention_mask=torch.ones(len(batch), 5, dtype=torch.long),
    )
    fake_model = MagicMock()
    fake_model.eval.return_value = fake_model
    fake_model.config.id2label = {0: "Environmental", 1: "Social", 2: "Governance", 3: "None"}

    def fake_call(**kwargs):
        n = kwargs["input_ids"].shape[0]
        logits = torch.zeros(n, 4)
        logits[:, 3] = 5.0  # forza "None" -> deve diventare "Other" e sparire
        return MagicMock(logits=logits)

    fake_model.side_effect = fake_call

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok), \
        patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=fake_model):
        result = label_dataframe_esgbert(_DF, Config(apply_leakage_mask=False))

    assert len(result) == 0  # "None" -> "Other" -> tutte le righe scartate


def test_label_dataframe_esgbert_maps_environmental_correctly():
    fake_tok = MagicMock()
    fake_tok.side_effect = lambda batch, truncation=True, padding=True, return_tensors="pt": _FakeEncoding(
        input_ids=torch.zeros(len(batch), 5, dtype=torch.long),
        attention_mask=torch.ones(len(batch), 5, dtype=torch.long),
    )
    fake_model = MagicMock()
    fake_model.eval.return_value = fake_model
    fake_model.config.id2label = {0: "Environmental", 1: "Social", 2: "Governance", 3: "None"}

    def fake_call(**kwargs):
        n = kwargs["input_ids"].shape[0]
        logits = torch.zeros(n, 4)
        logits[:, 0] = 5.0  # forza "Environmental"
        return MagicMock(logits=logits)

    fake_model.side_effect = fake_call

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok), \
        patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=fake_model):
        result = label_dataframe_esgbert(_DF, Config(apply_leakage_mask=False))

    assert len(result) == len(_DF)
    assert set(result["esg"]) == {"E"}


# --------------------------------------------------------------------- #
# 5. Ensemble / voto di maggioranza
# --------------------------------------------------------------------- #
def test_label_dataframe_ensemble_majority_vote():
    labels_a = ["E", "S", "G", "Other"]
    labels_b = ["E", "S", "S", "Other"]
    labels_c = ["S", "S", "G", "E"]
    result = label_dataframe_ensemble(_DF, [labels_a, labels_b, labels_c], Config(apply_leakage_mask=False))

    # riga 0: voti E,E,S -> maggioranza E
    # riga 1: voti S,S,S -> maggioranza S
    # riga 2: voti G,S,G -> maggioranza G
    # riga 3: voti Other,Other,E -> maggioranza Other -> scartata
    assert len(result) == 3
    assert result["esg"].tolist() == ["E", "S", "G"]


def test_label_dataframe_ensemble_no_majority_gives_other():
    labels_a = ["E"]
    labels_b = ["S"]
    labels_c = ["G"]
    df_single = pd.DataFrame({"sentence": ["some sentence"]})
    result = label_dataframe_ensemble(df_single, [labels_a, labels_b, labels_c], Config(apply_leakage_mask=False))
    assert len(result) == 0  # nessuna maggioranza assoluta (1 voto su 3 ciascuno) -> Other


def test_label_dataframe_ensemble_mismatched_length_raises():
    with pytest.raises(ValueError, match="stessa lunghezza"):
        label_dataframe_ensemble(_DF, [["E", "S"]], Config(apply_leakage_mask=False))
