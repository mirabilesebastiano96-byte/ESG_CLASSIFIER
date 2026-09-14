import pandas as pd

from esgpillar.config import Config
from esgpillar.labeling import ESRS_TO_PILLAR, label_dataframe, label_sentence, mask_lexicon


def test_label_sentence_environmental():
    topic, score = label_sentence("We reduced our carbon emissions significantly last year.")
    assert topic in ESRS_TO_PILLAR
    assert ESRS_TO_PILLAR[topic] == "E"
    assert score > 0


def test_label_sentence_governance():
    topic, score = label_sentence("The board conducted an audit of our compliance program.")
    assert ESRS_TO_PILLAR[topic] == "G"


def test_label_sentence_no_match_is_other():
    topic, score = label_sentence("We had lunch in the cafeteria yesterday.")
    assert topic == "Other"
    assert score == 0.0


def test_label_dataframe_end_to_end():
    df = pd.DataFrame({
        "company": ["A", "A", "B"],
        "sentence": [
            "We reduced carbon emissions and water consumption this year.",
            "The board approved a new anti-corruption compliance policy.",
            "This sentence has nothing to do with sustainability topics at all today.",
        ],
    })
    out = label_dataframe(df)
    assert set(out["esg"]).issubset({"E", "S", "G"})
    assert "label" in out.columns
    assert len(out) <= len(df)  # le frasi "Other" vengono scartate


def test_mask_lexicon_removes_label_generating_words():
    s = "The board conducted an audit of our compliance program."
    masked = mask_lexicon(s)
    assert "board" not in masked.lower()
    assert "audit" not in masked.lower()
    assert masked != s


def test_label_dataframe_applies_mask_by_default():
    df = pd.DataFrame({
        "company": ["A"],
        "sentence": ["The board approved a new anti-corruption compliance policy."],
    })
    out = label_dataframe(df)
    assert "sentence_model" in out.columns
    assert out["sentence_model"].iloc[0] != out["sentence"].iloc[0]
    assert "board" not in out["sentence_model"].iloc[0].lower()


def test_label_dataframe_mask_can_be_disabled():
    df = pd.DataFrame({
        "company": ["A"],
        "sentence": ["The board approved a new anti-corruption compliance policy."],
    })
    out = label_dataframe(df, config=Config(apply_leakage_mask=False))
    assert "sentence_model" not in out.columns
