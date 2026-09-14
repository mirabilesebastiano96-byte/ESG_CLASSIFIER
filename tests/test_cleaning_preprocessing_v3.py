import pandas as pd
import pytest

from esgpillar.cleaning import (
    find_near_duplicates,
    fix_encoding,
    is_tabular_fragment,
    remove_near_duplicates,
    remove_tabular_fragments,
)


# --------------------------------------------------------------------- #
# Riparazione encoding (mojibake)
# --------------------------------------------------------------------- #
def test_fix_encoding_repairs_real_mojibake():
    pytest.importorskip("ftfy")
    original = "We reduced emissions significantly — a major milestone for société."
    mojibake = original.encode("utf-8").decode("latin-1")
    assert fix_encoding(mojibake) == original


def test_fix_encoding_clean_text_unchanged():
    pytest.importorskip("ftfy")
    text = "We reduced emissions significantly this year."
    assert fix_encoding(text) == text


# --------------------------------------------------------------------- #
# Deduplica fuzzy
# --------------------------------------------------------------------- #
def test_find_near_duplicates_detects_similar_sentences():
    pytest.importorskip("rapidfuzz")
    texts = [
        "We reduced carbon emissions significantly this year.",
        "We reduced our carbon emissions significantly this year.",
        "Completely unrelated sentence about something else entirely.",
    ]
    pairs = find_near_duplicates(texts, threshold=85)
    assert len(pairs) == 1
    assert pairs[0][0] == 0 and pairs[0][1] == 1
    assert pairs[0][2] >= 85


def test_find_near_duplicates_no_matches_below_threshold():
    pytest.importorskip("rapidfuzz")
    texts = ["First unrelated sentence.", "Second completely different sentence."]
    pairs = find_near_duplicates(texts, threshold=95)
    assert pairs == []


def test_remove_near_duplicates_keeps_first_occurrence():
    pytest.importorskip("rapidfuzz")
    df = pd.DataFrame({"sentence": [
        "We reduced carbon emissions significantly this year.",
        "We reduced our carbon emissions significantly this year.",
        "The board approved a new governance policy.",
    ]})
    result = remove_near_duplicates(df, threshold=85)
    assert len(result) == 2
    assert result["sentence"].iloc[0] == "We reduced carbon emissions significantly this year."


# --------------------------------------------------------------------- #
# Rilevamento frammenti tabellari
# --------------------------------------------------------------------- #
def test_is_tabular_fragment_detects_table_row():
    assert is_tabular_fragment("12.5 34.2 8.9 -- 156.3 2.1%") is True


def test_is_tabular_fragment_false_for_real_sentence_with_numbers():
    assert is_tabular_fragment("Revenue grew by 25% this year thanks to strong demand.") is False


def test_is_tabular_fragment_empty_text():
    assert is_tabular_fragment("") is True


def test_remove_tabular_fragments():
    df = pd.DataFrame({"sentence": [
        "12.5 34.2 8.9 -- 156.3 2.1%",
        "Revenue grew by 25% this year thanks to strong demand.",
    ]})
    result = remove_tabular_fragments(df)
    assert len(result) == 1
    assert "Revenue" in result["sentence"].iloc[0]
