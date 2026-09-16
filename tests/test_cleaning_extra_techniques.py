import pytest

from esgpillar.cleaning import (
    correct_spelling,
    expand_contractions,
    expand_corporate_abbreviations,
    normalize_numbers,
)


# --------------------------------------------------------------------- #
# Espansione contrazioni
# --------------------------------------------------------------------- #
def test_expand_contractions():
    pytest.importorskip("contractions")
    result = expand_contractions("We don't reduce emissions, we can't afford it.")
    assert "do not" in result
    assert "cannot" in result
    assert "don't" not in result


def test_expand_contractions_no_contractions_unchanged():
    pytest.importorskip("contractions")
    text = "We reduced emissions significantly."
    assert expand_contractions(text) == text


# --------------------------------------------------------------------- #
# Espansione abbreviazioni societarie
# --------------------------------------------------------------------- #
def test_expand_corporate_abbreviations_default():
    result = expand_corporate_abbreviations("Amazon Corp. partnered with Beta Inc.")
    assert "Corporation" in result
    assert "Incorporated" in result
    assert "Corp." not in result


def test_expand_corporate_abbreviations_with_extra():
    result = expand_corporate_abbreviations(
        "Acme Grp. reported strong results.", extra={r"\bGrp\.": "Group"}
    )
    assert "Group" in result
    assert "Grp." not in result


def test_expand_corporate_abbreviations_case_insensitive():
    result = expand_corporate_abbreviations("acme corp. grew steadily.")
    assert "corporation" in result.lower()


# --------------------------------------------------------------------- #
# Normalizzazione numeri
# --------------------------------------------------------------------- #
def test_normalize_numbers_default_placeholder():
    result = normalize_numbers("We invested $10.5 million and cut emissions by 25%.")
    assert "<NUM>" in result
    assert "10.5" not in result
    assert "25" not in result


def test_normalize_numbers_custom_placeholder():
    result = normalize_numbers("Reduced by 25%.", placeholder="[NUMERO]")
    assert "[NUMERO]" in result
    assert "<NUM>" not in result


def test_normalize_numbers_no_numbers_unchanged():
    text = "No numbers appear in this sentence at all."
    assert normalize_numbers(text) == text


# --------------------------------------------------------------------- #
# Correzione ortografica
# --------------------------------------------------------------------- #
def test_correct_spelling_fixes_typo():
    pytest.importorskip("spellchecker")
    result = correct_spelling("We reduced our carbn emisions significantly.")
    assert "carbon" in result
    assert "emissions" in result


def test_correct_spelling_respects_max_corrections():
    pytest.importorskip("spellchecker")
    result = correct_spelling("We reduced our carbn emisions significantly.", max_corrections=1)
    # solo la prima parola sconosciuta deve essere corretta, l'altra resta invariata
    words = result.split()
    misspelled_present = "emisions" in words or "carbn" in words
    assert misspelled_present


def test_correct_spelling_no_errors_unchanged():
    pytest.importorskip("spellchecker")
    text = "We reduced our carbon emissions significantly"
    assert correct_spelling(text) == text
