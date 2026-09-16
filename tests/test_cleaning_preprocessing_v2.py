import pandas as pd
import pytest

from esgpillar.cleaning import (
    detect_boilerplate,
    normalize_currency,
    remove_accents,
    remove_boilerplate_column,
    segment_sentences,
    split_into_sentences,
)


# --------------------------------------------------------------------- #
# Segmentazione in frasi
# --------------------------------------------------------------------- #
def test_segment_sentences_splits_correctly():
    text = "We reduced emissions significantly. The board approved a policy. Training expanded."
    result = segment_sentences(text)
    assert len(result) == 3


def test_segment_sentences_handles_abbreviations():
    """Un punto dentro un'abbreviazione comune (Dr., Corp.) non deve
    spezzare la frase li'."""
    text = "Dr. Smith reviewed the report. The board approved it."
    result = segment_sentences(text)
    assert len(result) == 2


def test_split_into_sentences_explodes_dataframe():
    df = pd.DataFrame({
        "company": ["Acme", "Beta"],
        "sentence": [
            "We cut emissions. The board approved a policy.",
            "Water usage decreased.",
        ],
    })
    result = split_into_sentences(df)
    assert len(result) == 3
    assert result["company"].tolist() == ["Acme", "Acme", "Beta"]


# --------------------------------------------------------------------- #
# Rimozione accenti
# --------------------------------------------------------------------- #
def test_remove_accents():
    pytest.importorskip("unidecode")
    result = remove_accents("Café Müller società italiana")
    assert result == "Cafe Muller societa italiana"


def test_remove_accents_no_accents_unchanged():
    pytest.importorskip("unidecode")
    text = "We reduced emissions significantly"
    assert remove_accents(text) == text


# --------------------------------------------------------------------- #
# Normalizzazione valute
# --------------------------------------------------------------------- #
def test_normalize_currency_symbols():
    result = normalize_currency("We invested $10 million and €5.2 million.")
    assert "USD 10" in result
    assert "EUR 5.2" in result
    assert "$" not in result
    assert "€" not in result


def test_normalize_currency_words():
    result = normalize_currency("We paid 3 million dollars and 2 million euros in fines.")
    assert "USD" in result
    assert "EUR" in result
    assert "dollars" not in result.lower()


def test_normalize_currency_extra():
    result = normalize_currency("We paid 10 francs.", extra={r"\bfrancs?\b": "CHF"})
    assert "CHF" in result


def test_normalize_currency_no_currency_unchanged():
    text = "We reduced emissions significantly this year."
    assert normalize_currency(text) == text


# --------------------------------------------------------------------- #
# Rilevamento boilerplate
# --------------------------------------------------------------------- #
def test_detect_boilerplate_finds_repeated_sentence():
    documents = {
        "Acme_2021": ["We reduced emissions this year.",
                     "Copyright 2021 Acme Corp. All rights reserved.",
                     "Board approved new policy."],
        "Acme_2022": ["Emissions decreased further.",
                     "Copyright 2021 Acme Corp. All rights reserved.",
                     "New safety training launched."],
        "Beta_2021": ["Water usage declined.",
                     "Copyright 2021 Acme Corp. All rights reserved.",
                     "Governance improved significantly."],
    }
    result = detect_boilerplate(documents, min_doc_fraction=0.5)
    assert result == ["Copyright 2021 Acme Corp. All rights reserved."]


def test_detect_boilerplate_ignores_unique_sentences():
    documents = {
        "Acme": ["Unique sentence one.", "Copyright shared line."],
        "Beta": ["Unique sentence two.", "Copyright shared line."],
    }
    result = detect_boilerplate(documents, min_doc_fraction=0.5)
    assert "Unique sentence one." not in result
    assert "Unique sentence two." not in result
    assert "Copyright shared line." in result


def test_detect_boilerplate_single_document_finds_nothing():
    """Con un solo documento, niente puo' essere 'ripetuto tra documenti'
    -- verifica esplicita del bug trovato e corretto: una soglia solo
    percentuale, con pochi documenti, diventava banalmente vera per
    qualunque frase."""
    documents = {"Acme": ["Repeated line.", "Repeated line.", "Other line."]}
    result = detect_boilerplate(documents, min_doc_fraction=0.5)
    assert result == []


def test_detect_boilerplate_empty_documents():
    assert detect_boilerplate({}) == []


def test_detect_boilerplate_respects_min_doc_count():
    """Con min_doc_count=3, una frase che compare solo in 2 documenti su 4
    (50%, sopra min_doc_fraction) non deve comunque essere segnalata."""
    documents = {
        "A": ["shared line"], "B": ["shared line"],
        "C": ["other line c"], "D": ["other line d"],
    }
    result = detect_boilerplate(documents, min_doc_fraction=0.3, min_doc_count=3)
    assert result == []


def test_remove_boilerplate_column():
    df = pd.DataFrame({
        "company": ["Acme", "Acme", "Acme", "Beta", "Beta"],
        "sentence": [
            "We reduced emissions this year.",
            "Copyright 2021 Acme Corp. All rights reserved.",
            "Board approved new policy.",
            "Water usage declined.",
            "Copyright 2021 Acme Corp. All rights reserved.",
        ],
    })
    result = remove_boilerplate_column(df, doc_col="company", min_doc_fraction=0.5)
    assert len(result) == 3
    assert "Copyright 2021 Acme Corp. All rights reserved." not in result["sentence"].tolist()
