import pytest

from esgpillar.tokenization import Tokenizer


def test_lemma_normalization_default():
    tok = Tokenizer.from_companies(["Amazon"])
    result = tok("Amazon reduced its emissions significantly.")
    assert "emission" in result  # lemmatizzato, non "emissions"
    assert "reduce" in result


def test_porter_stemming():
    tok = Tokenizer.from_companies(["Amazon"], normalization="porter")
    result = tok("Amazon reduced its emissions significantly.")
    assert "emiss" in result
    assert "reduc" in result


def test_snowball_stemming():
    tok = Tokenizer.from_companies(["Amazon"], normalization="snowball")
    result = tok("Amazon reduced its emissions significantly.")
    assert "emiss" in result
    assert "reduc" in result


def test_none_normalization_keeps_original_forms():
    tok = Tokenizer.from_companies(["Amazon"], normalization="none")
    result = tok("Amazon reduced its emissions significantly.")
    assert "emissions" in result  # forma originale, non normalizzata
    assert "reduced" in result


def test_unknown_normalization_raises():
    with pytest.raises(ValueError, match="non riconosciuto"):
        Tokenizer.from_companies(["Amazon"], normalization="invalid")


def test_different_normalizations_give_different_results():
    text = "Companies are reducing emissions significantly."
    results = {
        norm: Tokenizer.from_companies([], normalization=norm)(text)
        for norm in ["lemma", "porter", "snowball", "none"]
    }
    # almeno lemma e none devono differire (lemma normalizza, none no)
    assert results["lemma"] != results["none"]
