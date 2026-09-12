from esgpillar.tokenization import Tokenizer


def test_tokenizer_lemmatizes_and_lowercases():
    tok = Tokenizer.from_companies(["Amazon"])
    out = tok("Amazon is reducing emissions and improving employee safety.")
    assert "emission" in out          # emissions -> emission
    assert "reduce" in out            # reducing -> reduce
    assert "employee" in out


def test_tokenizer_excludes_company_names():
    tok = Tokenizer.from_companies(["Amazon", "Enel"])
    out = tok("Amazon and Enel both improved their governance.")
    assert "amazon" not in out
    assert "enel" not in out


def test_tokenizer_excludes_stopwords_and_generic_terms():
    tok = Tokenizer.from_companies([])
    out = tok("The company published its annual sustainability report this year.")
    assert "the" not in out
    assert "company" not in out
    assert "sustainability" not in out
    assert "report" not in out


def test_tokenizer_empty_on_no_content_words():
    tok = Tokenizer.from_companies([])
    assert tok("the and of") == []
