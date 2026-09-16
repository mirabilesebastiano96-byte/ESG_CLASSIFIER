import pandas as pd

from esgpillar.cleaning import basic_clean, clean_dataframe
from esgpillar.config import Config


def test_basic_clean_fixes_hyphenation_and_urls():
    s = "Our sustain-\nability report is at www.example.com for more info."
    out = basic_clean(s)
    assert "sustainability" in out
    assert "www.example.com" not in out


def test_clean_dataframe_removes_short_and_duplicate_sentences():
    df = pd.DataFrame({"sentence": [
        "This is a perfectly normal sentence with enough words in it.",
        "This is a perfectly normal sentence with enough words in it.",  # duplicato
        "Too short.",
        "1 2 3 4 5 6 7 8 9",  # solo numeri
    ]})
    out = clean_dataframe(df, Config(min_words=5), verbose=False)
    assert len(out) == 1
    assert "sentence_raw" in out.columns
