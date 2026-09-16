import pandas as pd
import pytest

from esgpillar import split_train_val_test


def _make_df(n_companies=20, rows_per_company=5):
    rows = []
    for i in range(n_companies):
        for j in range(rows_per_company):
            rows.append({"company": f"C{i}", "sentence": f"sentence {i}-{j}", "label": (i + j) % 3})
    return pd.DataFrame(rows)


def test_split_default_proportions_approximately_correct():
    df = _make_df(n_companies=20, rows_per_company=5)
    train_df, val_df, test_df = split_train_val_test(df, seed=0)
    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(df)
    # con lo split per GRUPPO le proporzioni esatte possono variare leggermente
    # (dipende da come si dividono le aziende, non le righe singole)
    assert 0.5 < len(train_df) / total < 0.9
    assert 0.05 < len(val_df) / total < 0.3
    assert 0.05 < len(test_df) / total < 0.3


def test_split_no_company_overlap_between_sets():
    df = _make_df(n_companies=20, rows_per_company=5)
    train_df, val_df, test_df = split_train_val_test(df, seed=0)
    train_c, val_c, test_c = set(train_df["company"]), set(val_df["company"]), set(test_df["company"])
    assert train_c & val_c == set()
    assert train_c & test_c == set()
    assert val_c & test_c == set()


def test_split_custom_proportions():
    df = _make_df(n_companies=20, rows_per_company=5)
    train_df, val_df, test_df = split_train_val_test(df, train_size=0.6, val_size=0.2, test_size=0.2, seed=0)
    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(df)


def test_split_index_reset():
    df = _make_df(n_companies=20, rows_per_company=5)
    train_df, val_df, test_df = split_train_val_test(df, seed=0)
    for part in (train_df, val_df, test_df):
        assert list(part.index) == list(range(len(part)))


def test_split_reproducible_with_same_seed():
    df = _make_df(n_companies=20, rows_per_company=5)
    train1, val1, test1 = split_train_val_test(df, seed=42)
    train2, val2, test2 = split_train_val_test(df, seed=42)
    assert set(train1["company"]) == set(train2["company"])
    assert set(val1["company"]) == set(val2["company"])
    assert set(test1["company"]) == set(test2["company"])


def test_split_different_seeds_can_give_different_splits():
    df = _make_df(n_companies=20, rows_per_company=5)
    train1, _, _ = split_train_val_test(df, seed=0)
    train2, _, _ = split_train_val_test(df, seed=123)
    # non garantito in assoluto, ma con 20 aziende e' estremamente probabile
    # che semi diversi diano gruppi almeno in parte diversi
    assert set(train1["company"]) != set(train2["company"])


def test_split_proportions_not_summing_to_one_raises():
    df = _make_df()
    with pytest.raises(ValueError, match="sommare a 1.0"):
        split_train_val_test(df, train_size=0.5, val_size=0.2, test_size=0.2)


def test_split_too_few_companies_raises():
    df = pd.DataFrame({"company": ["A", "A", "B", "B"], "sentence": ["s1", "s2", "s3", "s4"], "label": [0, 1, 0, 1]})
    with pytest.raises(ValueError, match="almeno 3 aziende"):
        split_train_val_test(df)


def test_split_exactly_three_companies_works():
    df = pd.DataFrame({
        "company": ["A", "A", "B", "B", "C", "C"],
        "sentence": [f"s{i}" for i in range(6)],
        "label": [0, 1, 0, 1, 0, 1],
    })
    train_df, val_df, test_df = split_train_val_test(df, seed=0)
    total_companies = train_df["company"].nunique() + val_df["company"].nunique() + test_df["company"].nunique()
    assert total_companies == 3


def test_split_custom_column_names():
    df = pd.DataFrame({
        "org": [f"C{i}" for i in range(10) for _ in range(3)],
        "text": [f"s{i}" for i in range(30)],
        "y": [i % 3 for i in range(30)],
    })
    train_df, val_df, test_df = split_train_val_test(df, company_col="org", label_col="y", seed=0)
    assert set(train_df["org"]) & set(val_df["org"]) == set()
