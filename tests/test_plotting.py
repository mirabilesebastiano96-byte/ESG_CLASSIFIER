import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib
matplotlib.use("Agg")  # niente display: i test girano anche senza schermo

from esgpillar.plotting import (  # noqa: E402
    plot_class_distribution,
    plot_comparison,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_history,
)


def test_plot_training_history_with_accuracy():
    history = {
        "train_loss": [0.9, 0.6, 0.4, 0.3],
        "val_loss": [0.85, 0.65, 0.6, 0.62],
        "val_acc": [0.4, 0.6, 0.65, 0.63],
        "lr": [5e-4] * 4,
    }
    fig = plot_training_history(history, title="Test")
    assert fig is not None
    assert len(fig.axes) == 2  # asse loss + asse gemello per la metrica


def test_plot_training_history_without_metric_key():
    history = {"train_loss": [0.9, 0.5], "val_loss": [0.8, 0.7]}
    fig = plot_training_history(history)
    assert fig is not None
    assert len(fig.axes) == 1  # nessuna metrica -> nessun asse gemello


def test_plot_comparison():
    rng = np.random.default_rng(0)
    results = {
        (rep, net): {"test_acc": rng.uniform(0.4, 0.9)}
        for rep in ["GloVe", "ESGBERT"]
        for net in ["DNN", "BiLSTM"]
    }
    fig = plot_comparison(results, metric="test_acc", metric_label="Accuracy")
    assert fig is not None


def test_plot_comparison_empty_raises():
    with pytest.raises(ValueError):
        plot_comparison({}, metric="test_acc")


def test_plot_confusion_matrix():
    y_true = [0, 0, 1, 1, 2, 2, 0, 1]
    y_pred = [0, 1, 1, 1, 2, 0, 0, 1]
    fig = plot_confusion_matrix(y_true, y_pred)
    assert fig is not None


def test_plot_roc_curves():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 30)
    results = {
        "GloVe-DNN": {"proba": rng.dirichlet([1, 1, 1], size=30)},
        "ESGBERT-BiLSTM": {"proba": rng.dirichlet([1, 1, 1], size=30)},
    }
    fig = plot_roc_curves(y_true, results)
    assert fig is not None


def test_plot_class_distribution():
    labels = [0, 0, 0, 1, 1, 2]
    fig = plot_class_distribution(labels)
    assert fig is not None


def _make_wordcloud_df():
    import pandas as pd
    rows = []
    for _ in range(8):
        rows.append({"sentence": "Acme reduced carbon emissions through climate water programs.", "esg": "E"})
        rows.append({"sentence": "Acme expanded employee training and safety diversity programs.", "esg": "S"})
        rows.append({"sentence": "Acme board improved audit governance compliance procedures.", "esg": "G"})
    return pd.DataFrame(rows)


def test_plot_wordcloud_by_pillar_rectangle_no_tokenizer():
    from esgpillar.plotting import plot_wordcloud_by_pillar
    df = _make_wordcloud_df()
    fig = plot_wordcloud_by_pillar(df, shape="rectangle")
    assert fig is not None
    assert len(fig.axes) == 3


def test_plot_wordcloud_by_pillar_excludes_company_names():
    """Stesso bug corretto nel notebook: senza un tokenizzatore che esclude
    i nomi azienda, "acme" rischia di comparire come parola 'distintiva'."""
    pytest.importorskip("nltk")
    import nltk
    for pkg in ["punkt", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
               "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

    from esgpillar.plotting import plot_wordcloud_by_pillar
    from esgpillar.tokenization import Tokenizer

    df = _make_wordcloud_df()
    tok = Tokenizer.from_companies(["Acme"])
    fig = plot_wordcloud_by_pillar(df, tokenizer=tok, shape="rectangle", min_count=1)
    assert fig is not None
    # "acme" non deve mai finire nel testo passato a WordCloud: lo verifichiamo
    # controllando direttamente l'output del tokenizzatore usato dalla funzione
    assert "acme" not in tok("Acme reduced carbon emissions.")
