import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")

from esgpillar.plotting import plot_embeddings_2d


@pytest.mark.parametrize("method", ["pca", "tsne"])
def test_plot_embeddings_2d(method):
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(40, 20)).astype("float32")
    labels = rng.integers(0, 3, 40)

    fig = plot_embeddings_2d(vectors, labels, method=method)
    assert fig is not None
    ax = fig.axes[0]
    assert method.upper() in ax.get_xlabel() or "t-SNE" in ax.get_xlabel()


def test_plot_embeddings_2d_unknown_method_raises():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(10, 5)).astype("float32")
    labels = rng.integers(0, 3, 10)
    with pytest.raises(ValueError, match="non riconosciuto"):
        plot_embeddings_2d(vectors, labels, method="invalid")


def test_plot_embeddings_2d_handles_missing_class():
    """Se una classe non compare affatto tra le etichette, non deve crashare."""
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(10, 5)).astype("float32")
    labels = np.zeros(10, dtype=int)  # solo classe 0, mai 1 o 2
    fig = plot_embeddings_2d(vectors, labels, method="pca")
    assert fig is not None


def test_esgpillarclassifier_plot_embeddings_convenience_method():
    """clf.plot_embeddings() deve richiamare l'encoder gia' addestrato e produrre una figura."""
    from unittest.mock import MagicMock

    from esgpillar import Config, ESGPillarClassifier

    clf = ESGPillarClassifier(config=Config(), embedding="word2vec", network="DNN")
    clf._is_fitted = True
    rng = np.random.default_rng(0)
    fake_encoder = MagicMock(
        side_effect=lambda sentences: (None, rng.normal(size=(len(sentences), 10)).astype("float32"), None)
    )
    clf._encoder = fake_encoder

    fig = clf.plot_embeddings(["frase uno", "frase due", "frase tre"], [0, 1, 2], method="pca")
    assert fig is not None
    fake_encoder.assert_called_once()
