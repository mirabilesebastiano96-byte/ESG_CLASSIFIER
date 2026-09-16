from unittest.mock import MagicMock

import numpy as np
import pytest

from esgpillar import Config, ESGPillarClassifier


def _clf_with_fake_proba(fake_proba):
    clf = ESGPillarClassifier(config=Config(), embedding="word2vec", network="DNN")
    clf._is_fitted = True
    clf.predict_proba = MagicMock(return_value=fake_proba)
    return clf


def test_evaluate_returns_full_classification_report_structure():
    fake_proba = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.1, 0.8],
        [0.2, 0.7, 0.1],
    ])
    clf = _clf_with_fake_proba(fake_proba)
    report = clf.evaluate(["s1", "s2", "s3"], [0, 2, 1])

    assert set(report.keys()) == {
        "Environmental", "Social", "Governance", "accuracy", "macro avg", "weighted avg"
    }
    for class_name in ("Environmental", "Social", "Governance"):
        assert set(report[class_name].keys()) == {"precision", "recall", "f1-score", "support"}
    assert set(report["macro avg"].keys()) == {"precision", "recall", "f1-score", "support"}
    assert set(report["weighted avg"].keys()) == {"precision", "recall", "f1-score", "support"}


def test_evaluate_perfect_predictions_gives_accuracy_one():
    fake_proba = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.1, 0.8],
        [0.2, 0.7, 0.1],
    ])
    clf = _clf_with_fake_proba(fake_proba)
    report = clf.evaluate(["s1", "s2", "s3"], [0, 2, 1])
    assert report["accuracy"] == pytest.approx(1.0)
    for class_name in ("Environmental", "Social", "Governance"):
        assert report[class_name]["precision"] == pytest.approx(1.0)
        assert report[class_name]["recall"] == pytest.approx(1.0)
        assert report[class_name]["f1-score"] == pytest.approx(1.0)


def test_evaluate_reflects_specific_class_error():
    """Un errore su UNA classe specifica deve riflettersi nel recall
    di QUELLA classe, non genericamente in tutto il report."""
    fake_proba = np.array([
        [0.9, 0.05, 0.05],  # vero E, predetto E -> corretto
        [0.7, 0.2, 0.1],    # vero G, predetto E -> ERRORE
        [0.2, 0.7, 0.1],    # vero S, predetto S -> corretto
    ])
    clf = _clf_with_fake_proba(fake_proba)
    report = clf.evaluate(["s1", "s2", "s3"], [0, 2, 1])

    assert report["accuracy"] < 1.0
    assert report["Governance"]["recall"] == pytest.approx(0.0)  # l'unica frase G e' stata sbagliata
    assert report["Social"]["recall"] == pytest.approx(1.0)  # non toccata dall'errore


def test_evaluate_support_matches_true_label_counts():
    fake_proba = np.array([
        [0.9, 0.05, 0.05],
        [0.9, 0.05, 0.05],
        [0.1, 0.1, 0.8],
    ])
    clf = _clf_with_fake_proba(fake_proba)
    report = clf.evaluate(["s1", "s2", "s3"], [0, 0, 2])  # 2 Environmental, 0 Social, 1 Governance veri
    assert report["Environmental"]["support"] == 2
    assert report["Social"]["support"] == 0
    assert report["Governance"]["support"] == 1


def test_evaluate_handles_missing_class_in_labels():
    """Se una classe non compare affatto tra le etichette vere, il report
    non deve crashare -- deve solo avere support 0 per quella classe."""
    fake_proba = np.array([
        [0.9, 0.05, 0.05],
        [0.8, 0.1, 0.1],
    ])
    clf = _clf_with_fake_proba(fake_proba)
    report = clf.evaluate(["s1", "s2"], [0, 0])  # solo Environmental, mai Social/Governance
    assert report["Social"]["support"] == 0
    assert report["Governance"]["support"] == 0
