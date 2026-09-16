"""Classificatori classici di Machine Learning (non reti neurali), come
alternativa/termine di paragone alle architetture PyTorch di :mod:`esgpillar.models`.

Tutti lavorano su rappresentazioni POOLED (un vettore per frase, 2D) --
mai su sequenze: un Naive Bayes o una SVM non hanno un concetto nativo
di "sequenza di token", solo la DNN tra le reti neurali condivide questo
vincolo. Ogni modello e' uno stimatore scikit-learn standard
(``.fit(X, y)`` / ``.predict_proba(X)``), cosi' l'integrazione con
:class:`esgpillar.pipeline.ESGPillarClassifier` resta uniforme.

Organizzati per famiglia, come tipicamente presentati in letteratura sulla
classificazione testuale classica:

- **Lineari**: Logistic Regression, Linear SVM, SVM a kernel (RBF), SGD,
  Ridge, Perceptron.
- **Naive Bayes**: Gaussiano (funziona con feature negative, es. LSA/SVD),
  Multinomiale, Complement (pensato apposta per classi sbilanciate),
  Bernoulli (feature binarie) -- le ultime tre richiedono feature NON
  NEGATIVE (tipicamente conteggi/TF-IDF, non embedding densi che possono
  avere valori negativi).
- **Alberi ed ensemble**: Decision Tree, Random Forest, Extra Trees,
  Gradient Boosting, AdaBoost, Bagging, XGBoost e LightGBM (extra
  opzionali, molto citati in letteratura ma non parte di scikit-learn).
- **Altri**: K-Nearest Neighbors, Linear/Quadratic Discriminant Analysis,
  MLP (rete neurale shallow di scikit-learn, diversa dalle reti PyTorch
  di :mod:`esgpillar.models`).
- **Meta-ensemble**: Voting (media dei voti di piu' modelli) e Stacking
  (un modello finale impara a combinare le predizioni degli altri) --
  NON supportano override (vedi :func:`build_classical_model`).

Richiede l'extra ``classical`` per XGBoost/LightGBM
(``pip install esg-pillar-classifier[classical]``); gli altri modelli
sono gia' coperti dalla dipendenza core su scikit-learn.

Ogni ``_build_X(cfg, **overrides)`` accetta iperparametri PERSONALIZZATI
al posto dei valori di default -- es. ``_build_random_forest(cfg,
n_estimators=500, max_depth=10)``. Le chiavi devono corrispondere ai
parametri del rispettivo stimatore scikit-learn/XGBoost/LightGBM.
"""
from __future__ import annotations

from .config import Config

# Feature che richiedono valori NON NEGATIVI (conteggi/TF-IDF): usarle con
# un embedding denso che puo' avere valori negativi (GloVe, Word2Vec, BERT...)
# solleva un errore esplicito di scikit-learn, non un fallimento silenzioso.
NON_NEGATIVE_ONLY = {"MultinomialNB", "ComplementNB", "BernoulliNB"}

# Meta-ensemble: gli override non sono supportati (ambiguo a quale dei
# sotto-modelli si applicherebbero) -- vedi build_classical_model().
_NO_OVERRIDES = {"Voting", "Stacking"}


def _build_logreg(cfg: Config, **overrides):
    from sklearn.linear_model import LogisticRegression
    params = {"max_iter": 1000, "class_weight": "balanced", "random_state": cfg.seed, **overrides}
    return LogisticRegression(**params)


def _build_linear_svm(cfg: Config, **overrides):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC
    # LinearSVC non ha predict_proba nativo: CalibratedClassifierCV lo aggiunge
    # (calibrazione di Platt), necessario per restare coerenti con l'interfaccia
    # predict_proba usata in tutta la libreria. Gli override si applicano al
    # LinearSVC interno, non al wrapper CalibratedClassifierCV.
    params = {"class_weight": "balanced", "random_state": cfg.seed, "max_iter": 5000, **overrides}
    return CalibratedClassifierCV(LinearSVC(**params))


def _build_svm_rbf(cfg: Config, **overrides):
    from sklearn.svm import SVC
    params = {"kernel": "rbf", "probability": True, "class_weight": "balanced",
             "random_state": cfg.seed, **overrides}
    return SVC(**params)


def _build_sgd(cfg: Config, **overrides):
    from sklearn.linear_model import SGDClassifier
    params = {"loss": "log_loss", "class_weight": "balanced", "random_state": cfg.seed, **overrides}
    return SGDClassifier(**params)


def _build_ridge(cfg: Config, **overrides):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import RidgeClassifier
    # stesso motivo del LinearSVC: RidgeClassifier non ha predict_proba nativo;
    # gli override si applicano al RidgeClassifier interno
    params = {"class_weight": "balanced", "random_state": cfg.seed, **overrides}
    return CalibratedClassifierCV(RidgeClassifier(**params))


def _build_perceptron(cfg: Config, **overrides):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import Perceptron
    params = {"class_weight": "balanced", "random_state": cfg.seed, **overrides}
    return CalibratedClassifierCV(Perceptron(**params))


def _build_gaussian_nb(cfg: Config, **overrides):
    from sklearn.naive_bayes import GaussianNB
    return GaussianNB(**overrides)


def _build_multinomial_nb(cfg: Config, **overrides):
    from sklearn.naive_bayes import MultinomialNB
    return MultinomialNB(**overrides)


def _build_complement_nb(cfg: Config, **overrides):
    from sklearn.naive_bayes import ComplementNB
    return ComplementNB(**overrides)


def _build_bernoulli_nb(cfg: Config, **overrides):
    from sklearn.naive_bayes import BernoulliNB
    return BernoulliNB(**overrides)


def _build_decision_tree(cfg: Config, **overrides):
    from sklearn.tree import DecisionTreeClassifier
    params = {"class_weight": "balanced", "random_state": cfg.seed, **overrides}
    return DecisionTreeClassifier(**params)


def _build_random_forest(cfg: Config, **overrides):
    from sklearn.ensemble import RandomForestClassifier
    params = {"n_estimators": 200, "class_weight": "balanced", "random_state": cfg.seed, **overrides}
    return RandomForestClassifier(**params)


def _build_extra_trees(cfg: Config, **overrides):
    from sklearn.ensemble import ExtraTreesClassifier
    params = {"n_estimators": 200, "class_weight": "balanced", "random_state": cfg.seed, **overrides}
    return ExtraTreesClassifier(**params)


def _build_gradient_boosting(cfg: Config, **overrides):
    from sklearn.ensemble import GradientBoostingClassifier
    params = {"random_state": cfg.seed, **overrides}
    return GradientBoostingClassifier(**params)


def _build_adaboost(cfg: Config, **overrides):
    from sklearn.ensemble import AdaBoostClassifier
    params = {"random_state": cfg.seed, **overrides}
    return AdaBoostClassifier(**params)


def _build_bagging(cfg: Config, **overrides):
    from sklearn.ensemble import BaggingClassifier
    params = {"random_state": cfg.seed, **overrides}
    return BaggingClassifier(**params)


def _build_xgboost(cfg: Config, **overrides):
    from xgboost import XGBClassifier
    params = {"eval_metric": "mlogloss", "random_state": cfg.seed, **overrides}
    return XGBClassifier(**params)


def _build_lightgbm(cfg: Config, **overrides):
    from lightgbm import LGBMClassifier
    params = {"class_weight": "balanced", "random_state": cfg.seed, "verbosity": -1, **overrides}
    return LGBMClassifier(**params)


def _build_knn(cfg: Config, **overrides):
    from sklearn.neighbors import KNeighborsClassifier
    params = {"n_neighbors": 5, **overrides}
    return KNeighborsClassifier(**params)


def _build_lda(cfg: Config, **overrides):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return LinearDiscriminantAnalysis(**overrides)


def _build_qda(cfg: Config, **overrides):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    return QuadraticDiscriminantAnalysis(**overrides)


def _build_mlp_sklearn(cfg: Config, **overrides):
    from sklearn.neural_network import MLPClassifier
    # rete neurale "shallow" di scikit-learn -- diversa dalla DNN PyTorch di
    # esgpillar.models: nessun early stopping/gradient clipping personalizzato,
    # utile come confronto rapido senza la pipeline di training PyTorch.
    params = {"hidden_layer_sizes": (128, 64), "max_iter": 500, "random_state": cfg.seed, **overrides}
    return MLPClassifier(**params)


def _build_voting(cfg: Config):
    from sklearn.ensemble import VotingClassifier
    return VotingClassifier(
        estimators=[
            ("logreg", _build_logreg(cfg)),
            ("rf", _build_random_forest(cfg)),
            ("nb", _build_gaussian_nb(cfg)),
        ],
        voting="soft",
    )


def _build_stacking(cfg: Config):
    from sklearn.ensemble import StackingClassifier
    return StackingClassifier(
        estimators=[
            ("logreg", _build_logreg(cfg)),
            ("rf", _build_random_forest(cfg)),
            ("svm", _build_linear_svm(cfg)),
        ],
        final_estimator=_build_logreg(cfg),
    )


CLASSICAL_BUILDERS = {
    # --- lineari ---
    "LogisticRegression": _build_logreg,
    "LinearSVM": _build_linear_svm,
    "SVM_RBF": _build_svm_rbf,
    "SGD": _build_sgd,
    "Ridge": _build_ridge,
    "Perceptron": _build_perceptron,
    # --- naive bayes ---
    "GaussianNB": _build_gaussian_nb,
    "MultinomialNB": _build_multinomial_nb,
    "ComplementNB": _build_complement_nb,
    "BernoulliNB": _build_bernoulli_nb,
    # --- alberi ed ensemble ---
    "DecisionTree": _build_decision_tree,
    "RandomForest": _build_random_forest,
    "ExtraTrees": _build_extra_trees,
    "GradientBoosting": _build_gradient_boosting,
    "AdaBoost": _build_adaboost,
    "Bagging": _build_bagging,
    "XGBoost": _build_xgboost,
    "LightGBM": _build_lightgbm,
    # --- altri ---
    "KNN": _build_knn,
    "LDA": _build_lda,
    "QDA": _build_qda,
    "MLP_sklearn": _build_mlp_sklearn,
    # --- meta-ensemble ---
    "Voting": _build_voting,
    "Stacking": _build_stacking,
}


def build_classical_model(name: str, config: Config | None = None, **overrides):
    """Costruisce un classificatore classico per nome (vedi ``CLASSICAL_BUILDERS``),
    con iperparametri PERSONALIZZATI al posto dei valori di default.

    ``overrides``: kwargs passati al costruttore dello stimatore
    scikit-learn/XGBoost/LightGBM sottostante (es. ``n_estimators=500``
    per ``"RandomForest"``). Per ``"LinearSVM"``/``"Ridge"``/
    ``"Perceptron"`` (avvolti in ``CalibratedClassifierCV`` per avere
    ``predict_proba``) gli override si applicano allo stimatore INTERNO,
    non al wrapper.

    ``"Voting"`` e ``"Stacking"`` NON supportano override (combinano piu'
    sotto-modelli: non e' chiaro a QUALE si applicherebbe un dato
    iperparametro) -- passare ``overrides`` a questi due solleva
    ``ValueError`` esplicito, invece di ignorarli silenziosamente.

    Esempio
    -------
    >>> model = build_classical_model("RandomForest", n_estimators=500, max_depth=10)
    >>> model.fit(X_train, y_train)
    >>> model.predict_proba(X_test)
    """
    config = config or Config()
    if name not in CLASSICAL_BUILDERS:
        raise ValueError(
            f"Modello classico non riconosciuto: {name!r}. "
            f"Disponibili: {sorted(CLASSICAL_BUILDERS)}"
        )
    if name in _NO_OVERRIDES:
        if overrides:
            raise ValueError(
                f"{name!r} e' un meta-ensemble (combina piu' sotto-modelli): non supporta "
                f"iperparametri personalizzati, dato che non sarebbe chiaro a quale dei "
                f"sotto-modelli si applicherebbero. Ricevuto: {list(overrides)}."
            )
        return CLASSICAL_BUILDERS[name](config)
    return CLASSICAL_BUILDERS[name](config, **overrides)
