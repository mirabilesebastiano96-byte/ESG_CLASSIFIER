"""Split train/validation/test PER AZIENDA, con percentuali a scelta.

Nessuna azienda compare in piu' di uno dei tre insiemi -- stesso principio
di anti-leakage gia' usato internamente da
:meth:`esgpillar.pipeline.ESGPillarClassifier.fit` (che pero' fa solo uno
split a due vie, train/validation interno; questo modulo aggiunge un vero
TERZO insieme, il test, tenuto fuori da qualunque `fit()` per una
valutazione onesta finale -- lo stesso schema usato nei notebook di
riferimento, qui reso una funzione riusabile invece di codice ripetuto a
mano ogni volta).
"""
from __future__ import annotations

import pandas as pd


def split_train_val_test(df: pd.DataFrame, company_col: str = "company", label_col: str = "label",
                         train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15,
                         seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide ``df`` in tre DataFrame (train, validation, test), garantendo
    che nessuna azienda compaia in piu' di un insieme (split per gruppo,
    via ``GroupShuffleSplit``, non uno split casuale riga per riga -- che
    lascerebbe filtrare lo stile linguistico specifico di un'azienda tra
    training e valutazione).

    ``train_size``/``val_size``/``test_size``: frazioni che devono sommare
    a 1.0 (tolleranza 1e-6). Lo split avviene in due passaggi -- prima
    ``test_size`` viene separato dal resto, poi il resto e' diviso tra
    train e validation nella proporzione relativa data da ``val_size`` --
    cosi' ``test_size`` rappresenta sempre la frazione ESATTA dell'intero
    dataset, non della sola parte rimanente dopo aver tolto il test.

    Esempio
    -------
    >>> train_df, val_df, test_df = split_train_val_test(df, train_size=0.6, val_size=0.2, test_size=0.2)

    Ritorna ``(train_df, val_df, test_df)``, ciascuno con indice
    resettato. Solleva ``ValueError`` se le tre frazioni non sommano a 1,
    o se ``df`` ha meno aziende distinte di quante servirebbero per
    popolare tutti e tre gli insiemi (con pochissime aziende, uno split a
    tre vie puo' non essere possibile)."""
    from sklearn.model_selection import GroupShuffleSplit

    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train_size + val_size + test_size deve sommare a 1.0, "
            f"ottenuto {total:.6f} ({train_size} + {val_size} + {test_size})."
        )

    n_companies = df[company_col].nunique()
    if n_companies < 3:
        raise ValueError(
            f"Servono almeno 3 aziende distinte per uno split a tre vie (train/val/test), "
            f"trovate {n_companies}. Con cosi' poche aziende, considera uno split a due vie "
            "(solo train/test) invece di tre."
        )

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(df, df[label_col], groups=df[company_col]))
    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # val_size relativo al SOLO train+val rimasto (non all'intero df):
    # cosi' la frazione di test_size passata dall'utente resta quella
    # esatta sull'intero dataset, indipendentemente da come si divide il resto
    rel_val_size = val_size / (train_size + val_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=seed)
    train_idx, val_idx = next(gss_val.split(trainval_df, trainval_df[label_col], groups=trainval_df[company_col]))
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df
