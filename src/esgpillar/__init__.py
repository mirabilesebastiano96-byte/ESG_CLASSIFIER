"""esgpillar — estrazione, etichettatura debole (ESRS) e classificazione
di frasi ESG nei pilastri Environmental / Social / Governance.

Esempio rapido
--------------
>>> from esgpillar import Config, ESGPillarClassifier
>>> from esgpillar.labeling import label_dataframe
>>> import pandas as pd
>>>
>>> df = pd.read_csv("dataset.csv")            # colonne: company, sentence
>>> df = label_dataframe(df)                    # aggiunge 'esg', 'label'
>>>
>>> clf = ESGPillarClassifier(embedding="glove", network="BiLSTM")
>>> clf.fit(df)
>>> clf.predict(["We reduced carbon emissions by 30% in 2023."])
['Environmental']

Le singole fasi (estrazione, pulizia, etichettatura, tokenizzazione,
embedding, training) sono disponibili anche separatamente nei rispettivi
sottomoduli, per chi vuole ricostruirsi il disegno sperimentale completo
(split per azienda, confronto fra più rappresentazioni) invece di usare
solo l'API di alto livello.
"""
from .config import Config
from .labeling import ESRS_LEX, ESRS_NAMES, ESRS_TO_PILLAR, label_dataframe, label_sentence
from .pipeline import ESGPillarClassifier
from .tokenization import Tokenizer

__version__ = "0.1.0"

__all__ = [
    "Config",
    "ESGPillarClassifier",
    "Tokenizer",
    "label_dataframe",
    "label_sentence",
    "ESRS_LEX",
    "ESRS_TO_PILLAR",
    "ESRS_NAMES",
]
