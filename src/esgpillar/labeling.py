"""Etichettatura debole basata sui topic ESRS (E1-E5, S1-S4, G1).

Le frasi vengono assegnate a uno dei 10 topic ufficiali degli European
Sustainability Reporting Standards (Regolamento Delegato UE 2023/2772)
tramite un dizionario per topic costruito sulle definizioni ufficiali
dello standard. I topic sono poi aggregati nei tre pilastri E/S/G.

Nota metodologica (da riportare in tesi): questa e' supervisione debole
(weak supervision) basata su corrispondenza lessicale, non un gold
standard annotato da umani.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from .config import Config

ESRS_LEX: dict[str, list[str]] = {
    "E1": ["climate", "carbon", "emission", "emissions", "greenhouse", "ghg", "scope 1", "scope 2",
           "scope 3", "global warming", "decarbonization", "net-zero", "net zero", "paris agreement",
           "energy consumption", "renewable energy", "fossil fuel", "climate change", "climate risk"],
    "E2": ["pollution", "pollutant", "air quality", "soil contamination", "hazardous substance",
           "microplastic", "substances of concern", "contaminant", "emission to water", "emission to soil"],
    "E3": ["water", "marine", "ocean", "freshwater", "water withdrawal", "water consumption",
           "water discharge", "water stress", "wastewater", "water resource"],
    "E4": ["biodiversity", "ecosystem", "species", "habitat", "deforestation", "land use",
           "nature", "conservation", "protected area", "ecosystem service"],
    "E5": ["circular economy", "resource use", "waste", "recycling", "recycled", "raw material",
           "resource inflow", "resource outflow", "reuse", "packaging", "resource efficiency"],
    "S1": ["own workforce", "employee", "employees", "workforce", "working conditions",
           "health and safety", "diversity", "training", "collective bargaining", "equal pay",
           "non-discrimination", "work-life balance"],
    "S2": ["value chain worker", "supplier worker", "supply chain labor", "subcontractor",
           "migrant worker", "forced labor", "child labor", "working conditions in the supply chain"],
    "S3": ["affected communit", "local communit", "indigenous", "land rights", "resettlement",
           "community engagement", "human rights defender", "community impact"],
    "S4": ["consumer", "end-user", "end user", "customer safety", "product safety", "data privacy",
           "responsible marketing", "accessibility", "customer health"],
    "G1": ["business conduct", "anti-corruption", "anti-bribery", "corporate culture", "whistleblow",
           "lobbying", "political contribution", "supplier relationship", "payment practice",
           "board", "audit", "governance", "compliance"],
}

ESRS_TO_PILLAR: dict[str, str] = {
    **{f"E{i}": "E" for i in range(1, 6)},
    **{f"S{i}": "S" for i in range(1, 5)},
    "G1": "G",
}

ESRS_NAMES: dict[str, str] = {
    "E1": "Climate change", "E2": "Pollution", "E3": "Water & marine resources",
    "E4": "Biodiversity & ecosystems", "E5": "Resource use & circular economy",
    "S1": "Own workforce", "S2": "Workers in the value chain", "S3": "Affected communities",
    "S4": "Consumers & end-users", "G1": "Business conduct",
}


def _compile_terms(terms: list[str]) -> re.Pattern:
    escaped = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    return re.compile(rf"\b(?:{escaped})\b", re.I)


_ESRS_PATS: dict[str, re.Pattern] = {k: _compile_terms(v) for k, v in ESRS_LEX.items()}

_ALL_ESRS_TERMS = sorted({t for v in ESRS_LEX.values() for t in v}, key=len, reverse=True)
_MASK_PAT = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in _ALL_ESRS_TERMS) + r")\b", re.I)


def mask_lexicon(text: str) -> str:
    """Rimuove dal testo le parole del dizionario ESRS che hanno GENERATO
    l'etichetta (ogni frase etichettata le contiene, per costruzione).

    Senza questo passo un modello addestrato sul testo puo' imparare a
    riconoscere la parola-chiave invece del concetto, gonfiando le
    metriche in modo artificiale. E' un limite noto da dichiarare, non
    un dettaglio da nascondere: vedi ``Config.apply_leakage_mask``.
    """
    return _MASK_PAT.sub(" ", str(text))


def label_sentence(sentence: str) -> tuple[str, float]:
    """Etichetta una singola frase: ritorna (topic, confidenza).

    ``topic`` e' uno dei 10 codici ESRS oppure ``"Other"`` se non c'e'
    nessun match o se c'e' parita' tra piu' topic. ``confidenza`` e' la
    quota di match del topic vincente sul totale (0.0 se ``"Other"``).
    """
    counts = {k: len(p.findall(sentence)) for k, p in _ESRS_PATS.items()}
    total = sum(counts.values())
    if total == 0:
        return "Other", 0.0
    best = max(counts, key=counts.get)
    if list(counts.values()).count(counts[best]) > 1:  # parita' tra piu' topic
        return "Other", 0.0
    return best, counts[best] / total


def label_dataframe(df: pd.DataFrame, config: Config | None = None,
                    text_col: str = "sentence") -> pd.DataFrame:
    """Applica :func:`label_sentence` a un intero DataFrame.

    Aggiunge le colonne ``esrs`` (topic a grana fine), ``esg_score``
    (confidenza) ed ``esg`` (pilastro aggregato E/S/G/Other), poi filtra
    via le righe ``Other`` e aggiunge ``label`` (0=E, 1=S, 2=G).

    Parameters
    ----------
    df : pd.DataFrame
        Deve contenere la colonna di testo indicata da ``text_col``.
    config : Config, opzionale
        Se fornita e ``config.esrs_conf_thresh`` non e' ``None``, le
        frasi sotto soglia vengono anch'esse declassate a ``"Other"``.
    """
    config = config or Config()
    sentences = df[text_col].astype(str).tolist()
    labs, scores = zip(*[label_sentence(s) for s in sentences]) if sentences else ([], [])

    out = df.copy()
    out["esrs"] = labs
    out["esg_score"] = scores

    if config.esrs_conf_thresh is not None:
        out["esrs"] = np.where(out["esg_score"] >= config.esrs_conf_thresh, out["esrs"], "Other")

    out["esg"] = out["esrs"].map(ESRS_TO_PILLAR).fillna("Other")
    out = out[out["esg"].isin(["E", "S", "G"])].reset_index(drop=True)
    out["label"] = out["esg"].map({"E": 0, "S": 1, "G": 2})

    if config.apply_leakage_mask:
        out["sentence_model"] = out[text_col].map(mask_lexicon)

    return out


def label_distribution(df: pd.DataFrame) -> pd.Series:
    """Distribuzione delle frasi per topic ESRS a grana fine (comodo per l'EDA)."""
    return df["esrs"].value_counts()
