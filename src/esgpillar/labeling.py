"""Etichettatura debole basata sui topic ESRS (E1-E5, S1-S4, G1) -- il
metodo PREDEFINITO -- piu' quattro tecniche di etichettatura alternative
e una quinta per combinarle.

Il metodo predefinito assegna le frasi a uno dei 10 topic ufficiali degli
European Sustainability Reporting Standards (Regolamento Delegato UE
2023/2772) tramite un dizionario per topic costruito sulle definizioni
ufficiali dello standard. I topic sono poi aggregati nei tre pilastri E/S/G.

Nota metodologica (da riportare in tesi): questa e' supervisione debole
(weak supervision) basata su corrispondenza lessicale, non un gold
standard annotato da umani.

Le TECNICHE ALTERNATIVE (in fondo al modulo) lavorano a livello di
PILASTRO, non alla grana fine dei 10 topic:
``label_dataframe_tfidf`` (conteggio pesato per informativita' nel
corpus), ``label_dataframe_prototype`` (similarita' semantica a un
prototipo per pilastro, via un embedding a scelta), ``label_dataframe_zero_shot``
(un modello di NLI, nessun dizionario ne' training),
``label_dataframe_esgbert`` (la testa di classificazione originale di
ESGBERT, un vero modello fine-tunato, non supervisione debole),
``label_dataframe_ensemble`` (combina piu' etichettature con voto di
maggioranza, in stile Snorkel semplificato).
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


# ======================================================================= #
# TECNICHE DI ETICHETTATURA ALTERNATIVE
#
# Tutte lavorano a livello di PILASTRO (E/S/G/Other), non alla grana fine
# dei 10 topic ESRS -- quella granularita' resta una caratteristica
# specifica del metodo a dizionario sopra (:func:`label_sentence`).
#
# Nessuna sostituisce il metodo predefinito: sono alternative esplicite,
# da scegliere consapevolmente. Ciascuna ha assunzioni e limiti diversi,
# discussi nella rispettiva docstring.
# ======================================================================= #

# --------------------------------------------------------------------- #
# 1. Punteggio pesato TF-IDF (invece del conteggio grezzo)
# --------------------------------------------------------------------- #
def label_dataframe_tfidf(df: pd.DataFrame, config: Config | None = None,
                          text_col: str = "sentence") -> pd.DataFrame:
    """Come :func:`label_dataframe`, ma pesa ogni corrispondenza col
    dizionario ESRS per il suo punteggio TF-IDF ENTRO IL CORPUS fornito,
    invece di contare semplicemente quante volte compare.

    Perche' puo' essere meglio del conteggio grezzo: un termine del
    dizionario molto comune in TUTTO il corpus (es. compare in meta' delle
    frasi) e' meno INFORMATIVO di un termine raro che compare solo in
    poche frasi molto specifiche -- il conteggio grezzo tratta i due allo
    stesso modo, il TF-IDF no.

    Limite da conoscere: il punteggio TF-IDF dipende dal CORPUS fornito --
    la stessa frase puo' ricevere un'etichettatura leggermente diversa se
    la applichi a corpus diversi (il metodo a conteggio grezzo, invece,
    etichetta ogni frase in isolamento, sempre allo stesso modo).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    config = config or Config()
    sentences = df[text_col].astype(str).tolist()

    all_terms = sorted({t for v in ESRS_LEX.values() for t in v})
    # ngram_range=(1,3), non un token_pattern personalizzato: alcuni termini del
    # dizionario sono multi-parola (es. "renewable energy", "value chain worker"),
    # e solo l'analyzer con n-grammi genera correttamente le combinazioni di
    # parole adiacenti da confrontare col vocabolario fisso -- un pattern
    # personalizzato troppo permissivo rischia di inghiottire l'intera frase
    # come un solo token (bug reale, riscontrato e corretto in fase di test).
    vec = TfidfVectorizer(vocabulary=all_terms, lowercase=True, ngram_range=(1, 3))
    tfidf = vec.fit_transform(sentences)
    term_to_col = {t: i for i, t in enumerate(vec.get_feature_names_out())}

    labs, scores = [], []
    for i in range(len(sentences)):
        pillar_scores = {}
        for topic, terms in ESRS_LEX.items():
            s = sum(tfidf[i, term_to_col[t]] for t in terms if t in term_to_col)
            pillar_scores[topic] = s
        total = sum(pillar_scores.values())
        if total == 0:
            labs.append("Other")
            scores.append(0.0)
            continue
        best = max(pillar_scores, key=pillar_scores.get)
        labs.append(best)
        scores.append(pillar_scores[best] / total)

    out = df.copy()
    out["esrs"] = labs
    out["esg_score"] = scores
    out["esg"] = out["esrs"].map(ESRS_TO_PILLAR).fillna("Other")
    out = out[out["esg"].isin(["E", "S", "G"])].reset_index(drop=True)
    out["label"] = out["esg"].map({"E": 0, "S": 1, "G": 2})
    if config.apply_leakage_mask:
        out["sentence_model"] = out[text_col].map(mask_lexicon)
    return out


# --------------------------------------------------------------------- #
# 2. Similarita' a prototipi (embedding-based)
# --------------------------------------------------------------------- #
def label_dataframe_prototype(df: pd.DataFrame, encoder, config: Config | None = None,
                              text_col: str = "sentence", threshold: float = 0.0) -> pd.DataFrame:
    """Etichetta per similarita' a un "prototipo" per pilastro: un vettore
    ottenuto codificando i termini del dizionario ESRS di quel pilastro
    con ``encoder`` (un qualunque encoder di :mod:`esgpillar.embeddings`,
    es. SBERT), poi mediandoli. Ogni frase viene assegnata al pilastro il
    cui prototipo e' piu' vicino (similarita' coseno).

    A differenza del metodo a dizionario (corrispondenza ESATTA di
    parole), questo cattura similarita' SEMANTICA -- una frase puo' essere
    etichettata correttamente anche se non contiene NESSUNA parola esatta
    del dizionario, purche' l'encoder scelto catturi bene il significato.
    Il rovescio della medaglia: dipende interamente dalla qualita' delle
    rappresentazioni dell'encoder scelto, ed e' meno trasparente/
    ispezionabile di un dizionario esplicito (non puoi vedere "quale
    parola" ha causato l'etichetta).

    ``threshold``: se la similarita' migliore e' sotto questa soglia, la
    frase e' etichettata ``"Other"`` invece di forzare comunque un pilastro.
    """
    import numpy as np

    config = config or Config()
    pillar_terms = {
        "E": [t for topic in ["E1", "E2", "E3", "E4", "E5"] for t in ESRS_LEX[topic]],
        "S": [t for topic in ["S1", "S2", "S3", "S4"] for t in ESRS_LEX[topic]],
        "G": ESRS_LEX["G1"],
    }
    prototypes = {}
    for pillar, terms in pillar_terms.items():
        _, pooled, _ = encoder(terms)
        prototypes[pillar] = pooled.mean(axis=0)

    sentences = df[text_col].astype(str).tolist()
    _, pooled_sentences, _ = encoder(sentences)

    labs, scores = [], []
    proto_matrix = np.stack([prototypes[p] for p in ["E", "S", "G"]])
    proto_norms = np.linalg.norm(proto_matrix, axis=1)
    for vec in pooled_sentences:
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0:
            labs.append("Other")
            scores.append(0.0)
            continue
        sims = (proto_matrix @ vec) / (proto_norms * vec_norm + 1e-10)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim < threshold:
            labs.append("Other")
            scores.append(best_sim)
        else:
            labs.append(["E", "S", "G"][best_idx])
            scores.append(best_sim)

    out = df.copy()
    out["esg"] = labs
    out["esg_score"] = scores
    out = out[out["esg"].isin(["E", "S", "G"])].reset_index(drop=True)
    out["label"] = out["esg"].map({"E": 0, "S": 1, "G": 2})
    if config.apply_leakage_mask:
        out["sentence_model"] = out[text_col].map(mask_lexicon)
    return out


# --------------------------------------------------------------------- #
# 3. Zero-shot (riusa esgpillar.text_mining.zero_shot_classify)
# --------------------------------------------------------------------- #
def label_dataframe_zero_shot(df: pd.DataFrame, config: Config | None = None, text_col: str = "sentence",
                              model: str = "facebook/bart-large-mnli") -> pd.DataFrame:
    """Etichetta con classificazione zero-shot (vedi
    :func:`esgpillar.text_mining.zero_shot_classify`): nessun dizionario,
    nessun training, un modello di NLI valuta al volo quanto ciascuna
    frase "riguarda" Environmental/Social/Governance.

    Piu' lento del dizionario (un forward pass del modello per frase) e
    tipicamente meno accurato di un classificatore addestrato sul tuo
    dominio -- ma non richiede ne' un dizionario costruito a mano ne' dati
    etichettati, utile per un primo tentativo rapido o come confronto.
    """
    from .text_mining import zero_shot_classify

    config = config or Config()
    labels_map = {"Environmental": "E", "Social": "S", "Governance": "G"}
    candidate_labels = list(labels_map)

    sentences = df[text_col].astype(str).tolist()
    labs, scores = [], []
    for s in sentences:
        result = zero_shot_classify(s, candidate_labels, model=model)
        labs.append(labels_map[result["labels"][0]])
        scores.append(result["scores"][0])

    out = df.copy()
    out["esg"] = labs
    out["esg_score"] = scores
    out["label"] = out["esg"].map({"E": 0, "S": 1, "G": 2})
    if config.apply_leakage_mask:
        out["sentence_model"] = out[text_col].map(mask_lexicon)
    return out


# --------------------------------------------------------------------- #
# 4. ESGBERT come classificatore diretto (non solo come encoder)
# --------------------------------------------------------------------- #
def label_dataframe_esgbert(df: pd.DataFrame, config: Config | None = None,
                            text_col: str = "sentence", batch_size: int = 32) -> pd.DataFrame:
    """Etichetta usando la TESTA DI CLASSIFICAZIONE originale di ESGBERT
    (``yiyanghkust/finbert-esg``) -- un vero classificatore fine-tunato su
    2.000 frasi annotate a mano da report ESG reali, che produce
    direttamente Environmental/Social/Governance/None. A differenza di
    :func:`esgpillar.embeddings.load_esgbert` (che carica lo stesso
    modello come sorgente di EMBEDDING, senza testa di classificazione),
    qui si usa il modello COSI' COM'E', gia' addestrato per questo compito
    esatto -- non serve costruire un dizionario ne' addestrare nulla.

    Probabilmente il metodo piu' accurato tra tutte le alternative qui,
    proprio perche' non e' debolmente supervisionato: e' un classificatore
    vero, addestrato su dati annotati da umani (anche se non i TUOI dati --
    resta quindi da verificare quanto generalizzi al tuo dominio/settore
    specifico). Richiede l'extra ``transformers``.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    config = config or Config()
    tok = AutoTokenizer.from_pretrained(config.esgbert_model)
    model = AutoModelForSequenceClassification.from_pretrained(config.esgbert_model, num_labels=4).eval()
    id2label = model.config.id2label
    labels_map = {"Environmental": "E", "Social": "S", "Governance": "G", "None": "Other"}

    sentences = df[text_col].astype(str).tolist()
    labs, scores = [], []
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start + batch_size]
        enc = tok(batch, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        proba = torch.softmax(logits, dim=1)
        for i in range(len(batch)):
            idx = int(proba[i].argmax())
            raw_label = id2label[idx]
            labs.append(labels_map.get(raw_label, "Other"))
            scores.append(float(proba[i, idx]))

    out = df.copy()
    out["esg"] = labs
    out["esg_score"] = scores
    out = out[out["esg"].isin(["E", "S", "G"])].reset_index(drop=True)
    out["label"] = out["esg"].map({"E": 0, "S": 1, "G": 2})
    if config.apply_leakage_mask:
        out["sentence_model"] = out[text_col].map(mask_lexicon)
    return out


# --------------------------------------------------------------------- #
# 5. Combinazione di più funzioni di etichettatura (voto di maggioranza)
# --------------------------------------------------------------------- #
def label_dataframe_ensemble(df: pd.DataFrame, label_columns: list, config: Config | None = None,
                             text_col: str = "sentence") -> pd.DataFrame:
    """Combina PIU' etichettature gia' calcolate con voto di maggioranza,
    in stile Snorkel (supervisione debole programmatica): non una singola
    funzione di etichettatura, ma il consenso tra piu' di esse.

    A differenza di Snorkel "vero" (che stima un modello generativo
    dell'accuratezza/correlazione tra le funzioni di etichettatura), qui
    il voto e' un SEMPLICE voto di maggioranza -- piu' semplice da capire
    e diagnosticare, ma non pesa automaticamente le funzioni piu'
    affidabili piu' delle altre.

    ``label_columns``: lista di liste/Series di etichette pilastro
    (``"E"``/``"S"``/``"G"``/``"Other"``), una per ciascuna funzione di
    etichettatura da combinare, TUTTE della stessa lunghezza di ``df`` e
    nello stesso ordine (es. la colonna ``esg`` di piu' DataFrame prodotti
    da :func:`label_dataframe`, :func:`label_dataframe_tfidf`,
    :func:`label_dataframe_zero_shot`, ecc. -- purche' costruiti sulle
    stesse frasi, nello stesso ordine, PRIMA che ciascuna funzione filtri
    via le proprie righe "Other").

    Una frase e' etichettata "Other" se non c'e' maggioranza assoluta
    (es. 3 voti diversi su 3 funzioni), non solo se tutte le funzioni
    sono d'accordo su "Other".
    """
    from collections import Counter as _Counter

    config = config or Config()
    n = len(df)
    for col in label_columns:
        if len(col) != n:
            raise ValueError(
                f"Tutte le colonne di etichette devono avere la stessa lunghezza di df ({n}), "
                f"trovata una lunghezza {len(col)} -- assicurati di passare le etichette PRIMA "
                "che ciascuna funzione filtri via le proprie righe 'Other'."
            )

    labs = []
    for i in range(n):
        votes = [col[i] for col in label_columns]
        counts = _Counter(votes)
        best_label, best_count = counts.most_common(1)[0]
        if best_count > len(label_columns) / 2:
            labs.append(best_label)
        else:
            labs.append("Other")

    out = df.copy()
    out["esg"] = labs
    out = out[out["esg"].isin(["E", "S", "G"])].reset_index(drop=True)
    out["label"] = out["esg"].map({"E": 0, "S": 1, "G": 2})
    if config.apply_leakage_mask:
        out["sentence_model"] = out[text_col].map(mask_lexicon)
    return out
