"""Text mining a livello di documento/corpus: estrazione di keyword,
topic modeling, riassunto estrattivo, similarita' tra frasi.

Distinto da :mod:`esgpillar.text_analysis` (che lavora frase per frase:
POS, NER, sentiment) -- qui le tecniche guardano a un TESTO PIU' LUNGO
(un report, un insieme di frasi) o confrontano PIU' testi tra loro.
Nessuna di queste tecniche entra nella pipeline di classificazione dei
pilastri (:mod:`esgpillar.pipeline`): sono strumenti di analisi
complementari, utili ad esempio per capire di cosa parla un report prima
ancora di classificarlo, o per confrontare due report tra loro.

- **Estrazione di keyword/keyphrase**: tre tecniche.
  ``"tfidf"`` (le parole/frasi con il punteggio TF-IDF piu' alto nel
  documento rispetto al corpus), ``"rake"`` (Rapid Automatic Keyword
  Extraction -- estrae FRASI CHIAVE, non singole parole, via l'extra
  ``mining``), ``"textrank"`` (grafo di co-occorrenza tra parole +
  PageRank, via l'extra ``mining``).
- **Topic modeling**: LDA e NMF (entrambi in scikit-learn, nessuna
  dipendenza extra), per scoprire temi ricorrenti in un corpus senza
  bisogno di etichette.
- **Riassunto estrattivo**: TextRank a livello di FRASE (non parola) --
  un grafo di similarita' tra frasi + PageRank, poi si scelgono le frasi
  con punteggio piu' alto, in ordine originale.
- **Similarita'/paraphrase**: tra due frasi, via TF-IDF (leggero) o via
  un qualunque encoder di :mod:`esgpillar.embeddings` (es. SBERT, pensato
  apposta per la similarita' semantica tra frasi).
- **Leggibilita'**: tre indici classici (Flesch Reading Ease,
  Flesch-Kincaid Grade, Gunning Fog), via l'extra ``mining``.
- **Claim quantitativi**: estrazione di percentuali/importi/quantita' con
  il loro contesto, utile per isolare gli impegni numerici concreti di un
  report (a differenza delle affermazioni vaghe, che i numeri li evitano).
- **N-grammi/collocazioni**: bigrammi/trigrammi per frequenza o per PMI
  (Pointwise Mutual Information -- trova espressioni specialistiche del
  dominio anche se rare in assoluto), via ``nltk`` (nessun extra aggiuntivo).
- **Classificazione zero-shot**: etichette a scelta libera, senza
  addestrare nulla (via un modello di NLI), utile per esplorare nuove
  categorie prima di costruire un classificatore vero.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


# --------------------------------------------------------------------- #
# Estrazione di keyword/keyphrase
# --------------------------------------------------------------------- #
def keywords_tfidf(texts: Sequence[str], top_n: int = 10) -> list[tuple[str, float]]:
    """Le ``top_n`` parole con il punteggio TF-IDF medio piu' alto sull'intero
    corpus ``texts`` (ogni elemento e' un documento/frase). Nessuna
    dipendenza extra oltre a scikit-learn."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf = vec.fit_transform(texts)
    scores = np.asarray(tfidf.mean(axis=0)).ravel()
    vocab = vec.get_feature_names_out()
    order = np.argsort(-scores)[:top_n]
    return [(vocab[i], float(scores[i])) for i in order]


def keywords_rake(text: str, top_n: int = 10) -> list[tuple[str, float]]:
    """Estrae FRASI CHIAVE (non singole parole) con RAKE (Rapid Automatic
    Keyword Extraction): spezza il testo su stopword/punteggiatura, poi
    valuta ogni frase candidata per co-occorrenza delle parole al suo
    interno. Richiede l'extra ``mining`` (``rake-nltk``)."""
    from rake_nltk import Rake

    r = Rake()
    r.extract_keywords_from_text(text)
    ranked = r.get_ranked_phrases_with_scores()[:top_n]
    return [(phrase, float(score)) for score, phrase in ranked]


def _build_cooccurrence_graph(tokens: list[str], window: int = 4):
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(set(tokens))
    for i, tok in enumerate(tokens):
        for j in range(i + 1, min(i + window, len(tokens))):
            other = tokens[j]
            if other != tok:
                if graph.has_edge(tok, other):
                    graph[tok][other]["weight"] += 1
                else:
                    graph.add_edge(tok, other, weight=1)
    return graph


def keywords_textrank(text: str, top_n: int = 10, tokenizer=None, window: int = 4) -> list[tuple[str, float]]:
    """Estrae le ``top_n`` parole chiave con TextRank: un grafo dove i nodi
    sono le parole e gli archi collegano parole che co-occorrono entro
    ``window`` posizioni nel testo, pesati per numero di co-occorrenze;
    le parole vengono poi ranked con l'algoritmo PageRank sul grafo.
    Concettualmente lo stesso principio usato per il riassunto in
    :func:`summarize_textrank`, ma applicato a parole invece che a frasi.

    Se ``tokenizer`` (un :class:`esgpillar.tokenization.Tokenizer`) non e'
    passato, il testo viene semplicemente messo in minuscolo e diviso sugli
    spazi (nessuna rimozione di stopword). Richiede l'extra ``mining``
    (``networkx``).
    """
    import networkx as nx

    tokens = tokenizer(text) if tokenizer is not None else str(text).lower().split()
    if len(tokens) < 2:
        return [(t, 1.0) for t in tokens[:top_n]]
    graph = _build_cooccurrence_graph(tokens, window=window)
    scores = nx.pagerank(graph, weight="weight")
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]
    return [(word, float(score)) for word, score in ranked]


# --------------------------------------------------------------------- #
# Topic modeling
# --------------------------------------------------------------------- #
class TopicModel:
    """Incapsula un modello di topic (LDA o NMF) + il vettorizzatore usato
    per addestrarlo, cosi' i due restano sempre coerenti tra loro."""

    def __init__(self, model, vectorizer, method: str):
        self.model = model
        self.vectorizer = vectorizer
        self.method = method

    def top_words(self, n_words: int = 10) -> list[list[str]]:
        """Le ``n_words`` parole piu' rappresentative di ciascun topic."""
        vocab = self.vectorizer.get_feature_names_out()
        topics = []
        for component in self.model.components_:
            order = np.argsort(-component)[:n_words]
            topics.append([vocab[i] for i in order])
        return topics

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Distribuzione sui topic per ciascun testo (righe: documenti, colonne: topic)."""
        X = self.vectorizer.transform(texts)
        return self.model.transform(X)


def fit_topic_model(texts: Sequence[str], n_topics: int = 5, method: str = "lda", seed: int = 42) -> TopicModel:
    """Addestra un modello di topic non supervisionato sul corpus ``texts``.

    ``method``: ``"lda"`` (Latent Dirichlet Allocation -- probabilistico,
    interpretazione classica "documento = mix di topic") o ``"nmf"``
    (Non-negative Matrix Factorization -- spesso da' topic piu' netti e
    interpretabili su corpora piccoli/medi, nessuna assunzione
    probabilistica). Entrambi in scikit-learn, nessuna dipendenza extra.
    """
    from sklearn.decomposition import NMF, LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    if method == "lda":
        vectorizer = CountVectorizer(stop_words="english", max_features=5000)
        X = vectorizer.fit_transform(texts)
        model = LatentDirichletAllocation(n_components=n_topics, random_state=seed)
        model.fit(X)
    elif method == "nmf":
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        X = vectorizer.fit_transform(texts)
        model = NMF(n_components=n_topics, random_state=seed, init="nndsvda", max_iter=500)
        model.fit(X)
    else:
        raise ValueError(f"method={method!r} non riconosciuto. Usa 'lda' o 'nmf'.")

    return TopicModel(model=model, vectorizer=vectorizer, method=method)


# --------------------------------------------------------------------- #
# Riassunto estrattivo
# --------------------------------------------------------------------- #
def summarize_textrank(sentences: Sequence[str], n_sentences: int = 3) -> list[str]:
    """Riassunto ESTRATTIVO (seleziona frasi esistenti, non ne genera di
    nuove) con TextRank: un grafo dove i nodi sono le frasi e gli archi
    sono pesati dalla similarita' TF-IDF tra coppie di frasi; le frasi
    vengono ranked con PageRank, se ne scelgono le ``n_sentences``
    migliori, restituite nell'ORDINE ORIGINALE (per restare leggibili
    come riassunto coerente, non come lista sparsa). Richiede l'extra
    ``mining`` (``networkx``).

    Pensato per un singolo documento lungo (es. un report intero diviso
    in frasi), non per il corpus di training della classificazione ESG.
    """
    import networkx as nx
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    sentences = list(sentences)
    if len(sentences) <= n_sentences:
        return sentences

    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(sentences)
    sim_matrix = cosine_similarity(X)
    np.fill_diagonal(sim_matrix, 0)  # una frase non e' mai "simile a se stessa" nel grafo

    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph)
    ranked_idx = sorted(scores, key=lambda i: -scores[i])[:n_sentences]
    return [sentences[i] for i in sorted(ranked_idx)]  # riordinate come nel testo originale


# --------------------------------------------------------------------- #
# Similarita' / paraphrase
# --------------------------------------------------------------------- #
def similarity_tfidf(text_a: str, text_b: str) -> float:
    """Similarita' coseno tra due frasi via TF-IDF -- leggero, nessun
    embedding da caricare, ma cattura solo la sovrapposizione lessicale
    (due frasi che dicono la stessa cosa con parole diverse risultano
    poco simili). Per una similarita' SEMANTICA vera, vedi
    :func:`similarity_embeddings` con un encoder come SBERT."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform([text_a, text_b])
    return float(cosine_similarity(X[0], X[1])[0, 0])


def similarity_embeddings(text_a: str, text_b: str, encoder) -> float:
    """Similarita' coseno tra due frasi, usando un qualunque ENCODER di
    :mod:`esgpillar.embeddings` (o l'encoder gia' addestrato di un
    :class:`esgpillar.ESGPillarClassifier`, via ``clf._encoder``) per
    ottenere i vettori pooled. Con un encoder come SBERT (addestrato
    apposta per la similarita' tra frasi) cattura somiglianza SEMANTICA,
    non solo lessicale -- due frasi che dicono la stessa cosa con parole
    diverse possono risultare molto simili, a differenza di
    :func:`similarity_tfidf`."""
    _, pooled, _ = encoder([text_a, text_b])
    a, b = pooled[0], pooled[1]
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def find_similar_sentences(query: str, corpus: Sequence[str], encoder, top_k: int = 5) -> list[tuple[str, float]]:
    """Le ``top_k`` frasi di ``corpus`` piu' simili a ``query``, secondo
    l'encoder fornito (vedi :func:`similarity_embeddings`). Utile per
    trovare frasi ridondanti/duplicate tra piu' report, o per cercare
    frasi correlate a un tema in un corpus grande."""
    _, pooled_all, _ = encoder([query, *corpus])
    query_vec, corpus_vecs = pooled_all[0], pooled_all[1:]

    norms = np.linalg.norm(corpus_vecs, axis=1) * np.linalg.norm(query_vec)
    norms[norms == 0] = 1e-10
    sims = (corpus_vecs @ query_vec) / norms

    order = np.argsort(-sims)[:top_k]
    return [(corpus[i], float(sims[i])) for i in order]


# --------------------------------------------------------------------- #
# Leggibilita'
# --------------------------------------------------------------------- #
def readability_scores(text: str) -> dict:
    """Tre indici di leggibilita' classici (libreria ``textstat``, extra
    ``mining``), utili ad esempio per confrontare quanto un report ESG sia
    scritto in modo chiaro o inutilmente complesso (un linguaggio piu'
    complicato del necessario e' talvolta usato per offuscare, non solo
    per precisione tecnica -- un segnale rilevante in analisi di
    greenwashing, non solo di stile).

    - ``flesch_reading_ease``: 0-100, PIU' ALTO = piu' facile (100 = molto
      facile, sotto 30 = molto difficile, livello universitario avanzato).
    - ``flesch_kincaid_grade``: il grado scolastico USA stimato necessario
      per capire il testo (es. 12 = ultimo anno di superiori).
    - ``gunning_fog``: simile, pesa di piu' le parole "complesse" (3+ sillabe).

    Pensati per l'INGLESE (le formule si basano su sillabe/frasi tipiche
    dell'inglese): su testo in altre lingue i punteggi restano calcolabili
    ma la scala di riferimento non e' calibrata correttamente.
    """
    import textstat

    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
    }


def add_readability_columns(df, text_col: str = "sentence"):
    """Aggiunge le tre colonne di leggibilita' (vedi :func:`readability_scores`)
    al DataFrame, una per frase."""
    df = df.copy()
    scores = df[text_col].apply(readability_scores)
    for key in ["flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog"]:
        df[key] = scores.apply(lambda s: s[key])
    return df


# --------------------------------------------------------------------- #
# Estrazione di claim quantitativi
# --------------------------------------------------------------------- #
_QUANTITY_PATTERN = None


def _get_quantity_pattern():
    global _QUANTITY_PATTERN
    if _QUANTITY_PATTERN is None:
        import re
        _QUANTITY_PATTERN = re.compile(
            r'[\$€£]\s?\d+(?:[.,]\d+)?\s?(?:million|billion|thousand)?'
            r'|\d+(?:[.,]\d+)?\s?(?:%|percent|per\s?cent)'
            r'|\d+(?:[.,]\d+)?\s?(?:million|billion|thousand)\b',
            re.IGNORECASE,
        )
    return _QUANTITY_PATTERN


def extract_quantitative_claims(text: str, context_words: int = 5) -> list[dict]:
    """Estrae valori quantitativi (percentuali, importi in valuta, quantita'
    in milioni/miliardi) dal testo, ciascuno con il contesto circostante --
    utile per un report ESG, dove gli impegni concreti sono spesso espressi
    con numeri ("ridotto del 25%", "investiti 10 milioni di dollari") a
    differenza delle affermazioni vaghe che i numeri li evitano del tutto.

    Ritorna una lista di dict ``{"value": ..., "context_before": ...,
    "context_after": ..., "start": ..., "end": ...}`` (``start``/``end``
    sono gli offset carattere nel testo originale, utili per evidenziare
    il match in un'interfaccia).

    Nota implementativa: il contesto di ciascun valore viene calcolato
    dagli offset di parola nel testo ORIGINALE, non consumando il testo
    match dopo match -- altrimenti la finestra di contesto di un valore
    rischierebbe di "inghiottire" il valore successivo, facendolo sparire
    dai risultati.
    """
    text = str(text)
    pattern = _get_quantity_pattern()

    words = text.split()
    word_spans = []
    pos = 0
    for w in words:
        start = text.index(w, pos)
        word_spans.append((start, start + len(w)))
        pos = start + len(w)

    results = []
    for m in pattern.finditer(text):
        before_idx = [i for i, (s, e) in enumerate(word_spans) if e <= m.start()][-context_words:]
        after_idx = [i for i, (s, e) in enumerate(word_spans) if s >= m.end()][:context_words]
        results.append({
            "value": m.group().strip(),
            "context_before": " ".join(words[i] for i in before_idx),
            "context_after": " ".join(words[i] for i in after_idx),
            "start": m.start(),
            "end": m.end(),
        })
    return results


def add_quantitative_claims_column(df, text_col: str = "sentence", out_col: str = "quantitative_claims"):
    """Aggiunge una colonna con la lista dei claim quantitativi trovati in
    ciascuna frase (vedi :func:`extract_quantitative_claims`)."""
    df = df.copy()
    df[out_col] = df[text_col].apply(extract_quantitative_claims)
    return df


# --------------------------------------------------------------------- #
# N-gram e collocazioni
# --------------------------------------------------------------------- #
def top_ngrams(texts: Sequence[str], n: int = 2, method: str = "frequency", top_n: int = 10,
              tokenizer=None) -> list[tuple[tuple[str, ...], float]]:
    """Gli n-grammi (``n=2``: bigrammi, ``n=3``: trigrammi) piu' rilevanti
    nel corpus ``texts``, secondo due criteri alternativi.

    ``method="frequency"``: i piu' FREQUENTI in assoluto -- semplice, ma
    dominato da combinazioni comuni anche non specialistiche (es. "of the").
    ``method="pmi"``: Pointwise Mutual Information -- gli n-grammi le cui
    parole compaiono insieme PIU' SPESSO DI QUANTO CI SI ASPETTEREBBE dal
    caso, anche se rare in assoluto (es. "greenhouse gas" avrebbe un PMI
    alto anche se compare poche volte, perche' "greenhouse" e "gas" quasi
    non compaiono mai separatamente) -- generalmente piu' utile per trovare
    espressioni davvero specialistiche del dominio.

    Se ``tokenizer`` (un :class:`esgpillar.tokenization.Tokenizer`) non e'
    passato, il testo viene tokenizzato con ``nltk.word_tokenize`` (nessuna
    rimozione di stopword). Nessuna dipendenza extra (nltk e' gia' richiesto
    per la tokenizzazione condivisa).
    """
    from nltk import word_tokenize
    from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
    from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

    if n not in (2, 3):
        raise ValueError(f"n={n} non supportato. Usa 2 (bigrammi) o 3 (trigrammi).")

    all_tokens = []
    for t in texts:
        all_tokens.extend(tokenizer(t) if tokenizer is not None else word_tokenize(str(t)))

    if n == 2:
        finder = BigramCollocationFinder.from_words(all_tokens)
        measures = BigramAssocMeasures
    else:
        finder = TrigramCollocationFinder.from_words(all_tokens)
        measures = TrigramAssocMeasures

    if method == "frequency":
        return finder.ngram_fd.most_common(top_n)
    if method == "pmi":
        finder.apply_freq_filter(2)  # il PMI su n-grammi visti una sola volta e' rumore, non segnale
        scored = finder.score_ngrams(measures.pmi)
        return scored[:top_n]
    raise ValueError(f"method={method!r} non riconosciuto. Usa 'frequency' o 'pmi'.")


# --------------------------------------------------------------------- #
# Classificazione zero-shot
# --------------------------------------------------------------------- #
def _load_zero_shot_pipeline(model: str):
    """Isolata in una funzione a parte (non chiamata direttamente da
    :func:`zero_shot_classify`) cosi' i test possono sostituirla con un
    modello finto senza dover aggirare il caricamento lazy di
    ``transformers`` (``transformers.pipeline`` non e' un attributo di
    modulo normale -- e' risolto pigramente, quindi un patch diretto su
    ``transformers.pipeline`` non ha effetto)."""
    from transformers import pipeline
    return pipeline("zero-shot-classification", model=model)


def zero_shot_classify(text: str, candidate_labels: Sequence[str], model: str = "facebook/bart-large-mnli") -> dict:
    """Classifica ``text`` su etichette SCELTE DA TE AL VOLO, senza
    addestrare nulla -- a differenza di tutto il resto della libreria
    (:class:`esgpillar.ESGPillarClassifier` richiede un ``fit()`` su dati
    etichettati). Funziona tramite un modello di NLI (Natural Language
    Inference): internamente, verifica quanto il testo "implica" ciascuna
    frase-modello "This example is about {label}." -- non e' un
    classificatore addestrato sulle tue etichette, e' un ragionamento
    linguistico generico applicato al volo.

    Utile per ESPLORARE etichette nuove prima di decidere se vale la pena
    costruire un dataset ed addestrare un classificatore vero (piu'
    accurato, ma richiede lavoro); MENO adatto come sostituto definitivo
    di un modello addestrato sui tuoi dati -- l'accuratezza zero-shot e'
    tipicamente inferiore a un classificatore fine-tunato sul tuo dominio.

    Ritorna un dict ``{"labels": [...], "scores": [...]}``, ordinato dal
    punteggio piu' alto. Richiede l'extra ``transformers``.
    """
    classifier = zero_shot_classify._pipeline_cache.get(model)
    if classifier is None:
        classifier = _load_zero_shot_pipeline(model)
        zero_shot_classify._pipeline_cache[model] = classifier
    result = classifier(str(text), list(candidate_labels))
    return {"labels": result["labels"], "scores": result["scores"]}


zero_shot_classify._pipeline_cache = {}


def add_zero_shot_column(df, candidate_labels: Sequence[str], text_col: str = "sentence",
                         out_col: str = "zero_shot_label", model: str = "facebook/bart-large-mnli"):
    """Aggiunge la migliore etichetta zero-shot (e il suo punteggio) per
    ciascuna frase del DataFrame, tra ``candidate_labels``."""
    df = df.copy()
    results = df[text_col].apply(lambda t: zero_shot_classify(t, candidate_labels, model=model))
    df[out_col] = results.apply(lambda r: r["labels"][0])
    df[f"{out_col}_score"] = results.apply(lambda r: r["scores"][0])
    return df
