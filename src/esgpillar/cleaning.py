"""Pulizia del testo estratto: rumore tipico dell'estrazione da PDF,
frasi troppo corte/lunghe, frammenti tabellari, duplicati (anche
boilerplate ripetuto tra pagine/anni), e opzionalmente lingua diversa
da quella target.

Su un dataset gia' curato l'effetto e' piccolo; diventa importante
quando le frasi provengono da PDF grezzi (vedi :mod:`esgpillar.extraction`).

Oltre a :func:`basic_clean` (nella pipeline di default via
:func:`clean_dataframe`), undici tecniche ADDIZIONALI, da applicare
esplicitamente quando servono (extra ``cleaning``):
:func:`expand_contractions` (don't -> do not), :func:`correct_spelling`
(dizionario inglese generico -- attenzione ai falsi positivi su termini
tecnici), :func:`normalize_numbers` (sostituzione con placeholder fisso),
:func:`expand_corporate_abbreviations` (Corp. -> Corporation, ecc.),
:func:`segment_sentences`/:func:`split_into_sentences` (spezza paragrafi
in frasi singole), :func:`remove_accents` (café -> cafe),
:func:`normalize_currency` ($10 -> USD 10), :func:`detect_boilerplate`/
:func:`remove_boilerplate_column` (intestazioni/piè di pagina ripetuti
IDENTICI tra piu' documenti), :func:`fix_encoding` (ripara mojibake, es.
"sociÃ©tÃ©" -> "société" -- diverso da remove_accents, che invece
rimuove intenzionalmente gli accenti da un testo gia' leggibile),
:func:`find_near_duplicates`/:func:`remove_near_duplicates` (frasi QUASI
identiche, non solo identiche in senso stretto),
:func:`is_tabular_fragment`/:func:`remove_tabular_fragments` (righe che
sono probabilmente frammenti di tabella estratti male, non frasi vere).
"""
from __future__ import annotations

import re
from collections.abc import Sequence

import pandas as pd

from .config import Config


def basic_clean(text: str) -> str:
    """Pulizia di base: URL/email rimossi, trattini di fine riga ricuciti,
    apici tipografici normalizzati, bullet/numerazione iniziale rimossa."""
    s = str(text)
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"\S+@\S+", " ", s)
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)  # trattino di fine riga (PDF)
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"[\n\r\t]", " ", s)
    s = re.sub(r"^[\u2022\-\*\d.)\s]{1,6}(?=[A-Z])", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# --------------------------------------------------------------------- #
# Tecniche di pulizia aggiuntive (opzionali, non nella pipeline di default)
# --------------------------------------------------------------------- #
def expand_contractions(text: str) -> str:
    """Espande le contrazioni inglesi (``don't`` -> ``do not``, ``can't``
    -> ``cannot``). Utile PRIMA della tokenizzazione condivisa: senza
    espansione, ``"don't"`` puo' essere spezzato in modo incoerente a
    seconda del tokenizzatore, e la negazione (spesso il pezzo di
    informazione piu' importante della frase) rischia di andare persa o
    frammentata in modo imprevedibile. Richiede l'extra ``cleaning``."""
    import contractions
    return contractions.fix(str(text))


_CORPORATE_ABBREVIATIONS = {
    r"\bCorp\.": "Corporation", r"\bInc\.": "Incorporated", r"\bLtd\.": "Limited",
    r"\bCo\.": "Company", r"\bS\.p\.A\.": "Societa per Azioni", r"\bS\.A\.": "Societe Anonyme",
    r"\bLLC\b": "Limited Liability Company", r"\bplc\b": "public limited company",
}


def expand_corporate_abbreviations(text: str, extra: dict[str, str] | None = None) -> str:
    """Espande abbreviazioni societarie comuni (``Corp.`` -> ``Corporation``,
    ``Inc.`` -> ``Incorporated``, ecc.). Utile per coerenza col resto del
    testo, e perche' un punto finale dentro un'abbreviazione (``Corp.``)
    puo' confondere strumenti che segmentano le frasi sul punto.

    ``extra``: dizionario aggiuntivo ``{pattern_regex: sostituzione}`` per
    abbreviazioni specifiche del tuo corpus, oltre a quelle di default
    (case-insensitive, applicate DOPO quelle di default)."""
    s = str(text)
    for pattern, repl in _CORPORATE_ABBREVIATIONS.items():
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
    if extra:
        for pattern, repl in extra.items():
            s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
    return s


def normalize_numbers(text: str, placeholder: str = "<NUM>") -> str:
    """Sostituisce ogni numero con un placeholder fisso (default ``"<NUM>"``).

    Utile per rappresentazioni che altrimenti tratterebbero ogni valore
    numerico diverso (``"25%"``, ``"30%"``, ``"1.5%"``) come un token
    completamente distinto e mai visto prima, senza generalizzare che si
    tratta comunque di "una percentuale" -- normalizzando, il modello puo'
    imparare il pattern "c'e' un numero qui" senza dover vedere ogni
    valore specifico nel training.

    ATTENZIONE: se usi anche :func:`esgpillar.text_mining.extract_quantitative_claims`
    sullo stesso testo, applicalo PRIMA della normalizzazione -- dopo, i
    valori numerici non ci sono piu' da estrarre."""
    return re.sub(r"\b\d+(?:[.,]\d+)?\b", placeholder, str(text))


def correct_spelling(text: str, max_corrections: int | None = None) -> str:
    """Corregge le parole probabilmente scritte in modo errato (dizionario
    inglese), lasciando invariate quelle non riconosciute come errori
    (nomi propri, termini tecnici non standard, sigle) -- ``pyspellchecker``
    e' un correttore GENERICO, non specifico ESG: puo' "correggere"
    erroneamente un termine tecnico legittimo ma raro in un dizionario
    generico scambiandolo per un errore di battitura.

    ``max_corrections``: se impostato, corregge al massimo questo numero
    di parole per frase (protezione contro un testo molto rumoroso dove
    "correggere tutto" rischierebbe di stravolgere il significato
    originale). Richiede l'extra ``cleaning``, ed e' relativamente lento
    su corpora grandi (un controllo per parola)."""
    from spellchecker import SpellChecker

    sp = correct_spelling._checker if hasattr(correct_spelling, "_checker") else SpellChecker()
    correct_spelling._checker = sp

    words = str(text).split()
    unknown = sp.unknown(words)
    if not unknown:
        return str(text)

    n_corrected = 0
    out_words = []
    for w in words:
        if w.lower() in unknown and (max_corrections is None or n_corrected < max_corrections):
            correction = sp.correction(w.lower())
            out_words.append(correction if correction else w)
            n_corrected += 1
        else:
            out_words.append(w)
    return " ".join(out_words)


def segment_sentences(text: str) -> list[str]:
    """Spezza un testo (es. un paragrafo intero estratto da un PDF) in
    frasi singole, con ``nltk.sent_tokenize`` -- gestisce correttamente
    abbreviazioni comuni come "Dr." o "Corp." senza spezzare la frase li'
    per sbaglio (a differenza di un semplice ``.split(".")``).

    Utile quando l'estrazione da PDF (:mod:`esgpillar.extraction`) da'
    blocchi di testo piu' grandi di una singola frase -- il resto della
    pipeline (etichettatura, classificazione) lavora a livello di FRASE,
    non di paragrafo."""
    from nltk import sent_tokenize
    from .tokenization import ensure_nltk_data
    ensure_nltk_data()
    return sent_tokenize(str(text))


def split_into_sentences(df: pd.DataFrame, text_col: str = "sentence") -> pd.DataFrame:
    """Applica :func:`segment_sentences` a un intero DataFrame, ESPLODENDO
    ogni riga (un paragrafo) in piu' righe (una per frase) -- le altre
    colonne (es. ``company``, ``year``) vengono duplicate su ogni frase
    risultante dallo stesso paragrafo originale."""
    out = df.copy()
    out[text_col] = out[text_col].map(segment_sentences)
    return out.explode(text_col, ignore_index=True)


def remove_accents(text: str) -> str:
    """Rimuove accenti/diacritici, convertendo ogni carattere al suo
    equivalente ASCII piu' vicino (``café`` -> ``cafe``, ``Müller`` ->
    ``Muller``). Utile su testo con encoding sporco -- comune nell'
    estrazione da PDF quando i caratteri accentati vengono estratti male
    -- o su corpora multilingua dove vuoi un trattamento uniforme.

    ATTENZIONE: e' una perdita di informazione linguistica reale, non
    solo cosmetica -- su corpora in lingue dove gli accenti cambiano il
    significato della parola (es. francese, italiano) puo' introdurre
    ambiguita'. Da usare con consapevolezza, non come pulizia automatica
    di default. Richiede l'extra ``cleaning``."""
    from unidecode import unidecode
    return unidecode(str(text))


_CURRENCY_PATTERNS = [
    (r"\$\s?(?=\d)", "USD "),
    (r"€\s?(?=\d)", "EUR "),
    (r"£\s?(?=\d)", "GBP "),
    (r"¥\s?(?=\d)", "JPY "),
    (r"\bdollars?\b", "USD"),
    (r"\beuros?\b", "EUR"),
    (r"\bpounds?( sterling)?\b", "GBP"),
    (r"\byen\b", "JPY"),
]


def normalize_currency(text: str, extra: dict[str, str] | None = None) -> str:
    """Normalizza simboli/nomi di valuta ai rispettivi codici ISO
    (``$10`` -> ``USD 10``, ``10 dollars`` -> ``10 USD``... in realta'
    entrambi diventano semplicemente ``USD`` accanto al numero, non in
    una posizione fissa -- l'obiettivo e' rendere il TERMINE valuta
    uniforme, non riordinare la frase).

    Perche' puo' aiutare: senza normalizzazione, ``"$10 million"`` e
    ``"10 million dollars"`` condividono il concetto (un importo in
    dollari) ma non nessuna parola-chiave testuale in comune per un
    modello lessicale -- dopo la normalizzazione, entrambi contengono
    ``"USD"``.

    ``extra``: dizionario aggiuntivo ``{pattern_regex: sostituzione}`` per
    altre valute oltre alle quattro di default (USD/EUR/GBP/JPY),
    applicato dopo quelle di default."""
    s = str(text)
    for pattern, repl in _CURRENCY_PATTERNS:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
    if extra:
        for pattern, repl in extra.items():
            s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
    return s


def detect_boilerplate(documents: dict[str, list[str]], min_doc_fraction: float = 0.3,
                       min_doc_count: int = 2) -> list[str]:
    """Individua frasi/righe BOILERPLATE -- intestazioni, piè di pagina,
    disclaimer ripetuti identici tra piu' documenti -- confrontando PIU'
    documenti tra loro: una frase che compare (quasi) identica in molti
    documenti diversi difficilmente e' un contenuto originale specifico
    di ciascun report, piu' probabilmente e' un elemento ripetuto del
    template (es. "Copyright 2023 Acme Corp. All rights reserved.").

    A differenza della deduplica dentro un SINGOLO documento (gia' in
    :func:`clean_dataframe`), qui il segnale e' la RIPETIZIONE TRA
    documenti diversi -- una frase specifica di UN report (anche se
    ripetuta piu' volte al suo interno) non verrebbe segnalata da questa
    funzione se non compare anche in altri documenti.

    ``documents``: dict ``{nome_documento: [frase1, frase2, ...]}``.
    ``min_doc_fraction``: frazione minima di documenti in cui una frase
    (normalizzata: minuscolo, spazi compattati) deve comparire per essere
    considerata boilerplate (default 0.3 = almeno il 30% dei documenti).
    ``min_doc_count``: soglia minima ASSOLUTA di documenti (default 2),
    applicata INSIEME a ``min_doc_fraction`` (si usa il piu' alto tra i
    due, come numero di documenti richiesto). Necessaria perche' con
    poche documenti una soglia solo percentuale diventa degenere: con
    2 documenti, "compare in almeno 1" e' gia' matematicamente il 50% --
    quindi ogni frase, anche se specifica di un solo documento,
    supererebbe una soglia percentuale del 30% o del 50% senza questo
    secondo controllo.

    Ritorna la lista delle frasi (nella forma originale della prima
    occorrenza) giudicate boilerplate."""
    from collections import defaultdict

    n_docs = len(documents)
    if n_docs == 0:
        return []

    sentence_to_docs: dict[str, set[str]] = defaultdict(set)
    original_form: dict[str, str] = {}
    for doc_name, sentences in documents.items():
        for s in sentences:
            key = re.sub(r"\s+", " ", str(s).strip().lower())
            if not key:
                continue
            sentence_to_docs[key].add(doc_name)
            original_form.setdefault(key, s)

    threshold = max(min_doc_fraction * n_docs, min_doc_count)
    return [original_form[key] for key, docs in sentence_to_docs.items() if len(docs) >= threshold]


def remove_boilerplate_column(df: pd.DataFrame, doc_col: str = "company", text_col: str = "sentence",
                              min_doc_fraction: float = 0.3, min_doc_count: int = 2) -> pd.DataFrame:
    """Rimuove dal DataFrame le righe la cui frase e' stata rilevata come
    boilerplate (vedi :func:`detect_boilerplate`), raggruppando i
    documenti secondo ``doc_col`` (tipicamente ``"company"`` o un
    identificativo di report/anno)."""
    documents = df.groupby(doc_col)[text_col].apply(list).to_dict()
    boilerplate = set(detect_boilerplate(documents, min_doc_fraction=min_doc_fraction, min_doc_count=min_doc_count))
    return df[~df[text_col].isin(boilerplate)].reset_index(drop=True)


def fix_encoding(text: str) -> str:
    """Ripara MOJIBAKE -- testo decodificato con l'encoding sbagliato (es.
    UTF-8 letto come Latin-1: ``"société"`` diventa ``"sociÃ©tÃ©"``) --
    un problema comune nell'estrazione da PDF con metadati di encoding
    inconsistenti tra pagine o produttori diversi del documento.

    Diverso da :func:`remove_accents`: quest'ultimo presuppone un testo
    gia' leggibile e RIMUOVE gli accenti (perdita di informazione
    intenzionale); questa funzione invece RIPARA un testo illeggibile per
    un errore di decodifica, ricostruendo i caratteri originali corretti
    (nessuna perdita di informazione se la riparazione riesce). Usa
    ``ftfy`` ("fixes text for you"), che riconosce automaticamente i
    pattern di mojibake piu' comuni. Su testo gia' pulito non ha alcun
    effetto (e' sicuro da applicare "per sicurezza" anche quando non sei
    certo che il problema sia presente). Richiede l'extra ``cleaning``."""
    import ftfy
    return ftfy.fix_text(str(text))


def find_near_duplicates(texts: Sequence[str], threshold: float = 90.0) -> list[tuple[int, int, float]]:
    """Trova coppie di frasi QUASI identiche (non identiche in senso
    stretto) -- a differenza della deduplica esatta gia' in
    :func:`clean_dataframe` (che confronta stringhe normalizzate ma
    identiche), qui il confronto e' per SIMILARITA': due frasi con una
    sola parola diversa, o con un refuso, vengono comunque rilevate come
    "quasi duplicate".

    Ritorna una lista di terne ``(indice_i, indice_j, punteggio)`` per
    ogni coppia con similarita' (Levenshtein normalizzato, 0-100) >=
    ``threshold``, con ``i < j`` (ogni coppia riportata una sola volta).

    ATTENZIONE: il confronto e' O(n^2) -- su corpora molto grandi (decine
    di migliaia di frasi) puo' diventare lento; su un corpus di poche
    migliaia di frasi (tipico per un progetto di tesi) resta comunque
    pratico. Richiede l'extra ``cleaning`` (``rapidfuzz``)."""
    from rapidfuzz import fuzz

    texts = list(texts)
    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = fuzz.ratio(str(texts[i]), str(texts[j]))
            if score >= threshold:
                pairs.append((i, j, score))
    return pairs


def remove_near_duplicates(df: pd.DataFrame, text_col: str = "sentence", threshold: float = 90.0) -> pd.DataFrame:
    """Rimuove dal DataFrame le frasi quasi-duplicate (vedi
    :func:`find_near_duplicates`), tenendo solo la PRIMA occorrenza di
    ciascun gruppo di frasi simili tra loro."""
    texts = df[text_col].tolist()
    pairs = find_near_duplicates(texts, threshold=threshold)
    to_drop = {j for _, j, _ in pairs}  # tiene sempre l'indice piu' basso (i), scarta j
    return df.drop(df.index[list(to_drop)]).reset_index(drop=True)


def is_tabular_fragment(text: str, min_word_ratio: float = 0.4, max_digit_ratio: float = 0.4) -> bool:
    """Rileva se ``text`` e' probabilmente un frammento di TABELLA estratto
    male da un PDF (molte cifre/simboli isolati, poche parole di senso
    compiuto) piuttosto che una frase vera -- es. una riga come
    ``"12.5 34.2 8.9 -- 156.3"`` (valori numerici di una tabella,
    estratti come se fosse testo continuo).

    Diverso dal filtro ``letter_ratio`` gia' in :func:`clean_dataframe`
    (che guarda la proporzione di caratteri alfabetici sul totale): qui
    si guardano separatamente la proporzione di PAROLE VERE (token
    alfabetici di almeno 2 lettere) sul totale dei token, e la proporzione
    di token che sono PURAMENTE NUMERICI -- un criterio piu' mirato a
    distinguere "poche parole intervallate da molti numeri" (tipico di
    una tabella) da "frase con qualche numero dentro" (es. "Revenue grew
    25% this year"), che il solo letter_ratio potrebbe confondere.

    Ritorna ``True`` se il testo ha probabilmente origine tabellare."""
    tokens = str(text).split()
    if not tokens:
        return True

    word_tokens = sum(1 for t in tokens if re.search(r"[a-zA-Z]{2,}", t))
    digit_tokens = sum(1 for t in tokens if re.fullmatch(r"[\d.,%$€£+-]+", t))

    word_ratio = word_tokens / len(tokens)
    digit_ratio = digit_tokens / len(tokens)
    return word_ratio < min_word_ratio and digit_ratio > max_digit_ratio


def remove_tabular_fragments(df: pd.DataFrame, text_col: str = "sentence",
                             min_word_ratio: float = 0.4, max_digit_ratio: float = 0.4) -> pd.DataFrame:
    """Rimuove dal DataFrame le righe rilevate come frammenti tabellari
    (vedi :func:`is_tabular_fragment`)."""
    mask = df[text_col].map(lambda t: not is_tabular_fragment(t, min_word_ratio, max_digit_ratio))
    return df[mask].reset_index(drop=True)


def _letter_ratio(s: str) -> float:
    n = len(s.replace(" ", ""))
    return (sum(c.isalpha() for c in s) / n) if n else 0.0


def clean_dataframe(df: pd.DataFrame, config: Config | None = None,
                    text_col: str = "sentence", verbose: bool = True) -> pd.DataFrame:
    """Applica l'intera catena di pulizia a un DataFrame di frasi.

    La colonna originale viene conservata in ``{text_col}_raw``; ``text_col``
    diventa la versione pulita, usata da tutte le fasi successive.
    """
    config = config or Config()
    out = df.copy()
    out[f"{text_col}_raw"] = out[text_col]
    funnel = [("frasi iniziali", len(out))]

    out[text_col] = out[f"{text_col}_raw"].map(basic_clean)
    out = out[out[text_col].str.len() > 0].reset_index(drop=True)
    funnel.append(("dopo pulizia base", len(out)))

    wc = out[text_col].str.split().apply(len)
    out = out[(wc >= config.min_words) & (wc <= config.max_words)].reset_index(drop=True)
    funnel.append((f"lunghezza {config.min_words}-{config.max_words} parole", len(out)))

    out = out[out[text_col].map(_letter_ratio) >= config.letter_ratio].reset_index(drop=True)
    funnel.append(("rimossi frammenti numerici/simbolici", len(out)))

    if config.dedup_mode == "normalized":
        key = out[text_col].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    else:
        key = out[text_col]
    out = out.loc[~key.duplicated()].reset_index(drop=True)
    funnel.append((f"deduplicate ({config.dedup_mode})", len(out)))

    if config.lang_filter:
        try:
            from langdetect import DetectorFactory, detect
            DetectorFactory.seed = config.seed

            def _is_target(s: str) -> bool:
                try:
                    return detect(s) == config.target_lang
                except Exception:
                    return True  # in dubbio, tieni (frasi troppo corte per langdetect)

            out = out[out[text_col].map(_is_target)].reset_index(drop=True)
            funnel.append((f"lingua = '{config.target_lang}'", len(out)))
        except ImportError:
            if verbose:
                print("langdetect non installato ([lang] extra): filtro lingua saltato.")

    if verbose:
        print("Funnel di pulizia:")
        for step, n in funnel:
            print(f"  {n:5d}  {step}")
        removed = funnel[0][1] - len(out)
        pct = 100 * removed / funnel[0][1] if funnel[0][1] else 0
        print(f"Rimosse in totale: {removed} frasi su {funnel[0][1]} ({pct:.1f}%)")

    return out
