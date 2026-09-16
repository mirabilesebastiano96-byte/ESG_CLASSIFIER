"""Estrazione di frasi da documenti (PDF, Word .docx, testo .txt).

I report ESG sono spesso PDF multi-colonna pieni di titoli, tabelle e box
statistici: un ``extract_text()`` ingenuo mescola il testo tra colonne.
Per il PDF questo modulo combina piu' livelli di robustezza:

- rilevamento boilerplate (righe ripetute su molte pagine -> header/footer)
- suddivisione ricorsiva in colonne dalle coordinate delle parole
- filtro per dimensione del font (esclude titoli e numeri delle card)
- segmentazione in paragrafi per stacco verticale
- filtro di qualita' finale sulla frase (lunghezza, punteggiatura,
  rapporto cifre/maiuscole, presenza di un verbo), applicato a TUTTI i
  formati per coerenza

Word (.docx) e testo semplice (.txt) usano estrattori piu' semplici (non
serve gestire colonne/font) ma lo stesso filtro di qualita' finale.
:func:`extract_document` riconosce il formato dall'estensione;
:func:`extract_directory` processa una cartella con un mix di formati.

Richiede l'extra ``extraction`` (``pip install esg-pillar-classifier[extraction]``),
che include sia ``pdfplumber`` (PDF) sia ``python-docx`` (Word).
"""
from __future__ import annotations

import glob
import os
import re
import statistics
from collections import Counter

import pandas as pd

from .config import Config

_DROP_LINE = re.compile(
    r"^(goals?|actions?|progress|introduction|sustainability|people|performance key|"
    r"goal|status|priority|learn more|with gratitude|source|fonte|nota|figura|tabella)\b",
    re.I,
)


def meta_from_filename(path: str) -> tuple[str, int | None]:
    """Ricava (azienda, anno) dal nome del file, atteso nella forma ``Azienda_Anno.pdf``."""
    base = os.path.splitext(os.path.basename(path))[0]
    parts = re.split(r"[_\-\s]+", base)
    company = parts[0] if parts else base
    m = re.search(r"(19|20)\d{2}", base)
    year = int(m.group(0)) if m else None
    if year is None:
        m2 = re.search(r"\d{1,2}-([A-Za-z]{3})-(\d{2})\b", base)
        if m2:
            yy = int(m2.group(2))
            year = 2000 + yy if yy < 50 else 1900 + yy
    return company, year


def _boilerplate_lines(pdf, min_pages_ratio: float = 1 / 3) -> set[str]:
    freq: Counter = Counter()
    npages = len(pdf.pages)
    for pg in pdf.pages:
        t = pg.extract_text() or ""
        for line in set(x.strip() for x in t.split("\n") if x.strip()):
            freq[re.sub(r"\d+", "#", line.lower())] += 1
    thresh = max(3, int(npages * min_pages_ratio))
    return {k for k, v in freq.items() if v >= thresh}


def _split_columns(words, x_lo: float, x_hi: float, depth: int = 0):
    if not words or depth > 4:
        return [words] if words else []
    width = x_hi - x_lo
    bins = max(40, int(width / 2))
    binw = width / bins
    occ = [0] * bins
    for w in words:
        a = int((float(w["x0"]) - x_lo) // binw)
        b = int((min(float(w["x1"]), x_hi) - x_lo) // binw)
        for i in range(max(0, a), min(bins, b + 1)):
            occ[i] += 1
    gmin = max(1, int(9 // binw))
    best = None
    i = 0
    while i < bins:
        if occ[i] == 0:
            j = i
            while j < bins and occ[j] == 0:
                j += 1
            if (j - i) >= gmin and i > 2 and j < bins - 2:
                if best is None or (j - i) > best[1] - best[0]:
                    best = (i, j)
            i = j
        else:
            i += 1
    if best is None:
        return [words]
    cut = x_lo + ((best[0] + best[1]) / 2) * binw
    left = [w for w in words if (float(w["x0"]) + float(w["x1"])) / 2 < cut]
    right = [w for w in words if (float(w["x0"]) + float(w["x1"])) / 2 >= cut]
    return _split_columns(left, x_lo, cut, depth + 1) + _split_columns(right, cut, x_hi, depth + 1)


def _page_paragraphs(page, boiler: set[str], head: float = 0.06, foot: float = 0.94) -> list[str]:
    width, height = float(page.width), float(page.height)
    words = page.extract_words(extra_attrs=["size"], use_text_flow=False, keep_blank_chars=False)
    words = [w for w in words if head * height <= float(w["top"]) <= foot * height]
    if not words:
        return []
    sizes = [float(w["size"]) for w in words if "size" in w]
    body = statistics.median(sizes) if sizes else 10
    words = [w for w in words if body * 0.80 <= float(w.get("size", body)) <= body * 1.20]
    if not words:
        return []
    cols = _split_columns(words, 0.0, width)
    paras: list[list[str]] = []
    line_h = body * 1.25
    for col in cols:
        if not col:
            continue
        col.sort(key=lambda w: (round(float(w["top"]) / 2), float(w["x0"])))
        lines: list[list[dict]] = []
        cur: list[dict] = []
        last = None
        for w in col:
            if last is None or abs(float(w["top"]) - last) <= 3:
                cur.append(w)
            else:
                lines.append(cur)
                cur = [w]
            last = float(w["top"])
        if cur:
            lines.append(cur)
        para: list[str] = []
        prev = None
        for ln in lines:
            top = min(float(x["top"]) for x in ln)
            txt = " ".join(x["text"] for x in ln).strip()
            norm = re.sub(r"\d+", "#", txt.lower())
            if _DROP_LINE.match(txt) or norm in boiler:
                if para:
                    paras.append(para)
                    para = []
                prev = top
                continue
            if prev is not None and (top - prev) > line_h * 1.8:
                if para:
                    paras.append(para)
                    para = []
            para.append(txt)
            prev = top
        if para:
            paras.append(para)
    out = []
    for para in paras:
        buf = ""
        for t in para:
            buf = buf[:-1] + t.lstrip() if buf.endswith("-") else ((buf + " " + t).strip() if buf else t)
        out.append(re.sub(r"\s+", " ", buf))
    return out


def is_real_sentence(s: str, check_verb: bool = True, min_w: int = 8, max_w: int = 45) -> bool:
    """Filtro di qualita': la stringa e' davvero una frase di prosa e non un frammento."""
    from nltk import pos_tag, word_tokenize

    s = s.strip()
    words = s.split()
    n = len(words)
    if n < min_w or n > max_w:
        return False
    if not re.match(r"[A-Z\"']", s):
        return False
    if not re.search(r"[.!?]['\"]?$", s):
        return False
    num = sum(1 for t in words if re.search(r"\d", t) or re.fullmatch(r"[^\w]+", t))
    if num / n > 0.20:
        return False
    if sum(1 for t in words if t[:1].isupper()) / n > 0.5:
        return False
    if check_verb and not any(tag.startswith("VB") for _, tag in pos_tag(word_tokenize(s))):
        return False
    return True


def extract_pdf(path: str, config: Config | None = None) -> list[dict]:
    """Estrae le frasi di prosa da un singolo PDF, column-aware.

    Ritorna una lista di dict con chiavi ``company``, ``year``, ``sentence``,
    ``page``, ``pdf``.
    """
    import pdfplumber

    config = config or Config()
    company, year = meta_from_filename(path)
    rows: list[dict] = []
    seen: set[str] = set()
    with pdfplumber.open(path) as pdf:
        boiler = _boilerplate_lines(pdf)
        for pno, page in enumerate(pdf.pages, start=1):
            for para in _page_paragraphs(page, boiler):
                from nltk.tokenize import sent_tokenize
                for s in sent_tokenize(para):
                    s = s.strip()
                    key = re.sub(r"\W+", "", s.lower())[:80]
                    if key in seen:
                        continue
                    if not is_real_sentence(s, config.check_verb, config.min_words_sent, config.max_words_sent):
                        continue
                    seen.add(key)
                    rows.append({
                        "company": company, "year": year, "sentence": s,
                        "page": pno, "pdf": os.path.basename(path),
                    })
    return rows


def extract_docx(path: str, config: Config | None = None) -> list[dict]:
    """Estrae le frasi di prosa da un documento Word (.docx).

    Più semplice dell'estrazione da PDF (niente colonne/font da gestire):
    legge i paragrafi in ordine e applica lo stesso filtro di qualità
    (:func:`is_real_sentence`) usato per il PDF, per coerenza tra formati.
    """
    from docx import Document

    config = config or Config()
    company, year = meta_from_filename(path)
    rows: list[dict] = []
    seen: set[str] = set()

    doc = Document(path)
    from nltk.tokenize import sent_tokenize

    for para_no, para in enumerate(doc.paragraphs, start=1):
        text = para.text.strip()
        if not text:
            continue
        for s in sent_tokenize(text):
            s = s.strip()
            key = re.sub(r"\W+", "", s.lower())[:80]
            if key in seen:
                continue
            if not is_real_sentence(s, config.check_verb, config.min_words_sent, config.max_words_sent):
                continue
            seen.add(key)
            rows.append({
                "company": company, "year": year, "sentence": s,
                "page": para_no, "pdf": os.path.basename(path),
            })
    return rows


def extract_txt(path: str, config: Config | None = None, encoding: str = "utf-8") -> list[dict]:
    """Estrae le frasi di prosa da un file di testo semplice (.txt)."""
    config = config or Config()
    company, year = meta_from_filename(path)
    rows: list[dict] = []
    seen: set[str] = set()

    with open(path, encoding=encoding, errors="ignore") as f:
        raw = f.read()

    from nltk.tokenize import sent_tokenize

    for para_no, para in enumerate(raw.split("\n\n"), start=1):
        para = para.strip()
        if not para:
            continue
        for s in sent_tokenize(para):
            s = s.strip()
            key = re.sub(r"\W+", "", s.lower())[:80]
            if key in seen:
                continue
            if not is_real_sentence(s, config.check_verb, config.min_words_sent, config.max_words_sent):
                continue
            seen.add(key)
            rows.append({
                "company": company, "year": year, "sentence": s,
                "page": para_no, "pdf": os.path.basename(path),
            })
    return rows


_EXTRACTORS = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".txt": extract_txt,
}


def extract_document(path: str, config: Config | None = None) -> list[dict]:
    """Estrae le frasi da un singolo documento, riconoscendo il formato dall'estensione.

    Formati supportati: ``.pdf``, ``.docx``, ``.txt``. Solleva ``ValueError``
    per estensioni non riconosciute (es. il vecchio formato ``.doc``, che
    richiede una conversione preliminare in ``.docx``).
    """
    config = config or Config()
    ext = os.path.splitext(path)[1].lower()
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        raise ValueError(
            f"Formato non supportato: '{ext}' (file: {path}). "
            f"Formati supportati: {', '.join(sorted(_EXTRACTORS))}. "
            f"Per il vecchio formato .doc, convertilo prima in .docx."
        )
    return extractor(path, config)


def extract_directory(doc_dir: str | None = None, config: Config | None = None,
                      use_cache: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Estrae le frasi da tutti i documenti di una cartella (``config.pdf_dir`` di default).

    Accetta una cartella con un mix di PDF, Word (.docx) e file di testo
    (.txt): ogni file viene instradato all'estrattore giusto in base
    all'estensione (vedi :func:`extract_document`).

    Con ``use_cache=True`` (default), se ``config.extract_cache`` esiste
    gia' viene ricaricato invece di ri-estrarre da zero.
    """
    config = config or Config()
    doc_dir = doc_dir or config.pdf_dir

    if use_cache and os.path.exists(config.extract_cache):
        if verbose:
            print(f"Ricaricato dalla cache: {config.extract_cache}")
        return pd.read_csv(config.extract_cache)

    doc_files: list[str] = []
    for ext in _EXTRACTORS:
        doc_files.extend(glob.glob(os.path.join(doc_dir, f"*{ext}")))
    doc_files.sort()

    if not doc_files:
        raise FileNotFoundError(
            f"Nessun documento supportato trovato in '{doc_dir}' "
            f"(formati: {', '.join(sorted(_EXTRACTORS))}). "
            f"Metti li' i report (nominati 'Azienda_Anno.ext') o aggiorna config.pdf_dir."
        )

    all_rows: list[dict] = []
    for i, f in enumerate(doc_files, 1):
        rows = extract_document(f, config)
        all_rows.extend(rows)
        if verbose:
            print(f"  [{i}/{len(doc_files)}] {os.path.basename(f)}: {len(rows)} frasi")

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["sentence"]).reset_index(drop=True)
    if use_cache:
        df.to_csv(config.extract_cache, index=False)
    return df
