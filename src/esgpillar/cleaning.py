"""Pulizia del testo estratto: rumore tipico dell'estrazione da PDF,
frasi troppo corte/lunghe, frammenti tabellari, duplicati (anche
boilerplate ripetuto tra pagine/anni), e opzionalmente lingua diversa
da quella target.

Su un dataset gia' curato l'effetto e' piccolo; diventa importante
quando le frasi provengono da PDF grezzi (vedi :mod:`esgpillar.extraction`).
"""
from __future__ import annotations

import re

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
