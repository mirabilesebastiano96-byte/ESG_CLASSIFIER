"""Tokenizzatore condiviso, usato da tutte le rappresentazioni testuali.

Pipeline: lowercase -> rimozione stopword inglesi -> rimozione nomi
azienda (cosi' il modello non impara "se compare 'Amazon' -> classe X")
-> rimozione termini ESG generici (non distintivi di nessun pilastro)
-> lemmatizzazione POS-aware (emissions -> emission, reducing -> reduce).

Richiede i corpora nltk: punkt, averaged_perceptron_tagger, stopwords,
wordnet, omw-1.4 (scaricabili con :func:`ensure_nltk_data`).
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from .config import Config

_NLTK_PACKAGES = [
    "punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    "stopwords", "wordnet", "omw-1.4",
]


def ensure_nltk_data(quiet: bool = True) -> None:
    """Scarica i corpora nltk necessari, se mancanti. Sicuro da richiamare piu' volte."""
    import nltk
    for pkg in _NLTK_PACKAGES:
        try:
            nltk.download(pkg, quiet=quiet)
        except Exception:
            pass


class Tokenizer:
    """Tokenizzatore condiviso, con esclusione di nomi azienda specifici del corpus.

    Esempio
    -------
    >>> tok = Tokenizer.from_companies(["Amazon", "Enel"])
    >>> tok("Amazon reduced its emissions significantly.")
    ['reduce', 'emission', 'significantly']
    """

    def __init__(self, company_tokens: Iterable[str] = (),
                generic_terms: Iterable[str] = Config().generic_esg_terms):
        ensure_nltk_data()
        from nltk import pos_tag
        from nltk.corpus import stopwords as nltk_stopwords, wordnet
        from nltk.stem import WordNetLemmatizer

        self._pos_tag = pos_tag
        self._wordnet = wordnet
        self._lemmatizer = WordNetLemmatizer()
        self._stopwords = set(nltk_stopwords.words("english"))
        self._generic_terms = set(generic_terms)
        self._company_tokens = set(company_tokens)
        self._wordnet_pos = {
            "J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV,
        }

    @classmethod
    def from_companies(cls, companies: Iterable[str], **kwargs) -> "Tokenizer":
        """Costruisce il tokenizzatore estraendo i token dai nomi azienda del corpus."""
        company_tokens: set[str] = set()
        for c in companies:
            company_tokens.update(re.findall(r"[a-zA-Z]{2,}", str(c).lower()))
        return cls(company_tokens=company_tokens, **kwargs)

    def _lemma(self, tok: str, tag: str) -> str:
        return self._lemmatizer.lemmatize(tok, self._wordnet_pos.get(tag[0], self._wordnet.NOUN))

    def __call__(self, text: str) -> list[str]:
        toks = re.findall(r"[a-zA-Z]{2,}", str(text).lower())
        toks = [t for t in toks
               if t not in self._stopwords
               and t not in self._generic_terms
               and t not in self._company_tokens]
        if not toks:
            return []
        return [self._lemma(t, tag) for t, tag in self._pos_tag(toks)]

    @property
    def n_stopwords(self) -> int:
        return len(self._stopwords)

    @property
    def n_company_tokens(self) -> int:
        return len(self._company_tokens)
