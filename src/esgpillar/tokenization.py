"""Tokenizzatore condiviso, usato da tutte le rappresentazioni testuali.

Pipeline: lowercase -> rimozione stopword inglesi -> rimozione nomi
azienda (cosi' il modello non impara "se compare 'Amazon' -> classe X")
-> rimozione termini ESG generici (non distintivi di nessun pilastro)
-> NORMALIZZAZIONE, in una delle 4 varianti (parametro ``normalization``):

- ``"lemma"`` (default): lemmatizzazione POS-aware (emissions -> emission,
  reducing -> reduce) -- la forma canonica del dizionario, piu' lenta ma
  linguisticamente corretta (richiede il POS tagging).
- ``"porter"``: stemming con l'algoritmo di Porter (il piu' vecchio e
  aggressivo -- taglia i suffissi con regole euristiche, es.
  "emissions"/"emitting" -> "emiss"/"emit", forme non sempre parole vere).
- ``"snowball"``: stemming con l'algoritmo Snowball (successore di Porter,
  regole piu' raffinate, generalmente preferito a Porter oggi).
- ``"none"``: nessuna normalizzazione, i token restano nella forma originale
  (solo lowercase) -- utile come termine di paragone per misurare quanto la
  normalizzazione incida davvero sui risultati finali.

Richiede i corpora nltk: punkt, averaged_perceptron_tagger, stopwords,
wordnet, omw-1.4 (scaricabili con :func:`ensure_nltk_data`).
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal

from .config import Config

_NLTK_PACKAGES = [
    "punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    "stopwords", "wordnet", "omw-1.4",
]

NormalizationMode = Literal["lemma", "porter", "snowball", "none"]


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
    >>> tok_stem = Tokenizer.from_companies(["Amazon"], normalization="snowball")
    >>> tok_stem("Amazon reduced its emissions significantly.")
    ['reduc', 'emiss', 'signific']
    """

    def __init__(self, company_tokens: Iterable[str] = (),
                generic_terms: Iterable[str] = Config().generic_esg_terms,
                normalization: NormalizationMode = "lemma"):
        ensure_nltk_data()
        from nltk.corpus import stopwords as nltk_stopwords

        self.normalization = normalization
        self._stopwords = set(nltk_stopwords.words("english"))
        self._generic_terms = set(generic_terms)
        self._company_tokens = set(company_tokens)

        if normalization == "lemma":
            from nltk import pos_tag
            from nltk.corpus import wordnet
            from nltk.stem import WordNetLemmatizer
            self._pos_tag = pos_tag
            self._wordnet = wordnet
            self._lemmatizer = WordNetLemmatizer()
            self._wordnet_pos = {
                "J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV,
            }
        elif normalization == "porter":
            from nltk.stem import PorterStemmer
            self._stemmer = PorterStemmer()
        elif normalization == "snowball":
            from nltk.stem import SnowballStemmer
            self._stemmer = SnowballStemmer("english")
        elif normalization != "none":
            raise ValueError(
                f"normalization={normalization!r} non riconosciuto. "
                "Usa 'lemma', 'porter', 'snowball' o 'none'."
            )

    @classmethod
    def from_companies(cls, companies: Iterable[str], **kwargs) -> "Tokenizer":
        """Costruisce il tokenizzatore estraendo i token dai nomi azienda del corpus."""
        company_tokens: set[str] = set()
        for c in companies:
            company_tokens.update(re.findall(r"[a-zA-Z]{2,}", str(c).lower()))
        return cls(company_tokens=company_tokens, **kwargs)

    def _lemma(self, tok: str, tag: str) -> str:
        return self._lemmatizer.lemmatize(tok, self._wordnet_pos.get(tag[0], self._wordnet.NOUN))

    def _normalize(self, toks: list[str]) -> list[str]:
        if self.normalization == "lemma":
            return [self._lemma(t, tag) for t, tag in self._pos_tag(toks)]
        if self.normalization in ("porter", "snowball"):
            return [self._stemmer.stem(t) for t in toks]
        return toks  # "none"

    def __call__(self, text: str) -> list[str]:
        toks = re.findall(r"[a-zA-Z]{2,}", str(text).lower())
        toks = [t for t in toks
               if t not in self._stopwords
               and t not in self._generic_terms
               and t not in self._company_tokens]
        if not toks:
            return []
        return self._normalize(toks)

    @property
    def n_stopwords(self) -> int:
        return len(self._stopwords)

    @property
    def n_company_tokens(self) -> int:
        return len(self._company_tokens)
