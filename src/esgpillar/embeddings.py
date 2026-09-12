"""Rappresentazioni testuali pre-addestrate e congelate: GloVe, fastText
(statiche) ed ESGBERT/FinBERT-ESG (contestuale).

Le rappresentazioni statiche danno alla stessa parola sempre lo stesso
vettore; ESGBERT da' un vettore diverso a seconda della frase in cui la
parola compare — piu' costoso (un forward pass del transformer per
frase) ma potenzialmente piu' espressivo. ``pretrained_encode`` resta
la funzione di encoding condivisa per le rappresentazioni statiche;
:class:`ContextualEmbeddingEncoder` gestisce quella contestuale con la
STESSA interfaccia in uscita (seq, pooled, lengths), cosi' il resto
della libreria (reti, training) non deve sapere quale delle due sta
usando.

Richiede l'extra ``embeddings`` (GloVe/fastText) o ``transformers`` (ESGBERT).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .config import Config
from .tokenization import Tokenizer


def pretrained_encode(token_lists: Sequence[list[str]], kv, dim: int,
                      max_len: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encoding condiviso per embedding pre-addestrati congelati.

    Ritorna ``(seq, pooled, lengths)``:
    - ``seq``: (N, max_len, dim), sequenza di vettori per token (per Conv1D/BiLSTM);
    - ``pooled``: (N, dim), media dei vettori di token (per DNN);
    - ``lengths``: (N,), numero di token non-padding per frase.
    """
    n = len(token_lists)
    seq = np.zeros((n, max_len, dim), dtype="float32")
    pooled = np.zeros((n, dim), dtype="float32")
    lengths = np.ones(n, dtype="int64")
    for i, toks in enumerate(token_lists):
        vecs = [kv[t] for t in toks if t in kv.key_to_index]
        if vecs:
            arr = np.asarray(vecs, dtype="float32")
            length = min(len(arr), max_len)
            seq[i, :length] = arr[:length]
            pooled[i] = arr.mean(0)
            lengths[i] = length
    return seq, pooled, lengths


def load_glove(config: Config | None = None):
    """Carica GloVe pre-addestrato (``config.glove_name``) via gensim.downloader.

    La prima esecuzione scarica il modello (centinaia di MB); le successive
    usano la cache locale di gensim.
    """
    import gensim.downloader as gensim_api

    config = config or Config()
    return gensim_api.load(config.glove_name)


def load_fasttext_pretrained(config: Config | None = None):
    """Carica fastText pre-addestrato (``config.fasttext_pretrained_name``).

    Nota: questa variante, scaricata via ``gensim.downloader``, espone solo
    i vettori gia' nel vocabolario — non il modello sub-word completo. Una
    parola fuori vocabolario e' trattata come OOV, senza il vantaggio
    caratteristico di fastText (vettori dai n-grammi).
    """
    import gensim.downloader as gensim_api

    config = config or Config()
    return gensim_api.load(config.fasttext_pretrained_name)


class EmbeddingEncoder:
    """Incapsula un embedding pre-addestrato + tokenizzatore in un'unica funzione di encoding.

    Esempio
    -------
    >>> tok = Tokenizer.from_companies(["Enel", "Amazon"])
    >>> kv = load_glove()
    >>> enc = EmbeddingEncoder(kv=kv, dim=300, tokenizer=tok)
    >>> seq, pooled, lengths = enc(["We reduce our carbon emissions."])
    """

    def __init__(self, kv, dim: int, tokenizer: Tokenizer, max_len: int = 64):
        self.kv = kv
        self.dim = dim
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, sentences: Sequence[str]):
        token_lists = [self.tokenizer(s) for s in sentences]
        return pretrained_encode(token_lists, self.kv, self.dim, self.max_len)


def load_esgbert(config: Config | None = None):
    """Carica ESGBERT/FinBERT-ESG (frozen, nessun fine-tuning) da HuggingFace.

    Ritorna ``(tokenizer, model)``. Richiede l'extra ``transformers``.
    """
    from transformers import AutoModel, AutoTokenizer

    config = config or Config()
    tokenizer = AutoTokenizer.from_pretrained(config.esgbert_model)
    model = AutoModel.from_pretrained(config.esgbert_model)
    return tokenizer, model


class ContextualEmbeddingEncoder:
    """Encoder per rappresentazioni CONTESTUALI frozen (es. ESGBERT/FinBERT-ESG).

    A differenza di :class:`EmbeddingEncoder` (statico, un vettore fisso
    per parola), qui il vettore di ogni token dipende dal CONTESTO della
    frase: serve un forward pass del transformer per ogni batch di frasi
    (piu' lento delle rappresentazioni statiche, specialmente su CPU).
    Stessa interfaccia in uscita: ``seq`` (per Conv1D/BiLSTM), ``pooled``
    (media pesata sull'attention mask, per la DNN), ``lengths``.

    Esempio
    -------
    >>> tok, model = load_esgbert()
    >>> enc = ContextualEmbeddingEncoder(tokenizer=tok, model=model)
    >>> seq, pooled, lengths = enc(["The board approved a new audit policy."])
    """

    def __init__(self, tokenizer, model, dim: int | None = None, max_len: int = 64,
                batch_size: int = 32, device: str = "cpu"):
        self.tokenizer = tokenizer
        self.device = device
        self.model = model.to(device).eval()
        self.dim = dim or model.config.hidden_size
        self.max_len = max_len
        self.batch_size = batch_size

    def __call__(self, sentences: Sequence[str]):
        import torch

        n = len(sentences)
        seq = np.zeros((n, self.max_len, self.dim), dtype="float32")
        pooled = np.zeros((n, self.dim), dtype="float32")
        lengths = np.ones(n, dtype="int64")
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                batch = list(sentences[start:start + self.batch_size])
                enc = self.tokenizer(batch, truncation=True, max_length=self.max_len,
                                     padding=True, return_tensors="pt").to(self.device)
                out = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"]
                out_np = out.cpu().numpy()
                mask_np = mask.cpu().numpy()
                for i in range(len(batch)):
                    length = min(int(mask_np[i].sum()), self.max_len)
                    seq[start + i, :length] = out_np[i, :length]
                    pooled[start + i] = (out_np[i] * mask_np[i][:, None]).sum(0) / max(mask_np[i].sum(), 1)
                    lengths[start + i] = max(length, 1)
        return seq, pooled, lengths
