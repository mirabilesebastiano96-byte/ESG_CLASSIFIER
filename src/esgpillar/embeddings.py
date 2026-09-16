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

    Prova a indicizzare direttamente ogni token (``kv[t]``) invece di
    controllare prima ``t in kv.key_to_index``: per la maggior parte
    delle rappresentazioni (GloVe/Word2Vec/LSA) il risultato e' identico,
    ma per fastText (:func:`load_fasttext_corpus`) fa la differenza --
    quel modello puo' produrre un vettore anche per parole MAI VISTE nel
    training, componendolo dai suoi n-grammi di caratteri (subword). Con
    il controllo preventivo su ``key_to_index`` quel vantaggio andrebbe
    perso: la parola verrebbe scartata come se fosse OOV, anche se
    ``kv[t]`` avrebbe funzionato.

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
        vecs = []
        for t in toks:
            try:
                vecs.append(kv[t])
            except KeyError:
                continue
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


def load_word2vec(token_lists: Sequence[list[str]], config: Config | None = None):
    """Addestra Word2Vec (Skip-gram) DIRETTAMENTE sul corpus fornito.

    A differenza di GloVe/fastText (pre-addestrati su un corpus esterno,
    generico), questo vocabolario e' specifico del dominio ma appreso da
    un corpus tipicamente molto piu' piccolo. Skip-gram (``sg=1``, non
    CBOW) e' scelto perche' piu' accurato di CBOW sulle parole poco
    frequenti — frequenti nei corpora tecnici come quello ESG.

    Ritorna un oggetto ``.wv`` compatibile con :func:`pretrained_encode`
    (stessa interfaccia di ``KeyedVectors``: ``key_to_index``, indicizzazione).
    """
    from gensim.models import Word2Vec

    config = config or Config()
    model = Word2Vec(
        sentences=token_lists,
        vector_size=config.word2vec_dim,
        sg=1,
        window=config.word2vec_window,
        min_count=config.word2vec_min_count,
        workers=1,
        epochs=config.word2vec_epochs,
        seed=config.seed,
    )
    return model.wv


class _LSAKeyedVectors(dict):
    """Adattatore minimale: un dizionario parola -> vettore con l'interfaccia
    di ``gensim.KeyedVectors`` (``key_to_index``) richiesta da
    :func:`pretrained_encode`."""

    @property
    def key_to_index(self):
        return self


def load_lsa(token_lists: Sequence[list[str]], config: Config | None = None):
    """Costruisce una rappresentazione LSA PER PAROLA (non per frase).

    SVD sulla matrice TF-IDF trasposta (parola x documento): da' un
    vettore per ciascuna parola del vocabolario, cosi' Conv1D/BiLSTM
    ricevono sequenze vere con l'ordine delle parole preservato — non
    un unico vettore piatto per frase, che sarebbe l'errore comune di
    un'implementazione LSA ingenua per la classificazione di frasi.

    Nessun addestramento neurale nella costruzione dei vettori: utile
    come termine di paragone statistico, non neurale.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    config = config or Config()
    vec = TfidfVectorizer(
        tokenizer=lambda toks: toks, preprocessor=lambda x: x, token_pattern=None,
        lowercase=False, max_features=config.lsa_max_features, min_df=config.lsa_min_df,
    )
    tfidf = vec.fit_transform(token_lists)
    vocab = vec.get_feature_names_out()
    svd = TruncatedSVD(n_components=config.lsa_dim, random_state=config.seed)
    word_vecs = svd.fit_transform(tfidf.T)
    return _LSAKeyedVectors({w: word_vecs[i] for i, w in enumerate(vocab)})


def load_fasttext_corpus(token_lists: Sequence[list[str]], config: Config | None = None):
    """Addestra fastText DIRETTAMENTE sul corpus fornito (non pre-addestrato).

    A differenza di :func:`load_fasttext_pretrained` (via
    ``gensim.downloader``, che espone solo i vettori gia' nel vocabolario),
    questo e' un vero modello fastText con vettori di sub-word: puo'
    produrre un vettore anche per parole MAI VISTE nel training,
    componendolo dai suoi n-grammi di caratteri -- il vantaggio
    caratteristico di fastText, utile su corpora tecnici con molte varianti
    morfologiche di uno stesso termine (es. "emission"/"emissions"/
    "emitting"). Richiede la correzione in :func:`pretrained_encode` che
    prova l'indicizzazione diretta invece di controllare prima
    ``key_to_index`` (altrimenti l'OOV verrebbe scartato comunque).
    """
    from gensim.models import FastText

    config = config or Config()
    model = FastText(
        sentences=token_lists,
        vector_size=config.fasttext_corpus_dim,
        window=config.fasttext_corpus_window,
        min_count=config.fasttext_corpus_min_count,
        workers=1,
        epochs=config.fasttext_corpus_epochs,
        seed=config.seed,
    )
    return model.wv


def load_doc2vec(sentences: Sequence[str], config: Config | None = None):
    """Addestra Doc2Vec (Paragraph Vectors) sul corpus fornito.

    Paradigma diverso da tutti gli altri embedding di questo modulo: da'
    UN VETTORE PER FRASE INTERA, appreso direttamente (non ottenuto
    mediando vettori di parola). Non esiste quindi una "sequenza di
    token" nel senso usuale -- vedi :func:`doc2vec_encode`, che adatta
    questa rappresentazione all'interfaccia comune (seq, pooled, lengths)
    trattando ogni frase come una sequenza di lunghezza 1.

    Riceve le frasi GIA' TOKENIZZATE dal chiamante (lista di liste di
    parole), non le frasi grezze, per restare coerente con lo stesso
    tokenizzatore condiviso usato dalle altre rappresentazioni.
    """
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument

    config = config or Config()
    tagged = [TaggedDocument(words=toks, tags=[i]) for i, toks in enumerate(sentences)]
    model = Doc2Vec(
        documents=tagged,
        vector_size=config.doc2vec_dim,
        window=config.doc2vec_window,
        min_count=config.doc2vec_min_count,
        workers=1,
        epochs=config.doc2vec_epochs,
        seed=config.seed,
    )
    return model


def doc2vec_encode(token_lists: Sequence[list[str]], model, dim: int,
                   max_len: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adatta Doc2Vec all'interfaccia comune (seq, pooled, lengths).

    Ogni frase e' un SINGOLO vettore (non una sequenza di vettori per
    token): qui viene trattata come una sequenza di lunghezza 1, cosi'
    resta compatibile con reti sequenziali (Conv1D/BiLSTM/...) senza
    bisogno di un branch separato nel resto della libreria -- il kernel
    o la ricorrenza vedranno semplicemente un solo "token" reale seguito
    da padding nullo.
    """
    n = len(token_lists)
    seq = np.zeros((n, max_len, dim), dtype="float32")
    pooled = np.zeros((n, dim), dtype="float32")
    lengths = np.ones(n, dtype="int64")
    for i, toks in enumerate(token_lists):
        vec = model.infer_vector(toks)
        seq[i, 0] = vec
        pooled[i] = vec
    return seq, pooled, lengths


def load_sentence_transformer(config: Config | None = None):
    """Carica un modello Sentence-Transformers (SBERT), frozen.

    Paradigma diverso da ESGBERT/FinBERT/ecc.: quei modelli sono BERT
    generico con pooling aggiunto a posteriori (media sull'attention
    mask); un modello Sentence-Transformers e' invece addestrato ESPRESSAMENTE
    per produrre buoni vettori di frase (tipicamente con un obiettivo
    contrastivo/siamese su coppie di frasi), non solo un side-effect del
    pre-training per token. Da' un vettore per frase, stessa interfaccia
    di Doc2Vec (vedi :func:`sentence_transformer_encode`).

    Richiede il pacchetto ``sentence-transformers``
    (``pip install esg-pillar-classifier[sbert]``).
    """
    from sentence_transformers import SentenceTransformer

    config = config or Config()
    return SentenceTransformer(config.sentence_transformer_model)


def sentence_transformer_encode(sentences: Sequence[str], model, max_len: int = 64,
                                batch_size: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adatta Sentence-Transformers all'interfaccia comune (seq, pooled, lengths).

    Stessa idea di :func:`doc2vec_encode`: un vettore per frase, trattato
    come una sequenza di lunghezza 1 per restare compatibile con le reti
    sequenziali senza bisogno di un branch separato.
    """
    dim = model.get_sentence_embedding_dimension()
    n = len(sentences)
    seq = np.zeros((n, max_len, dim), dtype="float32")
    pooled = np.zeros((n, dim), dtype="float32")
    lengths = np.ones(n, dtype="int64")
    embeddings = model.encode(list(sentences), batch_size=batch_size, show_progress_bar=False)
    for i, vec in enumerate(embeddings):
        seq[i, 0] = vec
        pooled[i] = vec
    return seq, pooled, lengths


class Doc2VecEncoder:
    """Incapsula un modello Doc2Vec addestrato + tokenizzatore, con la
    STESSA interfaccia in uscita (seq, pooled, lengths) delle altre
    rappresentazioni -- cosi' il resto della libreria non deve sapere che
    qui c'e' un solo vettore per frase, non uno per token."""

    def __init__(self, model, dim: int, tokenizer: Tokenizer, max_len: int = 64):
        self.model = model
        self.dim = dim
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, sentences: Sequence[str]):
        token_lists = [self.tokenizer(s) for s in sentences]
        return doc2vec_encode(token_lists, self.model, self.dim, self.max_len)


class SentenceTransformerEncoder:
    """Incapsula un modello Sentence-Transformers (SBERT) frozen, con la
    stessa interfaccia in uscita delle altre rappresentazioni. A differenza
    di :class:`Doc2VecEncoder`, non serve un tokenizzatore condiviso: SBERT
    tokenizza internamente le frasi grezze."""

    def __init__(self, model, max_len: int = 64, batch_size: int = 32):
        self.model = model
        self.dim = model.get_sentence_embedding_dimension()
        self.max_len = max_len
        self.batch_size = batch_size

    def __call__(self, sentences: Sequence[str]):
        return sentence_transformer_encode(sentences, self.model, self.max_len, self.batch_size)


def load_elmo(config: Config | None = None):
    """Carica un modello ELMo pre-addestrato (frozen) con ``simple_elmo``.

    A DIFFERENZA di tutte le altre rappresentazioni di questo modulo, ELMo
    non si scarica automaticamente da un nome/id remoto: va scaricato A
    MANO come archivio ZIP dal repository NLPL
    (http://vectors.nlpl.eu/repository/) ed estratto (o lasciato zippato)
    in una cartella locale, indicata in ``config.elmo_model_path``. Se il
    percorso non esiste, viene sollevato un errore esplicito con le
    istruzioni, non un fallimento silenzioso.

    Richiede l'extra ``elmo`` (``pip install esg-pillar-classifier[elmo]``),
    che installa TENSORFLOW -- un SECONDO framework di deep learning
    oltre a PyTorch, usato per tutte le altre reti di questa libreria.
    E' un costo di dipendenza reale, non solo teorico: prima di scegliere
    ELMo, vale la pena chiedersi se una rappresentazione contestuale gia'
    disponibile (ESGBERT, FinBERT, RoBERTa, ...) non risolva lo stesso
    bisogno senza aggiungere TensorFlow al progetto.

    ``simple_elmo`` e' la scelta qui invece di ``allennlp`` (l'
    implementazione ELMo storicamente piu' diffusa): quest'ultima non si
    installa piu' su Python/pip moderni (dipendenze di build obsolete),
    mentre ``simple_elmo`` e' attivamente mantenuta e compatibile con
    TensorFlow 2.
    """
    import os

    config = config or Config()
    if not os.path.exists(config.elmo_model_path):
        raise FileNotFoundError(
            f"Modello ELMo non trovato in {config.elmo_model_path!r}. A differenza delle "
            "altre rappresentazioni, ELMo non si scarica automaticamente: scarica un "
            "archivio ZIP da http://vectors.nlpl.eu/repository/ (o un altro modello ELMo "
            "compatibile con simple_elmo), estrailo (o lascialo zippato) in una cartella, "
            "e imposta Config(elmo_model_path=\"quel/percorso\")."
        )
    # il controllo del percorso viene PRIMA dell'import di simple_elmo (che a sua volta
    # importa TensorFlow): un utente senza il modello scaricato ottiene un errore chiaro
    # e immediato, senza dover aspettare/incappare in un import pesante che non serve ancora
    from simple_elmo import ElmoModel

    model = ElmoModel()
    model.load(config.elmo_model_path)
    return model


class ElmoEncoder:
    """Incapsula un modello ELMo caricato + tokenizzatore, con la stessa
    interfaccia in uscita (seq, pooled, lengths) delle altre rappresentazioni.

    A differenza di :class:`ContextualEmbeddingEncoder` (BERT-style, con
    tokenizzazione WordPiece propria), ELMo lavora su parole GIA' divise
    dal tokenizzatore condiviso -- stesso schema delle rappresentazioni
    statiche (GloVe/Word2Vec), ma con vettori dipendenti dal contesto.
    """

    def __init__(self, model, tokenizer: Tokenizer, dim: int = 1024, max_len: int = 64):
        self.model = model
        self.tokenizer = tokenizer
        self.dim = dim
        self.max_len = max_len

    def __call__(self, sentences: Sequence[str]):
        token_lists = [self.tokenizer(s) or [""] for s in sentences]  # ElmoModel non ama liste vuote
        raw_seq = self.model.get_elmo_vectors(token_lists)  # (N, L_batch, dim), L_batch = frase piu' lunga nel batch
        pooled = self.model.get_elmo_vector_average(token_lists)  # (N, dim)

        n = len(sentences)
        seq = np.zeros((n, self.max_len, self.dim), dtype="float32")
        lengths = np.ones(n, dtype="int64")
        for i, toks in enumerate(token_lists):
            length = min(len(toks), self.max_len, raw_seq.shape[1])
            seq[i, :length] = raw_seq[i, :length]
            lengths[i] = max(length, 1)
        return seq, pooled.astype("float32"), lengths


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


def load_contextual_model(model_name: str):
    """Carica un modello Transformer HuggingFace generico (frozen) come
    coppia ``(tokenizer, model)``, utilizzabile con
    :class:`ContextualEmbeddingEncoder`. Funzione di base per tutte le
    varianti contestuali sotto (ESGBERT, FinBERT, ClimateBERT, BERT
    generico) — cambia solo il nome del modello.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def load_roberta(config: Config | None = None):
    """Carica RoBERTa-base (generico, nessuna specializzazione di dominio
    ne' di lingua) -- pre-addestramento robusto su un corpus piu' ampio
    e con scelte di ottimizzazione migliorate rispetto al BERT originale."""
    config = config or Config()
    return load_contextual_model(config.roberta_model)


def load_umberto(config: Config | None = None):
    """Carica UmBERTo (RoBERTa-based, addestrato su grandi corpora ITALIANI
    con SentencePiece e Whole Word Masking).

    ATTENZIONE: e' un modello di lingua ITALIANA -- il suo vocabolario e la
    sua tokenizzazione sono adatti a testo in italiano, non in inglese.
    Su un corpus inglese (come quello ESG usato nel resto di questa
    libreria) le prestazioni ne risentirebbero pesantemente: ha senso
    solo se il TUO corpus e' (anche) in italiano.
    """
    config = config or Config()
    return load_contextual_model(config.umberto_model)


def load_finbert(config: Config | None = None):
    """Carica FinBERT (finanziario generico, NON ESG-specifico) — via di
    mezzo tra un modello generico e uno specializzato ESG come ESGBERT."""
    config = config or Config()
    return load_contextual_model(config.finbert_model)


def load_climatebert(config: Config | None = None):
    """Carica ClimateBERT (specifico clima, un sottoinsieme di ESG —
    non copre Social/Governance quanto ESGBERT).

    Nota: il language model base di ClimateBERT e' stato pre-addestrato
    prevalentemente su PARAGRAFI, non frasi singole — su corpora a
    granularita' di frase (come questo) le prestazioni potrebbero
    risentirne rispetto a ESGBERT, addestrato su testo piu' simile."""
    config = config or Config()
    return load_contextual_model(config.climatebert_model)


def load_bert_base(config: Config | None = None):
    """Carica BERT-base generico, senza alcuna specializzazione di dominio —
    il termine di paragone per isolare l'effetto del pre-training di dominio
    (ESG/finanziario/clima) rispetto a un Transformer generico."""
    config = config or Config()
    return load_contextual_model(config.bert_base_model)


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
