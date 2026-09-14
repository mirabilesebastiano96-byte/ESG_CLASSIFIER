# esg-pillar-classifier

Estrazione, etichettatura debole (topic ESRS) e classificazione di frasi ESG
nei pilastri **Environmental / Social / Governance**.

Nasce dal lavoro di tesi magistrale su *"Analisi della reportistica ESG mediante
NLP e Deep Learning"* (Statistica e Data Science, Univ. di Palermo), impacchettato
come libreria riusabile.

## Cosa fa

1. **Estrazione** (`esgpillar.extraction`) — legge **PDF, Word (.docx) e file
   di testo (.txt)**: per il PDF, un parser *column-aware* che rileva colonne,
   filtra per dimensione del font, rimuove boilerplate ripetuto tra pagine e
   ricostruisce frasi di prosa pulite (non frammenti di tabella); per Word e
   testo, un'estrazione più diretta con lo stesso filtro di qualità finale.
   `extract_directory()` accetta una cartella con un mix di questi formati.
2. **Pulizia** (`esgpillar.cleaning`) — de-sillabazione, deduplica, filtro di
   lunghezza/lingua (pipeline di default). Più undici tecniche ADDIZIONALI
   (extra `cleaning`), da applicare esplicitamente quando servono:
   `expand_contractions` (don't -> do not), `correct_spelling` (dizionario
   inglese generico — attenzione ai falsi positivi su termini tecnici),
   `normalize_numbers` (sostituzione con placeholder fisso),
   `expand_corporate_abbreviations` (Corp. -> Corporation, con possibilità
   di aggiungere abbreviazioni personalizzate), `segment_sentences`/
   `split_into_sentences` (spezza paragrafi in frasi singole, gestendo
   correttamente abbreviazioni come "Dr."), `remove_accents` (café ->
   cafe — una perdita di informazione reale su alcune lingue, non solo
   cosmetica), `normalize_currency` ($10 -> USD 10, anche per le forme a
   parola come "10 dollars"), `detect_boilerplate`/
   `remove_boilerplate_column` (intestazioni/piè di pagina ripetuti
   IDENTICI tra più documenti — richiede una soglia sia percentuale sia
   assoluta di documenti, altrimenti con pochi documenti diventa
   degenere: con 2 soli documenti, "compare in almeno 1" è già
   matematicamente il 50%), `fix_encoding` (ripara mojibake, es.
   "sociÃ©tÃ©" -> "société" — diverso da `remove_accents`, che invece
   rimuove intenzionalmente gli accenti da un testo già leggibile),
   `find_near_duplicates`/`remove_near_duplicates` (frasi QUASI
   identiche, non solo identiche in senso stretto — a differenza della
   deduplica esatta della pipeline di default), `is_tabular_fragment`/
   `remove_tabular_fragments` (righe probabilmente frammenti di tabella
   estratti male, non frasi vere).
3. **Etichettatura** (`esgpillar.labeling`) — il metodo PREDEFINITO assegna
   ogni frase a uno dei 10 topic ufficiali **ESRS** (E1-E5, S1-S4, G1;
   Regolamento UE 2023/2772) via dizionario lessicale, aggregati poi nei
   tre pilastri E/S/G. Cinque **tecniche alternative** (a livello di solo
   pilastro, non dei 10 topic): `label_dataframe_tfidf` (conteggio pesato
   per informatività nel corpus, invece che grezzo), `label_dataframe_prototype`
   (similarità semantica a un prototipo per pilastro, via un embedding a
   scelta — cattura frasi senza nessuna parola esatta del dizionario),
   `label_dataframe_zero_shot` (un modello di NLI, nessun dizionario né
   training), `label_dataframe_esgbert` (la testa di classificazione
   *originale* di ESGBERT — un vero modello fine-tunato su 2.000 frasi
   annotate a mano, non supervisione debole), `label_dataframe_ensemble`
   (combina più etichettature con voto di maggioranza, stile Snorkel
   semplificato — nessuna delle cinque sostituisce il metodo predefinito,
   sono alternative esplicite da scegliere consapevolmente).
4. **Tokenizzazione condivisa** (`esgpillar.tokenization`) — rimozione
   stopword/nomi azienda/termini ESG generici, poi normalizzazione a
   scelta (`normalization=`): **lemmatizzazione** POS-aware (default),
   **stemming** (Porter o Snowball), o nessuna normalizzazione (`"none"`).
5. **Embedding** (`esgpillar.embeddings`) — quattordici opzioni, tutte con
   la stessa interfaccia di encoding in uscita. Statiche pre-addestrate su
   un corpus esterno (via `gensim.downloader`): **GloVe**, **fastText**.
   Statiche addestrate sul TUO corpus di training: **Word2Vec** Skip-gram,
   **fastText corpus** (con vere subword — gestisce parole mai viste, a
   differenza della variante pre-addestrata), **LSA** (TF-IDF + SVD per
   parola, statistica non neurale). Contestuali frozen, un vettore diverso
   a seconda della frase (via `transformers`): **ESGBERT**, **FinBERT**,
   **ClimateBERT**, **BERT-base** (generico), **RoBERTa** (generico, con
   pre-training piu' robusto), **UmBERTo** (RoBERTa-based ma addestrato su
   corpus ITALIANI — usarlo solo se il tuo corpus e' in italiano, non
   inglese). Contestuale ma "fuori famiglia": **ELMo** (via l'extra `elmo`
   — installa **TensorFlow** oltre a PyTorch, un costo di dipendenza reale;
   il modello va anche scaricato **a mano**, non automaticamente — vedi la
   docstring di `load_elmo` per le istruzioni). A livello di FRASE intera
   (non di parola): **Doc2Vec** (addestrato sul corpus) e
   **Sentence-Transformers/SBERT** (pre-addestrato apposta per produrre
   vettori di frase, via l'extra `sbert`).
6. **Reti neurali** (`esgpillar.models`, `esgpillar.training`) — undici
   architetture di famiglie diverse: **DNN** (feed-forward), **Conv1D**/
   **ResCNN**/**DPCNN** (convoluzionali — la seconda con blocchi residui in
   stile ResNet, la terza una piramide profonda che dimezza la sequenza a
   ogni livello, Johnson & Zhang 2017), **BiLSTM**/**BiGRU**/**GRU**
   (ricorrenti, l'ultima mono-direzionale), **RCNN** (ibrido
   ricorrente+convoluzionale, Lai et al. 2015), **Transformer** encoder e
   **AttentionPool** (attenzione), **MLPMixer** (niente attenzione né
   ricorrenza né convoluzione — solo MLP alternati che mescolano token e
   feature). Tutte con early stopping, gradient clipping e scheduler del
   learning rate.
7. **Modelli classici** (`esgpillar.classical`) — 24 classificatori
   scikit-learn/XGBoost/LightGBM, come alternativa alle reti neurali (usa
   lo stesso parametro `network=` di `ESGPillarClassifier`, lavorano solo
   su vettori pooled). **Lineari**: Logistic Regression, Linear SVM, SVM a
   kernel RBF, SGD, Ridge, Perceptron. **Naive Bayes**: Gaussiano,
   Multinomiale, Complement, Bernoulli (le ultime tre richiedono feature
   non negative). **Alberi/ensemble**: Decision Tree, Random Forest, Extra
   Trees, Gradient Boosting, AdaBoost, Bagging, XGBoost, LightGBM. **Altri**:
   KNN, LDA, QDA, MLP di scikit-learn. **Meta-ensemble**: Voting, Stacking.
8. **Analisi linguistica** (`esgpillar.text_analysis`) — strumenti di
   analisi descrittiva per-frase, separati dalla pipeline di
   classificazione: **POS tagging** (nltk), **NER** (nltk o, più accurato,
   spaCy via l'extra `nlp`), **sentiment** in tre varianti — **VADER** e
   **TextBlob** (lessici di polarità, veloci, ma con limiti noti su testo
   di dominio specifico — vedi la docstring di `sentiment_vader`) e
   **FinBERT** (modello fine-tunato per il sentiment finanziario, extra
   `transformers`). Funzioni `add_pos_column`/`add_entities_column`/
   `add_sentiment_column` per applicarle direttamente a un DataFrame.
9. **Text mining** (`esgpillar.text_mining`) — tecniche a livello di
   documento/corpus (extra `mining` per RAKE/TextRank/leggibilità).
   **Estrazione keyword/keyphrase**: TF-IDF, RAKE (frasi chiave, non
   singole parole), TextRank (grafo di co-occorrenza + PageRank). **Topic
   modeling**: LDA e NMF (scikit-learn, nessun extra). **Riassunto
   estrattivo**: TextRank a livello di frase — sceglie le frasi più
   centrali di un testo lungo, restituite nell'ordine originale.
   **Similarità/paraphrase** tra frasi: via TF-IDF (lessicale, leggero) o
   via un qualunque encoder di `esgpillar.embeddings` (semantico — SBERT è
   pensato apposta per questo). **Leggibilità**: Flesch Reading Ease,
   Flesch-Kincaid Grade, Gunning Fog — utili anche per capire se un report
   usa un linguaggio inutilmente complesso (rilevante in analisi di
   greenwashing, non solo di stile). **Claim quantitativi**: estrae
   percentuali/importi/quantità con il loro contesto (es. "ridotto del
   25%"). **N-grammi/collocazioni**: bigrammi/trigrammi per frequenza o
   per PMI (nltk, nessun extra). **Classificazione zero-shot**: etichette
   a scelta libera senza addestrare nulla (via un modello di NLI, extra
   `transformers`), utile per esplorare categorie nuove prima di
   costruire un classificatore vero (`find_similar_sentences` cerca le
   frasi più simili a una query in un corpus).
10. **API di alto livello** (`esgpillar.ESGPillarClassifier`) — `fit` /
   `predict` / `save` / `load` in poche righe.
11. **Tool web** (`esgpillar.tool`) — app Gradio: carica un PDF, ottieni la
   classificazione di ogni frase.
12. **Grafici** (`esgpillar.plotting`) — curve di training (loss + accuracy/F1),
   confronto tra rappresentazioni/reti, confusion matrix, curve ROC,
   distribuzione delle classi, **word cloud per pilastro** (parole distintive
   via lift, non semplicemente frequenti — con esclusione dei nomi azienda se
   passi un tokenizzatore), **proiezione 2D degli embedding** (PCA o t-SNE,
   colorata per pilastro — utile per capire *perché* una rappresentazione
   separa le classi meglio di un'altra, non solo di quanto). Più EDA
   aggiuntiva: `text_statistics` (vocabolario, diversità lessicale),
   `plot_top_ngrams` (bigrammi/trigrammi più frequenti o distintivi, extra
   `mining`), `plot_zipf` (distribuzione di frequenza delle parole),
   `plot_feature_correlation_by_pillar` (correlazione tra feature
   numeriche — es. leggibilità, sentiment — **separata per pilastro**,
   per non nascondere pattern diversi tra E/S/G in un'unica correlazione
   sull'intero corpus). `ESGPillarClassifier` espone anche
   `.plot_history()`, `.plot_confusion_matrix(...)` e
   `.plot_embeddings(...)` come scorciatoie.

## Installazione

```bash
# solo etichettatura/pulizia (dipendenze minime)
pip install esg-pillar-classifier

# con supporto PDF + embedding + reti (il caso d'uso tipico)
pip install "esg-pillar-classifier[extraction,embeddings,torch]"

# con ESGBERT (contestuale) al posto di / oltre a GloVe/fastText
pip install "esg-pillar-classifier[extraction,transformers]"

# tutto, incluso il tool Gradio e la visualizzazione
pip install "esg-pillar-classifier[all]"
```

Finché non è pubblicato su PyPI, installa direttamente da GitHub:

```bash
pip install "esg-pillar-classifier[all] @ git+https://github.com/YOUR-USERNAME/esg-pillar-classifier.git"
```

oppure in locale, per sviluppo:

```bash
git clone https://github.com/YOUR-USERNAME/esg-pillar-classifier.git
cd esg-pillar-classifier
pip install -e ".[all,dev]"
```

## Esempio rapido

```python
import pandas as pd
from esgpillar import Config, ESGPillarClassifier
from esgpillar.labeling import label_dataframe

# 1) un DataFrame con colonne "company" e "sentence"
df = pd.read_csv("dataset.csv")

# 2) etichettatura debole (topic ESRS -> pilastro E/S/G)
df = label_dataframe(df)

# 3) addestra il classificatore
clf = ESGPillarClassifier(embedding="glove", network="BiLSTM")
clf.fit(df, text_col="sentence", company_col="company", label_col="label")

# 4) classifica frasi nuove
clf.predict(["We reduced carbon emissions by 30% in 2023."])

# altre combinazioni valide (stessa API, cambiano solo i due parametri):
# clf = ESGPillarClassifier(embedding="word2vec", network="Transformer")   # addestrato sul corpus
# clf = ESGPillarClassifier(embedding="lsa", network="AttentionPool")      # statistico, per parola
# clf = ESGPillarClassifier(embedding="finbert", network="Conv1D")         # contestuale finanziario
# clf = ESGPillarClassifier(embedding="climatebert", network="BiGRU")      # contestuale, specifico clima
# clf = ESGPillarClassifier(embedding="fasttext_corpus", network="ResCNN") # fastText con vere subword
# clf = ESGPillarClassifier(embedding="doc2vec", network="MLPMixer")       # un vettore per frase
# clf = ESGPillarClassifier(embedding="sbert", network="GRU")              # pre-addestrato per frasi
# clf = ESGPillarClassifier(embedding="lsa", network="RandomForest")       # modello classico, non rete neurale
# clf = ESGPillarClassifier(embedding="word2vec", network="XGBoost")       # richiede l'extra "classical"
# ['Environmental']

# in alternativa, ESGBERT/FinBERT-ESG (contestuale, frozen):
# clf = ESGPillarClassifier(embedding="esgbert", network="BiLSTM")

# 5) salva / ricarica
clf.save("modello_esg")
clf2 = ESGPillarClassifier.load("modello_esg")
```

### Estrarre direttamente dai documenti (PDF, Word, testo)

```python
from esgpillar import Config
from esgpillar.extraction import extract_directory
from esgpillar.cleaning import clean_dataframe
from esgpillar.labeling import label_dataframe

config = Config(pdf_dir="./reports")
# ./reports puo' contenere un mix di .pdf, .docx e .txt: ogni file viene
# instradato all'estrattore giusto in base all'estensione
df = extract_directory(config=config)
df = clean_dataframe(df, config)
df = label_dataframe(df, config)
```

### Grafici

```python
from esgpillar.plotting import (
    plot_comparison, plot_confusion_matrix, plot_wordcloud_by_pillar, plot_sentences_per_company,
)
from esgpillar.tokenization import Tokenizer

# EDA: quante frasi per azienda (utile prima dello split per azienda)
fig = plot_sentences_per_company(df)
fig.savefig("frasi_per_azienda.png")

# dopo aver addestrato piu' combinazioni rappresentazione x rete...
fig = plot_comparison(results, metric="test_acc", metric_label="Accuracy")
fig.savefig("confronto.png")

# word cloud per pilastro -- passa un Tokenizer per escludere i nomi azienda
# (altrimenti rischiano di comparire come parole "distintive")
tok = Tokenizer.from_companies(df["company"].unique())
fig = plot_wordcloud_by_pillar(df, tokenizer=tok)
fig.savefig("wordcloud_pilastri.png")

# oppure, scorciatoie dopo clf.fit(...):
clf.plot_history()
clf.plot_confusion_matrix(test_sentences, test_labels)
```

### App web

```python
from esgpillar.tool import launch_tool
launch_tool(clf)   # apre un'app Gradio: carica un PDF, ottieni la classificazione
```

## Note metodologiche (da conoscere prima di usarla in un lavoro accademico)

- **Etichettatura debole, non gold standard**: le etichette derivano da un
  dizionario lessicale (topic ESRS), non da annotazione umana. Il modello
  impara a *imitare* quell'etichettatura automatica.
- **Split per azienda**: `fit()` separa training/validazione **per azienda**
  (mai la stessa azienda in entrambi gli insiemi) per evitare che il modello
  impari a riconoscere lo stile di scrittura di un'azienda specifica invece
  del concetto ESG.
- **Mascheratura anti-leakage attiva per default** (`Config.apply_leakage_mask
  = True`): `label_dataframe()` rimuove dal testo le parole del dizionario
  ESRS che hanno generato l'etichetta (colonna `sentence_model`), altrimenti
  il modello impara a riconoscere la parola-chiave invece del concetto,
  gonfiando le metriche in modo artificiale. `ESGPillarClassifier.fit()`
  rileva automaticamente `sentence_model` e la usa al posto del testo
  grezzo; `predict()` applica la stessa mascheratura alle frasi nuove, per
  coerenza training/inferenza. Disattivabile con `Config(apply_leakage_mask=False)`
  se vuoi confrontare i due scenari (aspettati metriche molto più alte, ma
  meno difendibili, senza la mascheratura).
- **fastText pre-addestrato senza vantaggio OOV**: la versione scaricata via
  `gensim.downloader` espone solo i vettori già nel vocabolario, non il
  modello sub-word completo.

## Sviluppo

```bash
pip install -e ".[all,dev]"
pytest
```

## Pubblicare su PyPI

```bash
python -m build
twine upload dist/*
```

## Licenza

MIT — vedi [LICENSE](LICENSE).
