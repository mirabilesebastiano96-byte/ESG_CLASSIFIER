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
   lunghezza/lingua.
3. **Etichettatura debole** (`esgpillar.labeling`) — assegna ogni frase a uno
   dei 10 topic ufficiali **ESRS** (E1-E5, S1-S4, G1; Regolamento UE 2023/2772)
   via dizionario lessicale, aggregati poi nei tre pilastri E/S/G.
4. **Tokenizzazione condivisa** (`esgpillar.tokenization`) — lemmatizzazione
   POS-aware, rimozione stopword/nomi azienda/termini ESG generici.
5. **Embedding pre-addestrati** (`esgpillar.embeddings`) — tre opzioni con la
   stessa interfaccia di encoding: **GloVe** e **fastText** (statiche, via
   `gensim.downloader`) ed **ESGBERT/FinBERT-ESG** (contestuale, frozen, via
   `transformers` — più lenta ma potenzialmente più espressiva: il vettore di
   ogni parola dipende dalla frase in cui compare).
6. **Reti neurali** (`esgpillar.models`, `esgpillar.training`) — DNN, Conv1D,
   BiLSTM, con early stopping, gradient clipping e scheduler del learning rate.
7. **API di alto livello** (`esgpillar.ESGPillarClassifier`) — `fit` /
   `predict` / `save` / `load` in poche righe.
8. **Tool web** (`esgpillar.tool`) — app Gradio: carica un PDF, ottieni la
   classificazione di ogni frase.
9. **Grafici** (`esgpillar.plotting`) — curve di training (loss + accuracy/F1),
   confronto tra rappresentazioni/reti, confusion matrix, curve ROC,
   distribuzione delle classi, **word cloud per pilastro** (parole distintive
   via lift, non semplicemente frequenti — con esclusione dei nomi azienda se
   passi un tokenizzatore). `ESGPillarClassifier` espone anche
   `.plot_history()` e `.plot_confusion_matrix(...)` come scorciatoie.

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
from esgpillar.plotting import plot_comparison, plot_confusion_matrix, plot_wordcloud_by_pillar
from esgpillar.tokenization import Tokenizer

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
