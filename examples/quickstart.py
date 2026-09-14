"""Esempio rapido: dall'estrazione dai PDF al classificatore addestrato.

Esegui con:  python examples/quickstart.py
(richiede: pip install "esg-pillar-classifier[extraction,embeddings,torch]")
"""
from esgpillar import Config, ESGPillarClassifier
from esgpillar.cleaning import clean_dataframe
from esgpillar.extraction import extract_directory
from esgpillar.labeling import label_dataframe

# --- 1) configurazione: personalizza qui i parametri che ti servono ---
config = Config(pdf_dir="./reports", val_size=0.15)

# --- 2) estrazione dai PDF (con cache: la prima volta estrae, poi ricarica) ---
df = extract_directory(config=config)
print(f"Frasi estratte: {len(df)} | aziende: {df['company'].nunique()}")

# --- 3) pulizia del testo ---
df = clean_dataframe(df, config)

# --- 4) etichettatura debole (topic ESRS -> pilastro E/S/G) ---
df = label_dataframe(df, config)
print(f"Frasi etichettate: {len(df)} | distribuzione: {df['esg'].value_counts().to_dict()}")

# --- 5) addestramento (embedding GloVe pre-addestrato + BiLSTM) ---
clf = ESGPillarClassifier(config=config, embedding="glove", network="BiLSTM")
clf.fit(df, text_col="sentence", company_col="company", label_col="label")

# --- 6) classificazione di frasi nuove ---
examples = [
    "We reduced our carbon emissions by 30% compared to the previous year.",
    "Our board strengthened anti-corruption compliance procedures this year.",
    "Employee training programs on workplace safety were expanded significantly.",
]
for sentence, label in zip(examples, clf.predict(examples)):
    print(f"[{label:14s}] {sentence}")

# --- 7) salva per riusarlo senza ri-addestrare ---
clf.save("modello_esg")
print("\nModello salvato in ./modello_esg/ — ricaricalo con ESGPillarClassifier.load('modello_esg')")
