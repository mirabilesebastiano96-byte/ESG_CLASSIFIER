"""Script di prova per la libreria esg-pillar-classifier.

Pensato per essere VELOCE (GloVe nella variante piu' piccola, 50
dimensioni invece di 300 — pochi minuti anche su un portatile), e
AUTOSUFFICIENTE: usa un mini-dataset scritto a mano, non servono PDF
veri ne' l'Excel a 20 aziende.

Uso:
    pip install "esg-pillar-classifier[embeddings,torch,viz] @ git+https://github.com/TUO-USERNAME/esg-pillar-classifier.git"
    python test_library.py

Se preferisci installarla in locale invece che da GitHub (es. l'hai
scaricata come cartella):
    pip install -e "./esg-pillar-classifier[embeddings,torch,viz]"
"""
import sys


def step(msg):
    print(f"\n{'=' * 60}\n{msg}\n{'=' * 60}")


# --------------------------------------------------------------------- #
step("1. Verifica import")
# --------------------------------------------------------------------- #
try:
    import pandas as pd

    from esgpillar import Config, ESGPillarClassifier
    from esgpillar.labeling import label_dataframe, label_sentence
    from esgpillar.plotting import plot_class_distribution, plot_comparison
except ImportError as e:
    print(f"[ERRORE] Import fallito: {e}")
    print("Hai installato la libreria con gli extra giusti? Prova:")
    print('  pip install "esg-pillar-classifier[embeddings,torch,viz]"')
    sys.exit(1)
print("Import OK.")


# --------------------------------------------------------------------- #
step("2. Etichettatura debole (nessun download necessario)")
# --------------------------------------------------------------------- #
# un mini-dataset scritto a mano, 4 aziende finte, poche frasi ciascuna,
# scelte per attivare chiaramente ciascun pilastro ESG
raw_sentences = {
    "AlphaCorp": [
        "We reduced our carbon emissions by 25 percent through renewable energy investments.",
        "Water consumption across our facilities decreased significantly this year.",
        "Employee training programs on workplace safety were expanded to all regional offices.",
        "The board of directors approved a new anti-corruption compliance policy.",
        "Our audit committee reviewed internal governance procedures during the quarter.",
    ],
    "BetaGroup": [
        "The company invested heavily in circular economy initiatives to reduce waste.",
        "Diversity and inclusion programs improved representation across all departments.",
        "Compliance with anti-bribery regulations was strengthened through new internal controls.",
        "Greenhouse gas emissions from our supply chain were measured for the first time.",
        "Worker health and safety incidents declined thanks to expanded training efforts.",
    ],
    "GammaLtd": [
        "Biodiversity protection measures were implemented near our manufacturing sites.",
        "The governance structure was revised to increase board independence.",
        "Community engagement programs supported local employment initiatives.",
        "Energy efficiency upgrades reduced our overall climate impact this year.",
        "Consumer data privacy protections were enhanced across all digital products.",
    ],
    "DeltaInc": [
        "Recycling rates improved after new circular economy programs were launched.",
        "The audit committee strengthened oversight of corporate governance practices.",
        "Employee wellbeing initiatives expanded access to mental health support services.",
        "Water withdrawal from local sources was reduced through efficiency measures.",
        "Board diversity increased following new director appointments this year.",
    ],
}
rows = [
    {"company": company, "sentence": s}
    for company, sentences in raw_sentences.items()
    for s in sentences
]
df_raw = pd.DataFrame(rows)
print(f"Dataset grezzo: {len(df_raw)} frasi, {df_raw['company'].nunique()} aziende.")

df = label_dataframe(df_raw)
print(f"Dopo l'etichettatura: {len(df)} frasi con un pilastro assegnato (E/S/G).")
print(df["esg"].value_counts().to_string())

# esempio diretto di label_sentence, senza passare da un DataFrame
esempio = "We reduced carbon emissions significantly through renewable energy."
topic, score = label_sentence(esempio)
print(f"\nEsempio: '{esempio}'\n  -> topic ESRS: {topic} (confidenza: {score:.2f})")


# --------------------------------------------------------------------- #
step("3. Grafico: distribuzione delle classi")
# --------------------------------------------------------------------- #
fig = plot_class_distribution(df["label"], title="Distribuzione E/S/G nel mini-dataset di prova")
fig.savefig("test_distribuzione_classi.png", dpi=120)
print("Salvato: test_distribuzione_classi.png")


# --------------------------------------------------------------------- #
step("4. Training (GloVe 50-dim + DNN — la combinazione piu' veloce)")
# --------------------------------------------------------------------- #
# ATTENZIONE: con solo 4 aziende, val_size va tenuto piccolo per avere
# almeno un'azienda per lato; su un dataset vero (10+ aziende) i default
# di Config vanno benissimo.
config = Config(seed=0, glove_name="glove-wiki-gigaword-50", glove_dim=50,
                epochs=15, patience=3, val_size=0.25)

print("Addestramento in corso (la prima volta scarica GloVe-50d, ~66MB)...")
clf = ESGPillarClassifier(config=config, embedding="glove", network="DNN")
clf.fit(df, text_col="sentence", company_col="company", label_col="label")
print("Training completato.")


# --------------------------------------------------------------------- #
step("5. Predizione su frasi nuove (mai viste)")
# --------------------------------------------------------------------- #
nuove_frasi = [
    "The company published its first climate risk disclosure report this year.",
    "New parental leave policies were introduced to support working parents.",
    "Shareholders voted on executive compensation packages during the annual meeting.",
]
predizioni = clf.predict(nuove_frasi)
probabilita = clf.predict_proba(nuove_frasi)
for frase, pred, proba in zip(nuove_frasi, predizioni, probabilita):
    print(f"\n[{pred:14s}] {frase}")
    for nome, p in zip(config.class_names, proba):
        print(f"    {nome:14s}: {p:.3f}")


# --------------------------------------------------------------------- #
step("6. Grafico: curva di training")
# --------------------------------------------------------------------- #
fig2 = clf.plot_history()
fig2.savefig("test_curva_training.png", dpi=120)
print("Salvato: test_curva_training.png")


# --------------------------------------------------------------------- #
step("7. Salvataggio e ricaricamento")
# --------------------------------------------------------------------- #
clf.save("modello_di_prova")
clf_ricaricato = ESGPillarClassifier.load("modello_di_prova")
predizioni_ricaricato = clf_ricaricato.predict(nuove_frasi)
assert predizioni_ricaricato == predizioni, "Le predizioni dopo il ricaricamento dovrebbero coincidere!"
print("Modello salvato in ./modello_di_prova/ e ricaricato correttamente (predizioni identiche).")


# --------------------------------------------------------------------- #
step("Tutto OK! La libreria funziona correttamente.")
# --------------------------------------------------------------------- #
print(
    "\nNota: questo e' un mini-dataset di 20 frasi in 4 aziende finte, pensato\n"
    "solo per verificare che l'installazione funzioni end-to-end -- non aspettarti\n"
    "risultati accurati (troppi pochi dati). Sul tuo dataset vero, usa i default\n"
    "di Config (GloVe 300-dim) e considera anche 'esgbert'/'fasttext' come\n"
    "rappresentazioni alternative da confrontare."
)
