"""Comandi da terminale: ``esgpillar-extract``, ``esgpillar-classify``."""
from __future__ import annotations

import argparse

from .config import Config


def extract_cli() -> None:
    """``esgpillar-extract --pdf-dir ./reports --out dataset.csv``"""
    from .extraction import extract_directory

    parser = argparse.ArgumentParser(description="Estrae frasi da una cartella di report PDF.")
    parser.add_argument("--pdf-dir", default="./reports", help="Cartella con i PDF")
    parser.add_argument("--out", default="dataset_from_pdfs.csv", help="CSV di output")
    args = parser.parse_args()

    config = Config(pdf_dir=args.pdf_dir, extract_cache=args.out)
    df = extract_directory(config=config, use_cache=False)
    print(f"\nFrasi estratte: {len(df)} | aziende: {df['company'].nunique()}")


def classify_cli() -> None:
    """``esgpillar-classify --model ./mio_modello --text "We reduced emissions."``"""
    from .pipeline import ESGPillarClassifier

    parser = argparse.ArgumentParser(description="Classifica una frase con un modello gia' addestrato.")
    parser.add_argument("--model", required=True, help="Cartella del modello salvato con .save()")
    parser.add_argument("--text", required=True, help="Frase da classificare")
    args = parser.parse_args()

    clf = ESGPillarClassifier.load(args.model)
    result = clf.predict([args.text])[0]
    proba = clf.predict_proba([args.text])[0]
    print(f"Pilastro: {result}")
    for name, p in zip(clf.config.class_names, proba):
        print(f"  {name:14s}: {p:.3f}")
