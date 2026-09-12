"""App web (Gradio) per classificare un report PDF intero.

Richiede l'extra ``tool`` (``pip install esg-pillar-classifier[tool]``)
oltre a ``extraction`` per l'estrazione dei PDF.
"""
from __future__ import annotations

import os
import tempfile

import pandas as pd

from .config import Config
from .pipeline import ESGPillarClassifier

_COL_ESG = {"Environmental": "#2e7d32", "Social": "#ef9a3d", "Governance": "#1565c0"}


def _process_pdf(file_path, clf: ESGPillarClassifier, config: Config):
    import matplotlib.pyplot as plt

    from .extraction import extract_pdf

    if file_path is None:
        return None, None, None, None, "Carica un file PDF."
    try:
        rows = extract_pdf(file_path, config)
    except Exception as e:  # noqa: BLE001
        return None, None, None, None, f"Errore nell'estrazione: {type(e).__name__}: {e}"
    if not rows:
        return None, None, None, None, "Nessuna frase estratta (PDF vuoto, scansione senza testo, o troppo corto)."

    df_sent = pd.DataFrame(rows)
    pred = clf.predict(df_sent["sentence"].tolist())
    proba = clf.predict_proba(df_sent["sentence"].tolist())
    df_sent["classe"] = pred
    df_sent["confidenza"] = proba.max(axis=1).round(3)

    counts = df_sent["classe"].value_counts().reindex(config.class_names).fillna(0).astype(int)
    summary_df = pd.DataFrame({
        "Pilastro": counts.index, "N. frasi": counts.values,
        "Quota %": (100 * counts.values / len(df_sent)).round(1),
    })

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values, color=[_COL_ESG[c] for c in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_title(f"Distribuzione ESG — {len(df_sent)} frasi")
    ax.set_ylabel("n. frasi")
    plt.tight_layout()

    out_table = df_sent[["page", "sentence", "classe", "confidenza"]]
    csv_path = os.path.join(tempfile.gettempdir(), "classificazione_esg.csv")
    out_table.to_csv(csv_path, index=False)

    status = f"Estratte e classificate {len(df_sent)} frasi da '{os.path.basename(file_path)}'."
    return summary_df, out_table, fig, csv_path, status


def launch_tool(clf: ESGPillarClassifier, config: Config | None = None, share: bool = True) -> None:
    """Avvia l'app Gradio: carica un PDF, lo classifica, mostra riepilogo e tabella.

    Parameters
    ----------
    clf : ESGPillarClassifier
        Classificatore gia' addestrato (``.fit(...)`` o ``.load(...)``).
    share : bool
        Se ``True`` (default) genera un link pubblico temporaneo, utile
        per mostrare il lavoro anche girando in locale.
    """
    import gradio as gr

    config = config or clf.config

    with gr.Blocks(title="Classificatore ESG") as demo:
        gr.Markdown(
            "# Classificatore ESG per report di sostenibilita'\n"
            f"Modello: **{clf.network_name}** su embedding **{clf.embedding_name}**. "
            "Carica un PDF: l'app estrae le frasi e le classifica in Environmental / Social / Governance."
        )
        with gr.Row():
            file_in = gr.File(label="Report PDF", file_types=[".pdf"], type="filepath")
            btn = gr.Button("Classifica", variant="primary")
        status_out = gr.Textbox(label="Stato", interactive=False)
        with gr.Row():
            summary_out = gr.Dataframe(label="Riepilogo per pilastro")
            chart_out = gr.Plot(label="Distribuzione")
        table_out = gr.Dataframe(label="Frasi classificate", wrap=True)
        csv_out = gr.File(label="Scarica CSV completo")

        btn.click(
            fn=lambda f: _process_pdf(f, clf, config),
            inputs=file_in,
            outputs=[summary_out, table_out, chart_out, csv_out, status_out],
        )

    demo.launch(share=share)
