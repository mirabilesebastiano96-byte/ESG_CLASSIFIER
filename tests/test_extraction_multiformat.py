import pytest

pytest.importorskip("docx")
pytest.importorskip("nltk")

from esgpillar.config import Config  # noqa: E402
from esgpillar.extraction import extract_directory, extract_document, extract_docx, extract_txt  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _ensure_nltk_data():
    import nltk
    for pkg in ["punkt", "punkt_tab"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


def _make_docx(path, sentences):
    from docx import Document
    doc = Document()
    for s in sentences:
        doc.add_paragraph(s)
    doc.save(path)


_SENTENCES = [
    "The company reduced carbon emissions significantly through renewable energy investments.",
    "Employee training programs on workplace safety were expanded across all regional offices.",
    "The board of directors approved a new anti-corruption compliance policy last quarter.",
]


def test_extract_docx(tmp_path):
    path = tmp_path / "Alpha_2023.docx"
    _make_docx(path, _SENTENCES)
    rows = extract_docx(str(path), Config(check_verb=False, min_words_sent=4))
    assert len(rows) >= 1
    assert all(r["company"] == "Alpha" for r in rows)
    assert all(r["year"] == 2023 for r in rows)


def test_extract_txt(tmp_path):
    path = tmp_path / "Beta_2022.txt"
    path.write_text("\n\n".join(_SENTENCES), encoding="utf-8")
    rows = extract_txt(str(path), Config(check_verb=False, min_words_sent=4))
    assert len(rows) >= 1
    assert all(r["company"] == "Beta" for r in rows)


def test_extract_document_dispatches_by_extension(tmp_path):
    docx_path = tmp_path / "Gamma_2021.docx"
    _make_docx(docx_path, _SENTENCES)
    txt_path = tmp_path / "Delta_2020.txt"
    txt_path.write_text("\n\n".join(_SENTENCES), encoding="utf-8")

    cfg = Config(check_verb=False, min_words_sent=4)
    rows_docx = extract_document(str(docx_path), cfg)
    rows_txt = extract_document(str(txt_path), cfg)
    assert len(rows_docx) >= 1
    assert len(rows_txt) >= 1


def test_extract_document_unsupported_extension_raises(tmp_path):
    path = tmp_path / "Report_2020.doc"
    path.write_text("dummy", encoding="utf-8")
    with pytest.raises(ValueError, match="non supportato"):
        extract_document(str(path))


def test_extract_directory_mixed_formats(tmp_path):
    _make_docx(tmp_path / "Alpha_2023.docx", _SENTENCES)
    beta_sentences = [
        "Beta reduced its water consumption through efficient irrigation technology upgrades.",
        "The workforce diversity program expanded recruitment across underrepresented groups significantly.",
        "Governance committees reviewed risk management procedures during the annual shareholder meeting.",
    ]
    (tmp_path / "Beta_2022.txt").write_text("\n\n".join(beta_sentences), encoding="utf-8")

    cfg = Config(pdf_dir=str(tmp_path), extract_cache=str(tmp_path / "cache.csv"),
                check_verb=False, min_words_sent=4)
    df = extract_directory(config=cfg, use_cache=False)
    assert set(df["company"]) == {"Alpha", "Beta"}
    assert len(df) >= 2
