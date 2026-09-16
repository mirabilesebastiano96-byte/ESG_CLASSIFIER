import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("gensim")
pytest.importorskip("sklearn")

from esgpillar import Config, ESGPillarClassifier  # noqa: E402

_ROWS = [
    {"company": "Acme", "sentence": "We reduced carbon emissions through renewable energy programs.", "label": 0},
    {"company": "Acme", "sentence": "Water consumption decreased across our manufacturing facilities.", "label": 0},
    {"company": "Acme", "sentence": "Employee training programs on workplace safety were expanded significantly.", "label": 1},
    {"company": "Acme", "sentence": "Diversity and inclusion initiatives improved representation across departments.", "label": 1},
    {"company": "Acme", "sentence": "The board approved a new anti corruption compliance policy this year.", "label": 2},
    {"company": "Acme", "sentence": "Audit committee strengthened oversight of governance procedures.", "label": 2},
    {"company": "Beta", "sentence": "Biodiversity protection measures were implemented near our sites.", "label": 0},
    {"company": "Beta", "sentence": "Energy efficiency upgrades reduced our overall climate impact.", "label": 0},
    {"company": "Beta", "sentence": "Worker health and safety incidents declined after training efforts.", "label": 1},
    {"company": "Beta", "sentence": "Community engagement programs supported local employment initiatives.", "label": 1},
    {"company": "Beta", "sentence": "Governance structure was revised to increase board independence.", "label": 2},
    {"company": "Beta", "sentence": "Compliance with anti bribery regulations was strengthened this quarter.", "label": 2},
    {"company": "Gamma", "sentence": "Recycling rates improved after circular economy programs launched.", "label": 0},
    {"company": "Gamma", "sentence": "Water withdrawal from local sources was reduced significantly.", "label": 0},
    {"company": "Gamma", "sentence": "Employee wellbeing initiatives expanded mental health support access.", "label": 1},
    {"company": "Gamma", "sentence": "Parental leave policies were introduced to support working parents.", "label": 1},
    {"company": "Gamma", "sentence": "Board diversity increased following new director appointments.", "label": 2},
    {"company": "Gamma", "sentence": "Shareholders reviewed executive compensation during the annual meeting.", "label": 2},
]


@pytest.mark.parametrize("embedding,dim_field", [("word2vec", "word2vec_dim"), ("lsa", "lsa_dim")])
def test_corpus_embedding_fit_predict_save_load_roundtrip(tmp_path, embedding, dim_field):
    df = pd.DataFrame(_ROWS)
    cfg = Config(**{dim_field: 12}, val_size=0.3, epochs=3, patience=1, lsa_min_df=1)

    clf = ESGPillarClassifier(config=cfg, embedding=embedding, network="DNN")
    clf.fit(df, text_col="sentence", company_col="company", label_col="label")

    test_sentences = ["We invested in solar panels to cut emissions.", "The audit found no compliance issues."]
    pred_before = clf.predict(test_sentences)
    assert len(pred_before) == 2

    save_path = tmp_path / f"model_{embedding}"
    clf.save(save_path)
    assert (save_path / "corpus_kv.pkl").exists(), "corpus_kv.pkl deve esistere per embedding addestrati sul corpus"

    reloaded = ESGPillarClassifier.load(save_path)
    pred_after = reloaded.predict(test_sentences)

    assert pred_after == pred_before, "le predizioni devono coincidere dopo il roundtrip save/load"


def test_corpus_embedding_without_texts_raises():
    """_build_encoder senza 'texts' ne' 'corpus_kv' deve fallire in modo esplicito, non silenzioso."""
    cfg = Config(lsa_min_df=1)
    clf = ESGPillarClassifier(config=cfg, embedding="lsa", network="DNN")
    with pytest.raises(ValueError, match="corpus"):
        clf._build_encoder(companies=["Acme"])
