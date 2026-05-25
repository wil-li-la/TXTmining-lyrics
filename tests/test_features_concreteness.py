from src.features.concreteness import extract

def test_keys():
    r = extract({"clean": "table chair house"})
    assert set(r.keys()) == {"mean_concreteness", "pct_concrete"}

def test_concrete_words_high_score():
    r_concrete = extract({"clean": "table chair house car tree dog book"})
    r_abstract = extract({"clean": "freedom justice belief idea hope thought concept"})
    assert r_concrete["mean_concreteness"] > r_abstract["mean_concreteness"]
    assert r_concrete["pct_concrete"] > r_abstract["pct_concrete"]

def test_empty_safe():
    r = extract({"clean": ""})
    assert r["mean_concreteness"] == 0.0
    assert r["pct_concrete"] == 0.0
