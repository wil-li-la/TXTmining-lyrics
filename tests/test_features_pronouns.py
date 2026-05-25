from src.features.pronouns import extract

def test_keys():
    r = extract({"tokenized": "i love you"})
    assert set(r.keys()) == {"pronoun_i", "pronoun_you", "pronoun_we", "pronoun_they"}

def test_i_heavy_text():
    r = extract({"tokenized": "i went and i saw and i thought i could"})
    assert r["pronoun_i"] > r["pronoun_you"]

def test_no_pronouns_safe():
    r = extract({"tokenized": "the cat sat on the mat"})
    assert all(v == 0.0 for v in r.values())
