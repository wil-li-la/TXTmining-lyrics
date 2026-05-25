from src.features.rhyme import extract

def test_extract_returns_three_keys():
    r = extract({"raw": "moon\nsoon\nstar\ncar"})
    assert set(r.keys()) == {"rhyme_density", "internal_rhyme", "mean_syllables_per_line"}

def test_strong_rhymes_have_high_density():
    text = "moon\nsoon\njune\ntune\n"  # all rhyme
    r = extract({"raw": text})
    assert r["rhyme_density"] > 0.5

def test_no_rhymes_low_density():
    text = "purple\norange\nsilver\n"  # famously non-rhyming
    r = extract({"raw": text})
    assert r["rhyme_density"] < 0.5

def test_empty_input_safe():
    r = extract({"raw": ""})
    assert r["rhyme_density"] == 0.0
    assert r["mean_syllables_per_line"] == 0.0
