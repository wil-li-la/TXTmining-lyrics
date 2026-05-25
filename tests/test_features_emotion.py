from src.features.emotion import extract, EMOTION_KEYS

def test_returns_all_emotion_keys():
    r = extract({"clean": "happy joy smile love"})
    assert set(r.keys()) == set(EMOTION_KEYS)

def test_happy_words_have_high_joy():
    r = extract({"clean": "happy joy smile delight cheerful glad"})
    assert r["emo_joy"] > r["emo_sadness"]

def test_angry_words_have_high_anger():
    r = extract({"clean": "rage fury hate destroy attack angry"})
    assert r["emo_anger"] > r["emo_trust"]

def test_empty_input_safe():
    r = extract({"clean": ""})
    assert all(v == 0.0 for v in r.values())
