from src.features.repetition import extract

def test_keys():
    r = extract({"raw": "a\nb\nc"})
    assert set(r.keys()) == {"repetition_entropy", "chorus_repeat_ratio", "mtld"}

def test_high_repetition_high_ratio():
    text = "chorus line one\nchorus line two\n" * 5 + "verse one\nverse two\n"
    r = extract({"raw": text})
    assert r["chorus_repeat_ratio"] > 0.5

def test_no_repetition_zero_ratio():
    text = "\n".join(f"unique line {i}" for i in range(20))
    r = extract({"raw": text})
    assert r["chorus_repeat_ratio"] < 0.1

def test_mtld_higher_for_diverse_vocab():
    diverse = " ".join(f"word{i}" for i in range(200))
    repetitive = " ".join(["hello", "world"] * 100)
    r_d = extract({"raw": diverse})
    r_r = extract({"raw": repetitive})
    assert r_d["mtld"] > r_r["mtld"]
