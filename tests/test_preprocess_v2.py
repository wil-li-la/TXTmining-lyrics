import pytest
from src.preprocess_v2 import make_views

def test_make_views_returns_three_keys():
    v = make_views("Hello world!\nI'm a teapot.")
    assert set(v.keys()) == {"raw", "tokenized", "clean"}

def test_raw_preserves_line_breaks():
    v = make_views("Line one\nLine two")
    assert "\n" in v["raw"]

def test_tokenized_keeps_pronouns():
    v = make_views("I love you and we are happy")
    assert "i" in v["tokenized"].split()
    assert "you" in v["tokenized"].split()
    assert "we" in v["tokenized"].split()

def test_clean_drops_stopwords():
    v = make_views("I love you and we are happy")
    tokens = v["clean"].split()
    assert "love" in tokens
    assert "happy" in tokens
    assert "i" not in tokens
    assert "you" not in tokens

def test_clean_drops_short_tokens():
    v = make_views("OK go run far")
    tokens = v["clean"].split()
    assert "run" in tokens  # 3 chars: passes
    assert "ok" not in tokens  # 2 chars: drops
    assert "go" not in tokens  # 2 chars: drops
