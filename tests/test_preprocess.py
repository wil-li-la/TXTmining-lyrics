"""Tests for preprocessing functions."""

from src.preprocess import clean_text, deduplicate_lines


def test_clean_text_lowercases():
    assert "hello world" in clean_text("Hello World")


def test_clean_text_removes_punctuation():
    result = clean_text("hello, world! it's a test.")
    assert "," not in result
    assert "!" not in result


def test_clean_text_removes_stopwords():
    result = clean_text("I am the best in the world")
    assert "the" not in result.split()
    assert "am" not in result.split()
    assert "best" in result.split()
    assert "world" in result.split()


def test_deduplicate_lines_removes_repeated_lines():
    text = "I love you\nYeah yeah\nI love you\nSo much\nYeah yeah"
    result = deduplicate_lines(text)
    lines = result.strip().split("\n")
    assert lines.count("I love you") == 1
    assert lines.count("Yeah yeah") == 1
    assert "So much" in lines


def test_deduplicate_lines_preserves_order():
    text = "first\nsecond\nfirst\nthird"
    result = deduplicate_lines(text)
    lines = result.strip().split("\n")
    assert lines == ["first", "second", "third"]
