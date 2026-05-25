"""Repetition + vocabulary-diversity features."""
import math
from collections import Counter

def _line_repeat_ratio(raw: str) -> float:
    lines = [l.strip().lower() for l in raw.split("\n") if l.strip()]
    if not lines:
        return 0.0
    counts = Counter(lines)
    repeats = sum(c for c in counts.values() if c >= 2)
    return repeats / len(lines)

def _line_bigram_entropy(raw: str) -> float:
    lines = [l.strip().lower() for l in raw.split("\n") if l.strip()]
    if len(lines) < 2:
        return 0.0
    bigrams = list(zip(lines[:-1], lines[1:]))
    counts = Counter(bigrams)
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def _mtld(text: str, threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy 2005)."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0

    def one_pass(toks):
        factors = 0
        types = set()
        running = 0
        for t in toks:
            types.add(t)
            running += 1
            ttr = len(types) / running
            if ttr <= threshold:
                factors += 1
                types = set()
                running = 0
        if running > 0:
            ttr = len(types) / running
            partial = (1 - ttr) / (1 - threshold) if ttr < 1 else 0
            factors += partial
        return len(toks) / factors if factors else len(toks)

    forward = one_pass(tokens)
    backward = one_pass(list(reversed(tokens)))
    return (forward + backward) / 2

def extract(views: dict) -> dict[str, float]:
    raw = views["raw"]
    return {
        "repetition_entropy": float(_line_bigram_entropy(raw)),
        "chorus_repeat_ratio": float(_line_repeat_ratio(raw)),
        "mtld": float(_mtld(raw)),
    }
