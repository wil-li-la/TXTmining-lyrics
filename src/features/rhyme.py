"""Rhyme + cadence features from raw line-broken lyrics, via CMU dict."""
import re
from collections import defaultdict
import pronouncing

_WORD = re.compile(r"[A-Za-z']+")

def _last_word(line: str) -> str | None:
    words = _WORD.findall(line)
    return words[-1].lower() if words else None

def _rhyming_part(word: str) -> str | None:
    phones = pronouncing.phones_for_word(word)
    if not phones:
        return None
    return pronouncing.rhyming_part(phones[0])

def _syllables(word: str) -> int:
    phones = pronouncing.phones_for_word(word)
    if not phones:
        # crude fallback
        return max(1, len(re.findall(r"[aeiouy]+", word.lower())))
    return pronouncing.syllable_count(phones[0])

def extract(views: dict) -> dict[str, float]:
    raw = views["raw"]
    lines = [l for l in raw.split("\n") if l.strip()]
    if not lines:
        return {"rhyme_density": 0.0, "internal_rhyme": 0.0, "mean_syllables_per_line": 0.0}

    # End-rhyme density: fraction of line pairs (within 4 lines) sharing a rhyming-part
    end_rhymes_parts: list[tuple[int, str]] = []
    for i, ln in enumerate(lines):
        w = _last_word(ln)
        if not w:
            continue
        rp = _rhyming_part(w)
        if rp:
            end_rhymes_parts.append((i, rp))

    if len(end_rhymes_parts) < 2:
        density = 0.0
    else:
        n_pairs = 0
        n_rhyme = 0
        for a in range(len(end_rhymes_parts)):
            for b in range(a + 1, min(a + 5, len(end_rhymes_parts))):
                n_pairs += 1
                if end_rhymes_parts[a][1] == end_rhymes_parts[b][1]:
                    n_rhyme += 1
        density = n_rhyme / n_pairs if n_pairs else 0.0

    # Internal rhyme: fraction of lines containing >= 2 words sharing a rhyming-part
    internal = 0
    for ln in lines:
        words = [w.lower() for w in _WORD.findall(ln)]
        parts = defaultdict(int)
        for w in words:
            rp = _rhyming_part(w)
            if rp:
                parts[rp] += 1
        if any(c >= 2 for c in parts.values()):
            internal += 1
    internal_rhyme = internal / len(lines)

    # Mean syllables per line
    syl_per_line = [
        sum(_syllables(w) for w in _WORD.findall(ln))
        for ln in lines
    ]
    mean_syl = sum(syl_per_line) / len(syl_per_line)

    return {
        "rhyme_density": float(density),
        "internal_rhyme": float(internal_rhyme),
        "mean_syllables_per_line": float(mean_syl),
    }
