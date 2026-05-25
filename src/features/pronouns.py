"""Pronoun-group ratios. Consumes the tokenized (stopwords-kept) view."""

_GROUPS = {
    "pronoun_i":    {"i", "me", "my", "mine", "myself"},
    "pronoun_you":  {"you", "your", "yours", "yourself", "yourselves"},
    "pronoun_we":   {"we", "us", "our", "ours", "ourselves"},
    "pronoun_they": {"they", "them", "their", "theirs", "themselves"},
}

def extract(views: dict) -> dict[str, float]:
    tokens = views["tokenized"].split()
    n = len(tokens)
    if n == 0:
        return {k: 0.0 for k in _GROUPS}
    return {
        k: sum(1 for t in tokens if t in members) / n
        for k, members in _GROUPS.items()
    }
