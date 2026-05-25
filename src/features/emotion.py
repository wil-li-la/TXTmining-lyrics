"""NRC EmoLex emotion + valence/arousal features via NRCLex."""
from nrclex import NRCLex

EMOTIONS = ["anger", "anticipation", "disgust", "fear",
            "joy", "sadness", "surprise", "trust"]
EMOTION_KEYS = [f"emo_{e}" for e in EMOTIONS] + ["valence", "arousal"]

def extract(views: dict) -> dict[str, float]:
    text = views["clean"]
    if not text.strip():
        return {k: 0.0 for k in EMOTION_KEYS}

    nrc = NRCLex(text)
    # NRCLex 4.x no longer auto-populates in __init__; call load_raw_text explicitly.
    if not getattr(nrc, "affect_dict", None):
        nrc.load_raw_text(text)
    freqs = nrc.affect_frequencies
    out = {f"emo_{e}": float(freqs.get(e, 0.0)) for e in EMOTIONS}

    # NRCLex also provides positive/negative; map to a single valence
    pos = freqs.get("positive", 0.0)
    neg = freqs.get("negative", 0.0)
    out["valence"] = float(pos - neg)

    # Arousal proxy: anger + fear + joy + surprise (high-arousal emotions)
    out["arousal"] = float(sum(freqs.get(e, 0.0) for e in ["anger", "fear", "joy", "surprise"]))
    return out
