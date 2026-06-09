"""Recommendation: whole-catalog retrieval + an LLM-written explanation.

See docs/adr/0001-recommend-by-retrieval-on-provenance-safe-features.md.

Recommendations are produced by deterministic retrieval over the precomputed
feature index (data/song_features.parquet) — ranked on provenance-safe features,
deduped by embedding. The language model only writes the "why these match"
prose over the already-ranked results. The old live tool-use search loop is
retained as a fallback for the (practically impossible) empty-retrieval case;
its prompt no longer hardcodes specific artists, so it cannot reintroduce bias.
"""
import json
from typing import Callable

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from src.agent.ranking import recommend, SAFE_FEATURES, LINE_FEATURES
from src.agent.tools import TOOL_REGISTRY, TOOL_SCHEMAS

load_dotenv()

_FEATURES_PATH = "data/song_features.parquet"
_META_PATH = "data/lyrics_processed.csv"
_index: pd.DataFrame | None = None


def _load_index(features_path: str = _FEATURES_PATH,
                meta_path: str = _META_PATH) -> pd.DataFrame:
    """Load & cache the catalog feature index joined to artist/title.

    song_features.parquet is keyed by song_id and already carries `year` and the
    `embedding`; artist/title come from lyrics_processed.csv. The join is an
    inner join so the one feature row without metadata is dropped.
    """
    global _index
    if _index is None:
        feat = pd.read_parquet(features_path)
        meta = pd.read_csv(meta_path, usecols=["song_id", "artist", "title"])
        df = feat.merge(meta, on="song_id", how="inner")
        df = df.dropna(subset=["artist", "title"])
        _index = df
    return _index


EXPLAIN_PROMPT = """You are a music recommendation assistant. The system has \
already ranked songs from a 1965-2025 catalog against the listener's lyrical \
taste profile (z-scored sliders). Your ONLY job is to explain, warmly and \
concisely, why these specific songs match — referencing the profile dimensions \
(emotional valence, arousal/intensity, concreteness, self-focus). Mention each \
song by artist and title. Do NOT invent songs, do NOT suggest others, and do \
NOT claim the songs are recent unless their year says so. 4-6 sentences."""


def _profile_text(profile: dict[str, float]) -> str:
    return "\n".join(f"- {k}: {float(v):+.1f} z" for k, v in profile.items())


def _explain(profile: dict[str, float], ranked: list[dict], model: str,
             log: Callable[[str], None]) -> str:
    """Ask the LLM to write the 'why these match' prose over ranked results."""
    songs = "\n".join(
        f"{i}. {r['artist']} — {r['title']} ({int(r.get('year', 0) or 0)}), "
        f"match {r['score']:.2f}"
        for i, r in enumerate(ranked, 1)
    )
    user = (
        f"Listener taste profile:\n{_profile_text(profile)}\n\n"
        f"Ranked matches:\n{songs}"
    )
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXPLAIN_PROMPT},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def run(profile: dict[str, float], stats: dict,
        candidate_sink: list[dict] | None = None,
        trace_sink: Callable[[str], None] | None = None,
        model: str = "gpt-4o-mini", k: int = 5) -> str:
    """Recommend k songs for the taste profile and explain the matches.

    Side effects:
    - Appends the ranked recommendations to candidate_sink (for the UI).
    - Streams human-readable progress via trace_sink(line).
    Returns the LLM explanation text.
    """
    log = trace_sink or (lambda s: print(s))
    index = _load_index()
    log(f"Ranking {len(index):,} songs against your taste profile…")
    ranked = recommend(index, profile, stats, k=k)

    if not ranked:
        log("No retrieval matches — falling back to live search.")
        return _live_search_fallback(profile, candidate_sink, trace_sink, model)

    feat_keys = list(stats.keys())
    if candidate_sink is not None:
        for r in ranked:
            candidate_sink.append({
                "artist": r["artist"],
                "title": r["title"],
                "year": int(r.get("year", 0) or 0),
                "score": float(r["score"]),
                "features": {kk: float(r.get(kk, 0.0)) for kk in feat_keys},
            })
    for r in ranked:
        log(f"  ✓ {r['artist']} — {r['title']} "
            f"({int(r.get('year', 0) or 0)})  match {r['score']:.3f}")

    log("Writing explanation…")
    return _explain(profile, ranked, model, log)


# ---------------------------------------------------------------------------
# Fallback: the original live tool-use search loop, kept for the empty-retrieval
# case. The system prompt no longer names specific artists (the source of the
# original bias) — it asks for a spread across genres, eras, and styles.
# ---------------------------------------------------------------------------

FALLBACK_PROMPT = """You are a music recommendation agent. Retrieval returned \
nothing, so find 4 songs that match the user's lyrical taste profile (z-scored \
sliders over valence, arousal, concreteness, self-focus, rhyme, repetition).

MANDATORY workflow — call each tool, in order, for each candidate:
1. search_recent_songs(query) — query a SPECIFIC ARTIST plus a SONG TITLE (never \
an album, never a bare year). DELIBERATELY VARY your picks: span different \
genres, decades, and artists — do not cluster on one artist or one current-pop \
scene. Each hit gives {artist, title, id}; pick the original studio cut.
2. fetch_lyrics(artist, title, song_id) — using the id from the search hit.
3. extract_features(artist, title, song_id) — same id; required for ranking.

Stop once 4 DIFFERENT artists are processed through extract_features, then write \
a short summary naming each song. Always pass the numeric id. Never invent songs."""


def _live_search_fallback(profile: dict[str, float],
                          candidate_sink: list[dict] | None,
                          trace_sink: Callable[[str], None] | None,
                          model: str, max_steps: int = 25) -> str:
    client = OpenAI()
    messages: list = [
        {"role": "system", "content": FALLBACK_PROMPT},
        {"role": "user", "content": _profile_text(profile)},
    ]
    log = trace_sink or (lambda s: print(s))
    for _ in range(max_steps):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOL_SCHEMAS,
        )
        msg = resp.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content or ""
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            log(f"TOOL {name}({json.dumps(args)[:120]})")
            try:
                result = TOOL_REGISTRY[name](**args)
            except Exception as e:
                result = {"error": str(e)}
                log(f"  ERROR: {e}")
            if name == "extract_features" and isinstance(result, dict) \
                    and "error" not in result and candidate_sink is not None:
                artist = result.pop("_artist", args.get("artist", "?"))
                title = result.pop("_title", args.get("title", "?"))
                candidate_sink.append(
                    {"artist": artist, "title": title, "features": result})
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": json.dumps(result) if not isinstance(result, str) else result,
            })
    return "max steps reached"
