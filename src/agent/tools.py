"""Tools the OpenAI agent can call: search, fetch lyrics, extract features."""
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
import lyricsgenius

from src.preprocess_v2 import make_views
from src.features import rhyme, repetition, emotion, concreteness, pronouns, embedding

load_dotenv()
_FEATURE_MODS = [rhyme, repetition, emotion, concreteness, pronouns, embedding]
_CACHE_PATH = "data/agent_cache.csv"


_genius: lyricsgenius.Genius | None = None


def _get_genius() -> lyricsgenius.Genius:
    global _genius
    if _genius is None:
        _genius = lyricsgenius.Genius(
            os.environ["GENIUS_ACCESS_TOKEN"],
            timeout=15, sleep_time=1,
            remove_section_headers=True,
        )
        _genius.verbose = False
    return _genius


def search_recent_songs(query: str, max_results: int = 8) -> list[dict[str, str]]:
    """Search Genius for songs matching the query. Returns structured {artist, title, year?} hits.

    Uses the official Genius search endpoint via LyricsGenius — far more reliable for
    music discovery than free-text web search (which surfaces blog roundups).
    """
    rows: list[dict[str, str]] = []
    try:
        genius = _get_genius()
        resp = genius.search_songs(query, per_page=max_results)
        for hit in resp.get("hits", [])[:max_results]:
            res = hit.get("result", {})
            artist = (res.get("primary_artist") or {}).get("name", "").strip()
            title = res.get("title", "").strip()
            if artist and title:
                rows.append({"artist": artist, "title": title})
    except Exception as e:
        print(f"  [search_recent_songs ERROR] {e}")
    return rows


def fetch_lyrics(artist: str, title: str) -> str | None:
    """Fetch lyrics via LyricsGenius search. Cached to disk."""
    cache_key = f"{artist.lower().strip()}|{title.lower().strip()}"
    if Path(_CACHE_PATH).exists():
        cache = pd.read_csv(_CACHE_PATH)
        hit = cache[cache["key"] == cache_key]
        if len(hit):
            return hit.iloc[0]["lyrics"]
    try:
        song = _get_genius().search_song(title=title, artist=artist, get_full_info=False)
        lyrics = song.lyrics if song and song.lyrics else None
    except Exception as e:
        print(f"  [fetch_lyrics ERROR] {e}")
        lyrics = None
    if lyrics:
        new_row = pd.DataFrame([{
            "key": cache_key, "artist": artist, "title": title, "lyrics": lyrics
        }])
        if Path(_CACHE_PATH).exists():
            new_row = pd.concat([pd.read_csv(_CACHE_PATH), new_row], ignore_index=True)
        new_row.to_csv(_CACHE_PATH, index=False)
    return lyrics


def extract_features(artist: str, title: str) -> dict[str, Any]:
    """Run all feature modules on the cached lyrics for (artist, title).

    Takes (artist, title) rather than raw lyrics so the agent doesn't have to
    pass long lyric strings back through tool args (avoids attribution bugs
    when the LLM trims or reformats the string).
    """
    lyrics = fetch_lyrics(artist, title)  # uses cache if already fetched
    if not lyrics:
        return {"error": f"no cached lyrics for {artist} - {title}"}
    views = make_views(lyrics)
    out: dict[str, Any] = {"_artist": artist, "_title": title}
    for mod in _FEATURE_MODS:
        out.update(mod.extract(views))
    return out


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_recent_songs",
            "description": "Search Genius's music catalog for songs. Returns structured {artist, title} hits. For recent music, use specific recent artist names (e.g., 'Sabrina Carpenter 2024', 'Olivia Rodrigo', 'Billie Eilish 2024') rather than vague queries like 'best 2024 songs' (which produces garbage).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query — works best with specific recent artist names, e.g. 'Sabrina Carpenter 2024' or 'Taylor Swift Fortnight'."},
                    "max_results": {"type": "integer", "default": 8},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_lyrics",
            "description": "Fetch lyrics for a specific song from Genius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["artist", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_features",
            "description": "Compute the full feature vector (rhyme, repetition, emotion, concreteness, pronouns, embedding) for a song. Must be called AFTER fetch_lyrics for the same (artist, title) — looks up cached lyrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["artist", "title"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "search_recent_songs": search_recent_songs,
    "fetch_lyrics": fetch_lyrics,
    "extract_features": extract_features,
}
