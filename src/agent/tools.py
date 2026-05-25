"""Tools the OpenAI agent can call: search, fetch lyrics, extract features."""
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from ddgs import DDGS
import lyricsgenius

from src.preprocess_v2 import make_views
from src.features import rhyme, repetition, emotion, concreteness, pronouns, embedding

load_dotenv()
_FEATURE_MODS = [rhyme, repetition, emotion, concreteness, pronouns, embedding]
_CACHE_PATH = "data/agent_cache.csv"
_TITLE_ARTIST = re.compile(r"^\s*([^-|]+?)\s*[-–]\s*([^|]+?)\s*[|]?\s*", flags=re.UNICODE)


def _parse_title(text: str) -> dict[str, str] | None:
    """Extract (artist, title) from a search-result title like 'Olivia Rodrigo - drivers license | Billboard'."""
    m = _TITLE_ARTIST.match(text)
    if not m:
        return None
    artist = m.group(1).strip()
    title = m.group(2).strip()
    if len(artist) > 60 or len(title) > 60:
        return None
    return {"artist": artist, "title": title}


def search_recent_songs(query: str, max_results: int = 8) -> list[dict[str, str]]:
    """Web search; return parsed (artist, title) pairs."""
    rows: list[dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results * 2):
                parsed = _parse_title(r.get("title", ""))
                if parsed:
                    rows.append(parsed)
                if len(rows) >= max_results:
                    break
    except Exception as e:
        print(f"  [search_recent_songs ERROR] {e}")
    return rows


_genius: lyricsgenius.Genius | None = None


def _get_genius() -> lyricsgenius.Genius:
    global _genius
    if _genius is None:
        _genius = lyricsgenius.Genius(
            os.environ["GENIUS_ACCESS_TOKEN"],
            timeout=15, sleep_time=1, verbose=False,
            remove_section_headers=True,
        )
    return _genius


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


def extract_features(lyrics: str) -> dict[str, Any]:
    """Run all feature modules on raw lyrics."""
    views = make_views(lyrics)
    out: dict[str, Any] = {}
    for mod in _FEATURE_MODS:
        out.update(mod.extract(views))
    return out


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_recent_songs",
            "description": "Search the web for recent songs matching a query. Returns a list of {artist, title}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, e.g. 'Billboard Hot 100 2025 emotional ballads'"},
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
            "description": "Compute the full feature vector (rhyme, repetition, emotion, concreteness, pronouns, embedding) for given lyrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lyrics": {"type": "string"},
                },
                "required": ["lyrics"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "search_recent_songs": search_recent_songs,
    "fetch_lyrics": fetch_lyrics,
    "extract_features": extract_features,
}
