"""Tools the OpenAI agent can call: search, fetch lyrics, extract features."""
import json
import os
import urllib.parse
import urllib.request
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


def search_recent_songs(query: str, max_results: int = 8) -> list[dict[str, Any]]:
    """Search Genius for songs matching the query. Returns structured {artist, title, id} hits.

    Uses the official Genius search endpoint via LyricsGenius — far more reliable for
    music discovery than free-text web search (which surfaces blog roundups).

    Genius search mixes real songs with annotation pages (release calendars, listening
    logs, translations) whose URLs end in '-annotated' and have no fetchable lyrics. Only
    real song pages end in '-lyrics', so we keep those and drop the rest. The numeric
    `id` is the stable handle callers pass to fetch_lyrics/extract_features — never the
    title, which Genius decorates (e.g. 'LUNCH (Mixed) [Sep 2024]') and can't round-trip.
    """
    rows: list[dict[str, Any]] = []
    try:
        genius = _get_genius()
        resp = genius.search_songs(query, per_page=max_results)
        for hit in resp.get("hits", [])[:max_results]:
            res = hit.get("result", {})
            if not (res.get("url") or "").endswith("-lyrics"):
                continue  # annotation / calendar / translation page, not a song
            artist = (res.get("primary_artist") or {}).get("name", "").strip()
            title = (res.get("title") or "").strip()
            song_id = res.get("id")
            if artist and title and song_id is not None:
                rows.append({"artist": artist, "title": title, "id": int(song_id)})
    except Exception as e:
        print(f"  [search_recent_songs ERROR] {e}")
    return rows


def _lrclib_lyrics(artist: str, title: str) -> str | None:
    """Fetch plain lyrics from lrclib.net (free, no-auth JSON API).

    Used as the primary source because it serves lyric *text* over an API — Genius's
    lyric pages are HTML-scraped, which datacenter IPs (e.g. the HF Space) get blocked
    from. lrclib matches by artist + track name, so we pass the clean title from search.
    """
    qs = urllib.parse.urlencode({"artist_name": artist, "track_name": title})
    req = urllib.request.Request(
        f"https://lrclib.net/api/get?{qs}",
        headers={"User-Agent": "pop-lyrics-taste-profiler (github.com/wil-li-la)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
        lyrics = data.get("plainLyrics") or data.get("syncedLyrics")
        return lyrics if lyrics and lyrics.strip() else None
    except Exception as e:
        print(f"  [lrclib ERROR] {e}")
        return None


def fetch_lyrics(artist: str, title: str, song_id: int) -> str | None:
    """Fetch lyrics for a song, keyed/cached by stable Genius song_id. Cached to disk.

    Source order: lrclib.net (API, cloud-friendly) first, then Genius scrape-by-id as a
    fallback. Genius is fetched by id — never re-searched by title — because Genius
    decorates titles ('Feather (Mixed) [Apr. 2024]') that don't round-trip. lyrics
    *text* scraping from genius.com is blocked on datacenter IPs, which is why lrclib
    leads. artist/title drive the lrclib lookup and cache readability.
    """
    cache_key = str(song_id)
    if Path(_CACHE_PATH).exists():
        cache = pd.read_csv(_CACHE_PATH)
        hit = cache[cache["key"].astype(str) == cache_key]
        if len(hit):
            val = hit.iloc[0]["lyrics"]
            return val if isinstance(val, str) and val.strip() else None
    lyrics = _lrclib_lyrics(artist, title)
    if not lyrics:
        try:
            lyrics = _get_genius().lyrics(song_id=int(song_id), remove_section_headers=True)
            lyrics = lyrics if lyrics and lyrics.strip() else None
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


def extract_features(artist: str, title: str, song_id: int) -> dict[str, Any]:
    """Run all feature modules on the lyrics for the given Genius song_id.

    Takes (artist, title, song_id) rather than raw lyrics so the agent doesn't have
    to pass long lyric strings back through tool args (avoids attribution bugs when
    the LLM trims or reformats the string). song_id is the stable handle from
    search_recent_songs; lyrics are looked up / fetched via that id.
    """
    lyrics = fetch_lyrics(artist, title, song_id)  # uses cache if already fetched
    if not lyrics:
        return {"error": f"no lyrics for {artist} - {title} (id {song_id})"}
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
            "description": "Search Genius's music catalog for songs. Returns a list of {artist, title, id} hits — `id` is the numeric Genius song id you MUST pass to fetch_lyrics and extract_features. Query with a specific artist plus a song or album name (e.g. 'Sabrina Carpenter Espresso', 'Olivia Rodrigo GUTS'). Do NOT append a bare year like 'Sabrina Carpenter 2024' — that surfaces release-calendar and listening-log pages, not songs.",
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
            "description": "Fetch and cache lyrics for a specific song from Genius, by its numeric id from search_recent_songs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist": {"type": "string"},
                    "title": {"type": "string"},
                    "song_id": {"type": "integer", "description": "The `id` field from the search_recent_songs hit for this song."},
                },
                "required": ["artist", "title", "song_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_features",
            "description": "Compute the full feature vector (rhyme, repetition, emotion, concreteness, pronouns, embedding) for a song. Must be called AFTER fetch_lyrics for the same song_id — looks up the cached lyrics by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist": {"type": "string"},
                    "title": {"type": "string"},
                    "song_id": {"type": "integer", "description": "The same `id` you passed to fetch_lyrics."},
                },
                "required": ["artist", "title", "song_id"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "search_recent_songs": search_recent_songs,
    "fetch_lyrics": fetch_lyrics,
    "extract_features": extract_features,
}
