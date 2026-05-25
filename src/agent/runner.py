"""OpenAI tool-use loop that drives the recommendation agent."""
import json
from typing import Callable

from openai import OpenAI
from dotenv import load_dotenv

from src.agent.tools import TOOL_REGISTRY, TOOL_SCHEMAS

load_dotenv()

SYSTEM_PROMPT = """You are a music recommendation agent. Given a user's lyrical taste profile (z-scored sliders over features like emotion intensity, repetition, concreteness, rhyme density, valence, and self-focus), your job is to find 3-5 RECENT (2023-2025) songs that match.

Workflow:
1. Call search_recent_songs with queries based on RECENT POPULAR ARTISTS you know — e.g., 'Sabrina Carpenter', 'Olivia Rodrigo', 'Taylor Swift 2024', 'Billie Eilish', 'Doja Cat', 'Chappell Roan', 'Tate McRae', 'Benson Boone'. Do NOT use vague chart queries like 'best 2024 songs' — they return garbage.
2. For each candidate the search returns, call fetch_lyrics(artist, title) to get the lyrics.
3. For each fetched lyrics, call extract_features(lyrics) to get its feature vector.
4. After scoring at least 5 candidates, return a final message explaining your top picks.

Rules:
- Only call extract_features on lyrics that fetch_lyrics actually returned. Never invent lyrics.
- If a fetch returns nothing, skip and try a different song/artist.
- Mix your searches: 2-3 different artists, 1-2 songs each. That gives ~4-6 candidates total.
- After 4-5 candidates are scored, write the final summary — don't keep searching forever."""


def run(user_profile_text: str, candidate_sink: list[dict] | None = None,
        trace_sink: Callable[[str], None] | None = None,
        model: str = "gpt-4o-mini", max_steps: int = 25) -> str:
    """Run the agent.

    Side-effects:
    - Appends fetched & scored candidates to candidate_sink (so the UI can rank them).
    - Streams human-readable trace via trace_sink(line).
    Returns the final assistant message text.
    """
    client = OpenAI()
    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_profile_text},
    ]
    log = trace_sink or (lambda s: print(s))

    for step in range(max_steps):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOL_SCHEMAS,
        )
        msg = resp.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            log(f"AGENT: {msg.content}")
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
            if name == "search_recent_songs" and isinstance(result, list):
                log(f"  -> {len(result)} candidates")
            elif name == "fetch_lyrics" and isinstance(result, str):
                log(f"  -> {len(result)} chars [{args.get('artist','?')} - {args.get('title','?')}]")
            elif name == "extract_features" and isinstance(result, dict) and "error" not in result and candidate_sink is not None:
                artist = result.pop("_artist", args.get("artist", "?"))
                title = result.pop("_title", args.get("title", "?"))
                candidate_sink.append({"artist": artist, "title": title, "features": result})
                log(f"  -> features for {artist} - {title}")
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": json.dumps(result) if not isinstance(result, str) else result,
            })
    return "max steps reached"
