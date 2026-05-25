"""OpenAI tool-use loop that drives the recommendation agent."""
import json
from typing import Callable

from openai import OpenAI
from dotenv import load_dotenv

from src.agent.tools import TOOL_REGISTRY, TOOL_SCHEMAS

load_dotenv()

SYSTEM_PROMPT = """You are a music recommendation agent. Given a user's lyrical taste profile (z-scored sliders over features like emotion intensity, repetition, concreteness, rhyme density, valence, and self-focus), find 3-5 RECENT (2023-2025) songs that match.

MANDATORY workflow — you MUST call each tool, in order:

1. search_recent_songs(query) — Use queries with SPECIFIC RECENT ARTISTS: 'Sabrina Carpenter', 'Olivia Rodrigo', 'Taylor Swift 2024', 'Billie Eilish', 'Doja Cat', 'Chappell Roan', 'Tate McRae', 'Benson Boone', 'Gracie Abrams'. Do 2-3 searches across different artists. NEVER use vague queries like 'best 2024 songs'.

2. fetch_lyrics(artist, title) — Call once per candidate from the search results to retrieve and cache the lyrics.

3. extract_features(artist, title) — REQUIRED for every candidate. This call computes the feature vector that lets the system rank songs against the user's profile. Without it, the recommendation pipeline returns nothing.

4. ONLY AFTER 3-5 candidates have been processed through extract_features, write the final summary. Mention each song you scored.

Hard rules:
- You MUST call extract_features for at least 3 songs before writing the final message. Do not skip step 3.
- extract_features takes (artist, title) — NOT raw lyrics. It reads from the cache populated by fetch_lyrics.
- If a fetch returns nothing, skip that song and try another.
- Do not invent songs you didn't actually fetch."""


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
