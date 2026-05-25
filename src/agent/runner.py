"""OpenAI tool-use loop that drives the recommendation agent."""
import json
from typing import Callable

from openai import OpenAI
from dotenv import load_dotenv

from src.agent.tools import TOOL_REGISTRY, TOOL_SCHEMAS

load_dotenv()

SYSTEM_PROMPT = """You are a music recommendation agent. Given a user's lyrical taste profile (z-scored sliders over features like emotion intensity, repetition, concreteness, rhyme density, valence, and self-focus), your job is to find 3-5 RECENT (post-2023) songs that match.

Workflow:
1. Call search_recent_songs with 1-2 well-chosen queries (e.g., 'Billboard Hot 100 2025 emotional ballad').
2. For each candidate the search returns, call fetch_lyrics(artist, title) to get the lyrics.
3. For each fetched lyrics, call extract_features(lyrics) to get its feature vector.
4. After gathering 5-8 scored candidates, return a final message explaining your top picks.

Tips:
- Vary your search queries to surface different candidates if the first batch is thin.
- If a fetch fails, skip and move on.
- Do not invent songs. Only recommend ones whose lyrics you actually retrieved."""


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

    last_search_candidates: list[dict] = []
    last_fetched_title: str | None = None

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
                last_search_candidates = result
                log(f"  -> {len(result)} candidates")
            elif name == "fetch_lyrics" and isinstance(result, str):
                last_fetched_title = args.get("title")
                log(f"  -> {len(result)} chars")
            elif name == "extract_features" and isinstance(result, dict) and candidate_sink is not None:
                title = last_fetched_title or "?"
                artist = next(
                    (c["artist"] for c in last_search_candidates if c["title"] == title), "?"
                )
                candidate_sink.append({"artist": artist, "title": title, "features": result})
                log(f"  -> features for {artist} - {title}")
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": json.dumps(result) if not isinstance(result, str) else result,
            })
    return "max steps reached"
