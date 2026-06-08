"""OpenAI tool-use loop that drives the recommendation agent."""
import json
from typing import Callable

from openai import OpenAI
from dotenv import load_dotenv

from src.agent.tools import TOOL_REGISTRY, TOOL_SCHEMAS

load_dotenv()

SYSTEM_PROMPT = """You are a music recommendation agent. Given a user's lyrical taste profile (z-scored sliders over features like emotion intensity, repetition, concreteness, rhyme density, valence, and self-focus), find 3-5 RECENT (2023-2025) songs that match.

MANDATORY workflow — you MUST call each tool, in order:

1. search_recent_songs(query) — Query a SPECIFIC RECENT ARTIST plus a SONG TITLE (never an album name), e.g. 'Sabrina Carpenter Espresso', 'Olivia Rodrigo vampire', 'Chappell Roan Good Luck Babe', 'Tate McRae greedy', 'Gracie Abrams Risk', 'Billie Eilish Birds of a Feather'. NEVER query an album name ('Olivia Rodrigo GUTS', 'Billie Eilish Hit Me Hard and Soft') — albums return setlist/liner-note pages with no lyrics. NEVER append a bare year ('Sabrina Carpenter 2024') or use vague queries ('best 2024 songs'). Each hit gives {artist, title, id}; from the results pick the ORIGINAL studio cut (skip entries labelled clean, demo, live, remix, or a translation) and carry its `id`.

2. fetch_lyrics(artist, title, song_id) — Call once per candidate, passing the `id` from the search hit, to retrieve and cache the lyrics.

3. extract_features(artist, title, song_id) — REQUIRED for every candidate, with the SAME id. This call computes the feature vector that lets the system rank songs against the user's profile. Without it, the recommendation pipeline returns nothing.

4. As soon as 4 candidates have been processed through extract_features, STOP searching and write the final summary. Mention each song you scored.

Be efficient — this loop runs live and every extra round is slow:
- Aim for ~4 searches total, each a DIFFERENT artist+song. Do not repeat a query you already ran.
- After each successful fetch_lyrics, call extract_features for that song before searching again, so you stop the moment you reach 4.

Hard rules:
- You MUST call extract_features for at least 3 songs before writing the final message. Do not skip step 3.
- Always pass the numeric `id` from search results to fetch_lyrics and extract_features — never guess or omit it.
- If a fetch returns nothing, skip that song and pick another id from the SAME search results before running a new search.
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
            label = f"{args.get('artist','?')} - {args.get('title','?')}"
            if name == "search_recent_songs" and isinstance(result, list):
                log(f"  -> {len(result)} candidates")
            elif name == "fetch_lyrics":
                if isinstance(result, str) and result:
                    log(f"  -> {len(result)} chars [{label}]")
                else:
                    log(f"  -> NO LYRICS [{label}] (id {args.get('song_id','?')})")
            elif name == "extract_features":
                if isinstance(result, dict) and "error" in result:
                    log(f"  -> SKIP [{label}]: {result['error']}")
                elif isinstance(result, dict) and candidate_sink is not None:
                    artist = result.pop("_artist", args.get("artist", "?"))
                    title = result.pop("_title", args.get("title", "?"))
                    candidate_sink.append({"artist": artist, "title": title, "features": result})
                    log(f"  -> features for {artist} - {title}")
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": json.dumps(result) if not isinstance(result, str) else result,
            })
    return "max steps reached"
