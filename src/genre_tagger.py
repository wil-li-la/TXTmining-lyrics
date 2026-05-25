"""Tag each song with a coarse genre.

MusicBrainz was found empirically to return no tags for ~all songs in this
corpus (older 1965-2015 walkerkq entries don't have rich tags). So this
version skips MusicBrainz and batches OpenAI calls (20 songs per request)
for speed: ~250 calls * 1.5s = ~6 min for 4,869 songs.
"""
import json
import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GENRES = ["pop", "rap", "r&b", "rock", "country", "dance", "folk", "other"]
BATCH_SIZE = 20


def tag_batch(client: OpenAI, batch: list[dict]) -> list[str]:
    """Tag a batch of songs in a single OpenAI call. Returns list of genre labels in order."""
    enumerated = "\n".join(
        f"{i+1}. {row['artist']} - {row['title']}" for i, row in enumerate(batch)
    )
    prompt = (
        f"Classify the genre of each numbered song. Reply with a JSON array of {len(batch)} "
        f"strings, each one of: {', '.join(GENRES)}. If you don't recognize a song, use 'other'.\n\n"
        f"Songs:\n{enumerated}\n\n"
        f"Reply with ONLY the JSON array, like: [\"pop\",\"rock\",...]"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400, temperature=0,
        )
        content = resp.choices[0].message.content.strip()
        # Strip code fences if present
        if content.startswith("```"):
            content = content.strip("`").lstrip("json").strip()
        labels = json.loads(content)
        if not isinstance(labels, list) or len(labels) != len(batch):
            print(f"  [WARN] batch returned wrong shape: {labels!r}")
            return ["other"] * len(batch)
        return [(l if isinstance(l, str) and l.lower() in GENRES else "other") for l in (str(x).lower() for x in labels)]
    except Exception as e:
        print(f"  [batch ERROR] {e}")
        return ["other"] * len(batch)


def main(in_path: str = "data/lyrics_full.csv",
         out_path: str = "data/genre_tags.csv") -> None:
    df = pd.read_csv(in_path)
    client = OpenAI()

    if os.path.exists(out_path):
        done = pd.read_csv(out_path)
        done_ids = set(done["song_id"])
        print(f"  resuming: {len(done_ids)} already tagged")
    else:
        done = pd.DataFrame(columns=["song_id", "genre", "source"])
        done_ids = set()

    todo = df[~df["song_id"].isin(done_ids)].reset_index(drop=True)
    print(f"  {len(todo)} songs to tag in batches of {BATCH_SIZE}")

    rows = []
    for start in range(0, len(todo), BATCH_SIZE):
        batch_df = todo.iloc[start:start + BATCH_SIZE]
        batch = batch_df.to_dict("records")
        labels = tag_batch(client, batch)
        for row, label in zip(batch, labels):
            rows.append({"song_id": row["song_id"], "genre": label, "source": "openai_batch"})
        print(f"  [{start + len(batch)}/{len(todo)}] last: {batch[-1]['artist']} - {batch[-1]['title']} -> {labels[-1]}")
        if (start // BATCH_SIZE) % 5 == 4:  # checkpoint every 5 batches = 100 songs
            pd.concat([done, pd.DataFrame(rows)]).to_csv(out_path, index=False)

    out = pd.concat([done, pd.DataFrame(rows)])
    out.to_csv(out_path, index=False)
    print(f"\nTagged {len(out)} songs -> {out_path}")
    print(out["genre"].value_counts())


if __name__ == "__main__":
    main()
