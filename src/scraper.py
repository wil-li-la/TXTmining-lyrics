"""
scraper.py — Fetch lyrics from Genius for a curated Billboard song list.

Usage:
    python src/scraper.py

Output:
    data/lyrics.csv  (columns: title, artist, year, decade, lyrics)
"""

import csv
import os
import random
import re
import time

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Song list: 100 entries, 25 per decade (1990s–2020s)
# Each tuple: (title, artist, year, decade)
# ---------------------------------------------------------------------------

SONGS = [
    # 1990s
    ("Baby One More Time", "Britney Spears", 1999, "1990s"),
    ("Wannabe", "Spice Girls", 1996, "1990s"),
    ("No Scrubs", "TLC", 1999, "1990s"),
    ("Waterfalls", "TLC", 1995, "1990s"),
    ("I Will Always Love You", "Whitney Houston", 1992, "1990s"),
    ("Creep", "TLC", 1994, "1990s"),
    ("MMMBop", "Hanson", 1997, "1990s"),
    ("Livin La Vida Loca", "Ricky Martin", 1999, "1990s"),
    ("Genie In A Bottle", "Christina Aguilera", 1999, "1990s"),
    ("I Want It That Way", "Backstreet Boys", 1999, "1990s"),
    ("Un-Break My Heart", "Toni Braxton", 1996, "1990s"),
    ("Believe", "Cher", 1998, "1990s"),
    ("Kiss From A Rose", "Seal", 1995, "1990s"),
    ("End Of The Road", "Boyz II Men", 1992, "1990s"),
    ("Torn", "Natalie Imbruglia", 1997, "1990s"),
    ("Iris", "Goo Goo Dolls", 1998, "1990s"),
    ("Losing My Religion", "R.E.M.", 1991, "1990s"),
    ("Smells Like Teen Spirit", "Nirvana", 1991, "1990s"),
    ("Black Or White", "Michael Jackson", 1991, "1990s"),
    ("Gangsta's Paradise", "Coolio", 1995, "1990s"),
    ("Macarena", "Los Del Rio", 1996, "1990s"),
    ("Wonderwall", "Oasis", 1995, "1990s"),
    ("Return Of The Mack", "Mark Morrison", 1996, "1990s"),
    ("Semi-Charmed Life", "Third Eye Blind", 1997, "1990s"),
    ("Smooth", "Santana", 1999, "1990s"),

    # 2000s
    ("Crazy In Love", "Beyonce", 2003, "2000s"),
    ("In Da Club", "50 Cent", 2003, "2000s"),
    ("Hey Ya", "Outkast", 2003, "2000s"),
    ("Since U Been Gone", "Kelly Clarkson", 2004, "2000s"),
    ("Toxic", "Britney Spears", 2004, "2000s"),
    ("Yeah", "Usher", 2004, "2000s"),
    ("Umbrella", "Rihanna", 2007, "2000s"),
    ("Hips Don't Lie", "Shakira", 2006, "2000s"),
    ("Poker Face", "Lady Gaga", 2008, "2000s"),
    ("Single Ladies", "Beyonce", 2008, "2000s"),
    ("Beautiful Day", "U2", 2000, "2000s"),
    ("Lose Yourself", "Eminem", 2002, "2000s"),
    ("Complicated", "Avril Lavigne", 2002, "2000s"),
    ("Mr. Brightside", "The Killers", 2004, "2000s"),
    ("Gold Digger", "Kanye West", 2005, "2000s"),
    ("Hollaback Girl", "Gwen Stefani", 2005, "2000s"),
    ("Rehab", "Amy Winehouse", 2006, "2000s"),
    ("Bleeding Love", "Leona Lewis", 2007, "2000s"),
    ("Viva La Vida", "Coldplay", 2008, "2000s"),
    ("I Gotta Feeling", "Black Eyed Peas", 2009, "2000s"),
    ("Boom Boom Pow", "Black Eyed Peas", 2009, "2000s"),
    ("Love Story", "Taylor Swift", 2008, "2000s"),
    ("Low", "Flo Rida", 2007, "2000s"),
    ("Irreplaceable", "Beyonce", 2006, "2000s"),
    ("SexyBack", "Justin Timberlake", 2006, "2000s"),

    # 2010s
    ("Rolling In The Deep", "Adele", 2010, "2010s"),
    ("Someone Like You", "Adele", 2011, "2010s"),
    ("Call Me Maybe", "Carly Rae Jepsen", 2012, "2010s"),
    ("Happy", "Pharrell Williams", 2013, "2010s"),
    ("Uptown Funk", "Mark Ronson", 2014, "2010s"),
    ("Shape Of You", "Ed Sheeran", 2017, "2010s"),
    ("Despacito", "Luis Fonsi", 2017, "2010s"),
    ("Old Town Road", "Lil Nas X", 2019, "2010s"),
    ("Closer", "The Chainsmokers", 2016, "2010s"),
    ("Bad Guy", "Billie Eilish", 2019, "2010s"),
    ("Blank Space", "Taylor Swift", 2014, "2010s"),
    ("Shake It Off", "Taylor Swift", 2014, "2010s"),
    ("Thinking Out Loud", "Ed Sheeran", 2014, "2010s"),
    ("Royals", "Lorde", 2013, "2010s"),
    ("Wrecking Ball", "Miley Cyrus", 2013, "2010s"),
    ("Lean On", "Major Lazer", 2015, "2010s"),
    ("Sorry", "Justin Bieber", 2015, "2010s"),
    ("Havana", "Camila Cabello", 2017, "2010s"),
    ("God's Plan", "Drake", 2018, "2010s"),
    ("Thank U Next", "Ariana Grande", 2018, "2010s"),
    ("Shallow", "Lady Gaga", 2018, "2010s"),
    ("Sunflower", "Post Malone", 2018, "2010s"),
    ("Blinding Lights", "The Weeknd", 2019, "2010s"),
    ("Sicko Mode", "Travis Scott", 2018, "2010s"),
    ("7 Rings", "Ariana Grande", 2019, "2010s"),

    # 2020s
    ("Levitating", "Dua Lipa", 2020, "2020s"),
    ("Watermelon Sugar", "Harry Styles", 2020, "2020s"),
    ("drivers license", "Olivia Rodrigo", 2021, "2020s"),
    ("good 4 u", "Olivia Rodrigo", 2021, "2020s"),
    ("Stay", "The Kid Laroi", 2021, "2020s"),
    ("As It Was", "Harry Styles", 2022, "2020s"),
    ("Anti-Hero", "Taylor Swift", 2022, "2020s"),
    ("About Damn Time", "Lizzo", 2022, "2020s"),
    ("Heat Waves", "Glass Animals", 2022, "2020s"),
    ("Flowers", "Miley Cyrus", 2023, "2020s"),
    ("Kill Bill", "SZA", 2023, "2020s"),
    ("Vampire", "Olivia Rodrigo", 2023, "2020s"),
    ("Cruel Summer", "Taylor Swift", 2023, "2020s"),
    ("Last Night", "Morgan Wallen", 2023, "2020s"),
    ("Paint The Town Red", "Doja Cat", 2023, "2020s"),
    ("Espresso", "Sabrina Carpenter", 2024, "2020s"),
    ("Fortnight", "Taylor Swift", 2024, "2020s"),
    ("Beautiful Things", "Benson Boone", 2024, "2020s"),
    ("Lose Control", "Teddy Swims", 2024, "2020s"),
    ("Birds Of A Feather", "Billie Eilish", 2024, "2020s"),
    ("A Bar Song", "Shaboozey", 2024, "2020s"),
    ("Not Like Us", "Kendrick Lamar", 2024, "2020s"),
    ("Die With A Smile", "Lady Gaga", 2024, "2020s"),
    ("Taste", "Sabrina Carpenter", 2024, "2020s"),
    ("APT", "Rose", 2024, "2020s"),
]

# ---------------------------------------------------------------------------
# URL builder
# ---------------------------------------------------------------------------

def _make_url(artist: str, title: str) -> str:
    """Build a Genius URL for a given artist and song title.

    Pattern: https://genius.com/Artist-name-Song-title-lyrics
    """
    combined = f"{artist} {title}"
    combined = combined.replace("&", "and")
    combined = re.sub(r"[^\w\s-]", "", combined)
    combined = re.sub(r"\s+", "-", combined.strip())
    return f"https://genius.com/{combined}-lyrics"


# ---------------------------------------------------------------------------
# Lyrics fetcher
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def fetch_lyrics(artist: str, title: str) -> str | None:
    """Fetch lyrics for a song from Genius.

    Returns the lyrics string, or None if fetching/parsing fails.
    """
    url = _make_url(artist, title)
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"    [ERROR] HTTP request failed for {artist} - {title}: {exc}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Genius stores lyrics in <div data-lyrics-container="true"> elements.
    containers = soup.find_all("div", attrs={"data-lyrics-container": "true"})
    if not containers:
        print(f"    [WARN] Lyrics not found for {artist} - {title} ({url})")
        return None

    lines = []
    for container in containers:
        for br in container.find_all("br"):
            br.replace_with("\n")
        lines.append(container.get_text())

    text = "\n".join(lines).strip()
    # Remove section headers like [Verse 1], [Chorus], etc.
    text = re.sub(r"\[.*?\]", "", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text if len(text) > 100 else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "lyrics.csv")


def main() -> None:
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)

    total = len(SONGS)
    success_count = 0
    fail_count = 0

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["title", "artist", "year", "decade", "lyrics"]
        )
        writer.writeheader()

        for i, (title, artist, year, decade) in enumerate(SONGS, start=1):
            print(f"[{i:>3}/{total}] Fetching: {artist} — {title} ({year})")
            lyrics = fetch_lyrics(artist, title)

            if lyrics:
                success_count += 1
                print(f"    OK ({len(lyrics)} chars)")
            else:
                fail_count += 1

            writer.writerow(
                {
                    "title": title,
                    "artist": artist,
                    "year": year,
                    "decade": decade,
                    "lyrics": lyrics or "",
                }
            )

            # Polite delay between requests (skip after the last song)
            if i < total:
                delay = random.uniform(5, 10)
                print(f"    Waiting {delay:.1f}s …")
                time.sleep(delay)

    print(
        f"\nDone. {success_count}/{total} songs fetched successfully "
        f"({fail_count} failed)."
    )
    print(f"Results saved to: {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
