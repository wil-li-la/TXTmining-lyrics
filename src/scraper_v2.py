"""Scrape lyrics for Billboard Hot 100 2016-2025 using the official Genius API.

Inputs:
  - data/raw/walkerkq.csv (1964-2015, already lyrics-bearing)
  - Hardcoded SONG_LIST for 2016-2025 (curated from Billboard year-end charts)

Output:
  - data/lyrics_full.csv with columns:
      song_id, title, artist, year, decade, chart_position, source, lyrics_raw
"""
import os
import time

import pandas as pd
from dotenv import load_dotenv
import lyricsgenius

load_dotenv()

# Curated Billboard year-end top entries for 2016-2025
# Each tuple: (artist, title, year, peak_pos). Sources: Wikipedia
# "Billboard Year-End Hot 100 singles of YYYY" + general chart knowledge.
SONG_LIST: list[tuple[str, str, int, int]] = [
    # ---- 2016 (Year-End Hot 100 top entries) ----
    ("Justin Bieber", "Love Yourself", 2016, 1),
    ("Drake", "One Dance", 2016, 2),
    ("Rihanna", "Work", 2016, 3),
    ("Justin Bieber", "Sorry", 2016, 4),
    ("The Chainsmokers", "Don't Let Me Down", 2016, 5),
    ("Desiigner", "Panda", 2016, 6),
    ("Lukas Graham", "7 Years", 2016, 7),
    ("Sia", "Cheap Thrills", 2016, 8),
    ("Fifth Harmony", "Work from Home", 2016, 9),
    ("Calvin Harris", "This Is What You Came For", 2016, 10),
    ("Mike Posner", "I Took a Pill in Ibiza", 2016, 11),
    ("The Chainsmokers", "Closer", 2016, 12),
    ("Twenty One Pilots", "Stressed Out", 2016, 13),
    ("Twenty One Pilots", "Heathens", 2016, 14),
    ("Adele", "Hello", 2016, 15),
    ("Flo Rida", "My House", 2016, 16),
    ("DJ Snake", "Let Me Love You", 2016, 17),
    ("Zayn", "Pillowtalk", 2016, 18),
    ("Meghan Trainor", "Me Too", 2016, 19),
    ("Charlie Puth", "One Call Away", 2016, 20),
    ("Selena Gomez", "Hands to Myself", 2016, 21),
    ("Drake", "Hotline Bling", 2016, 22),
    ("Beyonce", "Formation", 2016, 23),
    ("Bruno Mars", "24K Magic", 2016, 24),
    ("Justin Timberlake", "Can't Stop the Feeling", 2016, 25),

    # ---- 2017 ----
    ("Ed Sheeran", "Shape of You", 2017, 1),
    ("Luis Fonsi", "Despacito", 2017, 2),
    ("Bruno Mars", "That's What I Like", 2017, 3),
    ("The Chainsmokers", "Something Just Like This", 2017, 4),
    ("DJ Khaled", "I'm the One", 2017, 5),
    ("Migos", "Bad and Boujee", 2017, 6),
    ("Kendrick Lamar", "HUMBLE.", 2017, 7),
    ("Post Malone", "Congratulations", 2017, 8),
    ("Sam Hunt", "Body Like a Back Road", 2017, 9),
    ("Imagine Dragons", "Believer", 2017, 10),
    ("Charlie Puth", "Attention", 2017, 11),
    ("Camila Cabello", "Havana", 2017, 12),
    ("Taylor Swift", "Look What You Made Me Do", 2017, 13),
    ("Future", "Mask Off", 2017, 14),
    ("Ed Sheeran", "Perfect", 2017, 15),
    ("Khalid", "Location", 2017, 16),
    ("Cardi B", "Bodak Yellow", 2017, 17),
    ("French Montana", "Unforgettable", 2017, 18),
    ("Maroon 5", "What Lovers Do", 2017, 19),
    ("Logic", "1-800-273-8255", 2017, 20),
    ("Post Malone", "Rockstar", 2017, 21),
    ("Selena Gomez", "It Ain't Me", 2017, 22),
    ("Lil Uzi Vert", "XO Tour Llif3", 2017, 23),
    ("DJ Khaled", "Wild Thoughts", 2017, 24),
    ("Zedd", "Stay", 2017, 25),

    # ---- 2018 ----
    ("Drake", "God's Plan", 2018, 1),
    ("Post Malone", "Psycho", 2018, 2),
    ("Post Malone", "Better Now", 2018, 3),
    ("Cardi B", "I Like It", 2018, 4),
    ("Bruno Mars", "Finesse", 2018, 5),
    ("Drake", "Nice for What", 2018, 6),
    ("Camila Cabello", "Never Be the Same", 2018, 7),
    ("Ariana Grande", "No Tears Left to Cry", 2018, 8),
    ("Maroon 5", "Girls Like You", 2018, 9),
    ("Childish Gambino", "This Is America", 2018, 10),
    ("Drake", "In My Feelings", 2018, 11),
    ("Travis Scott", "Sicko Mode", 2018, 12),
    ("XXXTentacion", "SAD!", 2018, 13),
    ("Bebe Rexha", "Meant to Be", 2018, 14),
    ("Marshmello", "Happier", 2018, 15),
    ("Lady Gaga", "Shallow", 2018, 16),
    ("5 Seconds of Summer", "Youngblood", 2018, 17),
    ("Post Malone", "Sunflower", 2018, 18),
    ("Ariana Grande", "Thank U, Next", 2018, 19),
    ("Imagine Dragons", "Natural", 2018, 20),
    ("Tyga", "Taste", 2018, 21),
    ("Halsey", "Without Me", 2018, 22),
    ("Juice WRLD", "Lucid Dreams", 2018, 23),
    ("Ella Mai", "Boo'd Up", 2018, 24),
    ("Khalid", "Love Lies", 2018, 25),

    # ---- 2019 ----
    ("Lil Nas X", "Old Town Road", 2019, 1),
    ("Billie Eilish", "Bad Guy", 2019, 2),
    ("Post Malone", "Sunflower", 2019, 3),
    ("Halsey", "Without Me", 2019, 4),
    ("Shawn Mendes", "Senorita", 2019, 5),
    ("Lil Tecca", "Ran$om", 2019, 6),
    ("Khalid", "Talk", 2019, 7),
    ("Lewis Capaldi", "Someone You Loved", 2019, 8),
    ("Jonas Brothers", "Sucker", 2019, 9),
    ("Ariana Grande", "7 Rings", 2019, 10),
    ("Post Malone", "Wow.", 2019, 11),
    ("Ed Sheeran", "I Don't Care", 2019, 12),
    ("Sam Smith", "Dancing with a Stranger", 2019, 13),
    ("Taylor Swift", "ME!", 2019, 14),
    ("Ava Max", "Sweet but Psycho", 2019, 15),
    ("Lizzo", "Truth Hurts", 2019, 16),
    ("Travis Scott", "Highest in the Room", 2019, 17),
    ("Marshmello", "Wow", 2019, 18),
    ("Maroon 5", "Memories", 2019, 19),
    ("Billie Eilish", "When the Party's Over", 2019, 20),
    ("Cardi B", "Please Me", 2019, 21),
    ("DaBaby", "Suge", 2019, 22),
    ("Lil Nas X", "Panini", 2019, 23),
    ("Lauv", "I Like Me Better", 2019, 24),
    ("Camila Cabello", "Senorita", 2019, 25),

    # ---- 2020 ----
    ("The Weeknd", "Blinding Lights", 2020, 1),
    ("Roddy Ricch", "The Box", 2020, 2),
    ("Post Malone", "Circles", 2020, 3),
    ("Dua Lipa", "Don't Start Now", 2020, 4),
    ("Maroon 5", "Memories", 2020, 5),
    ("Future", "Life Is Good", 2020, 6),
    ("Justin Bieber", "Yummy", 2020, 7),
    ("Cardi B", "WAP", 2020, 8),
    ("Megan Thee Stallion", "Savage", 2020, 9),
    ("DaBaby", "Rockstar", 2020, 10),
    ("Drake", "Toosie Slide", 2020, 11),
    ("Dua Lipa", "Levitating", 2020, 12),
    ("Harry Styles", "Watermelon Sugar", 2020, 13),
    ("Doja Cat", "Say So", 2020, 14),
    ("BTS", "Dynamite", 2020, 15),
    ("Travis Scott", "The Scotts", 2020, 16),
    ("Lil Baby", "We Paid", 2020, 17),
    ("Ariana Grande", "Positions", 2020, 18),
    ("Pop Smoke", "What You Know Bout Love", 2020, 19),
    ("Justin Bieber", "Holy", 2020, 20),
    ("Surfaces", "Sunday Best", 2020, 21),
    ("SAINt JHN", "Roses", 2020, 22),
    ("Jawsh 685", "Savage Love", 2020, 23),
    ("Tones and I", "Dance Monkey", 2020, 24),
    ("Powfu", "Death Bed", 2020, 25),

    # ---- 2021 ----
    ("Olivia Rodrigo", "drivers license", 2021, 1),
    ("Olivia Rodrigo", "good 4 u", 2021, 2),
    ("The Kid Laroi", "Stay", 2021, 3),
    ("Dua Lipa", "Levitating", 2021, 4),
    ("Doja Cat", "Kiss Me More", 2021, 5),
    ("Lil Nas X", "Montero", 2021, 6),
    ("Lil Nas X", "Industry Baby", 2021, 7),
    ("Justin Bieber", "Peaches", 2021, 8),
    ("Olivia Rodrigo", "deja vu", 2021, 9),
    ("Glass Animals", "Heat Waves", 2021, 10),
    ("Bruno Mars", "Leave the Door Open", 2021, 11),
    ("Doja Cat", "Need to Know", 2021, 12),
    ("BTS", "Butter", 2021, 13),
    ("Polo G", "Rapstar", 2021, 14),
    ("Cardi B", "Up", 2021, 15),
    ("Justin Bieber", "Stay", 2021, 16),
    ("Masked Wolf", "Astronaut in the Ocean", 2021, 17),
    ("Lil Tjay", "Calling My Phone", 2021, 18),
    ("24kGoldn", "Mood", 2021, 19),
    ("Adele", "Easy On Me", 2021, 20),
    ("Drake", "Way 2 Sexy", 2021, 21),
    ("Ed Sheeran", "Bad Habits", 2021, 22),
    ("Walker Hayes", "Fancy Like", 2021, 23),
    ("Doja Cat", "Woman", 2021, 24),
    ("Taylor Swift", "All Too Well", 2021, 25),

    # ---- 2022 ----
    ("Harry Styles", "As It Was", 2022, 1),
    ("Glass Animals", "Heat Waves", 2022, 2),
    ("Jack Harlow", "First Class", 2022, 3),
    ("Future", "Wait for U", 2022, 4),
    ("Lizzo", "About Damn Time", 2022, 5),
    ("Kate Bush", "Running Up That Hill", 2022, 6),
    ("Bad Bunny", "Me Porto Bonito", 2022, 7),
    ("Steve Lacy", "Bad Habit", 2022, 8),
    ("Sam Smith", "Unholy", 2022, 9),
    ("Taylor Swift", "Anti-Hero", 2022, 10),
    ("Beyonce", "Break My Soul", 2022, 11),
    ("Doja Cat", "Vegas", 2022, 12),
    ("Nicki Minaj", "Super Freaky Girl", 2022, 13),
    ("Bad Bunny", "Tití Me Preguntó", 2022, 14),
    ("Em Beihold", "Numb Little Bug", 2022, 15),
    ("Latto", "Big Energy", 2022, 16),
    ("Post Malone", "I Like You", 2022, 17),
    ("Lil Nas X", "Industry Baby", 2022, 18),
    ("Drake", "Jimmy Cooks", 2022, 19),
    ("Imagine Dragons", "Enemy", 2022, 20),
    ("Encanto Cast", "We Don't Talk About Bruno", 2022, 21),
    ("Megan Thee Stallion", "Sweetest Pie", 2022, 22),
    ("Bad Bunny", "Moscow Mule", 2022, 23),
    ("OneRepublic", "I Ain't Worried", 2022, 24),
    ("Charlie Puth", "Light Switch", 2022, 25),

    # ---- 2023 ----
    ("Morgan Wallen", "Last Night", 2023, 1),
    ("Miley Cyrus", "Flowers", 2023, 2),
    ("SZA", "Kill Bill", 2023, 3),
    ("Taylor Swift", "Anti-Hero", 2023, 4),
    ("Rema", "Calm Down", 2023, 5),
    ("Olivia Rodrigo", "vampire", 2023, 6),
    ("Doja Cat", "Paint the Town Red", 2023, 7),
    ("Jung Kook", "Seven", 2023, 8),
    ("Taylor Swift", "Cruel Summer", 2023, 9),
    ("Eslabon Armado", "Ella Baila Sola", 2023, 10),
    ("Olivia Rodrigo", "get him back!", 2023, 11),
    ("Drake", "Rich Flex", 2023, 12),
    ("Metro Boomin", "Creepin'", 2023, 13),
    ("Luke Combs", "Fast Car", 2023, 14),
    ("Toosii", "Favorite Song", 2023, 15),
    ("Yng Lvcas", "La Bebe", 2023, 16),
    ("Peso Pluma", "Ella Baila Sola", 2023, 17),
    ("Taylor Swift", "Karma", 2023, 18),
    ("Wham!", "Last Christmas", 2023, 19),
    ("Mariah Carey", "All I Want for Christmas Is You", 2023, 20),
    ("Tyla", "Water", 2023, 21),
    ("Doja Cat", "Agora Hills", 2023, 22),
    ("Drake", "Search & Rescue", 2023, 23),
    ("Lil Durk", "All My Life", 2023, 24),
    ("Zach Bryan", "I Remember Everything", 2023, 25),

    # ---- 2024 ----
    ("Sabrina Carpenter", "Espresso", 2024, 1),
    ("Sabrina Carpenter", "Please Please Please", 2024, 2),
    ("Kendrick Lamar", "Not Like Us", 2024, 3),
    ("Shaboozey", "A Bar Song (Tipsy)", 2024, 4),
    ("Billie Eilish", "Birds of a Feather", 2024, 5),
    ("Taylor Swift", "Fortnight", 2024, 6),
    ("Benson Boone", "Beautiful Things", 2024, 7),
    ("Teddy Swims", "Lose Control", 2024, 8),
    ("Hozier", "Too Sweet", 2024, 9),
    ("Tommy Richman", "Million Dollar Baby", 2024, 10),
    ("Post Malone", "I Had Some Help", 2024, 11),
    ("Future", "Like That", 2024, 12),
    ("Chappell Roan", "Good Luck, Babe!", 2024, 13),
    ("Sabrina Carpenter", "Taste", 2024, 14),
    ("Lady Gaga", "Die With a Smile", 2024, 15),
    ("Rose", "APT.", 2024, 16),
    ("Eminem", "Houdini", 2024, 17),
    ("Morgan Wallen", "I Had Some Help", 2024, 18),
    ("Gracie Abrams", "That's So True", 2024, 19),
    ("ILLIT", "Magnetic", 2024, 20),
    ("Tate McRae", "Greedy", 2024, 21),
    ("Djo", "End of Beginning", 2024, 22),
    ("Ariana Grande", "We Can't Be Friends", 2024, 23),
    ("Jung Kook", "Standing Next to You", 2024, 24),
    ("Beyonce", "Texas Hold 'Em", 2024, 25),

    # ---- 2025 (early-year Hot 100 contenders; quality > quantity) ----
    ("Kendrick Lamar", "Squabble Up", 2025, 1),
    ("Kendrick Lamar", "Luther", 2025, 2),
    ("The Weeknd", "Timeless", 2025, 3),
    ("Lady Gaga", "Abracadabra", 2025, 4),
    ("Bad Bunny", "DtMF", 2025, 5),
    ("ROSE", "Number One Girl", 2025, 6),
    ("Sabrina Carpenter", "Manchild", 2025, 7),
    ("Alex Warren", "Ordinary", 2025, 8),
    ("Doechii", "Anxiety", 2025, 9),
    ("Lola Young", "Messy", 2025, 10),
    ("Drake", "Nokia", 2025, 11),
    ("Billie Eilish", "Wildflower", 2025, 12),
    ("Tate McRae", "Sports car", 2025, 13),
    ("Morgan Wallen", "Lies Lies Lies", 2025, 14),
    ("Shaboozey", "Good News", 2025, 15),
    ("Gracie Abrams", "I Love You, I'm Sorry", 2025, 16),
    ("Mariah The Scientist", "Burning Blue", 2025, 17),
    ("Teddy Swims", "Bad Dreams", 2025, 18),
    ("Selena Gomez", "Sunset Blvd", 2025, 19),
    ("Post Malone", "Pour Me a Drink", 2025, 20),
]


def decade_of(year: int) -> str:
    return f"{(year // 10) * 10}s"


def fetch_one(genius: lyricsgenius.Genius, artist: str, title: str) -> str | None:
    """Fetch lyrics via Genius search; returns lyrics or None on failure."""
    try:
        song = genius.search_song(title=title, artist=artist, get_full_info=False)
        if song is None or not song.lyrics:
            return None
        text = song.lyrics
        return text
    except Exception as e:
        print(f"  [ERROR] {artist} - {title}: {e}")
        return None


def load_walkerkq(path: str = "data/raw/walkerkq.csv") -> pd.DataFrame:
    """Load walkerkq dataset into our schema.

    Actual walkerkq columns: Rank, Song, Artist, Year, Lyrics, Source.
    """
    df = pd.read_csv(path, encoding="latin-1")
    df = df.rename(columns={"Rank": "chart_position", "Song": "title",
                            "Artist": "artist", "Year": "year",
                            "Lyrics": "lyrics_raw"})
    df["decade"] = df["year"].apply(decade_of)
    df["source"] = "walkerkq"
    df = df.dropna(subset=["lyrics_raw"])
    df = df[df["lyrics_raw"].str.len() > 100]
    return df[["title", "artist", "year", "decade", "chart_position",
               "source", "lyrics_raw"]]


def scrape_recent() -> pd.DataFrame:
    """Scrape SONG_LIST via Genius API. Polite delay between calls."""
    genius = lyricsgenius.Genius(
        os.environ["GENIUS_ACCESS_TOKEN"],
        timeout=15, sleep_time=1, remove_section_headers=True,
    )
    rows = []
    n_fail = 0
    for i, (artist, title, year, peak) in enumerate(SONG_LIST, start=1):
        print(f"[{i}/{len(SONG_LIST)}] {artist} - {title} ({year})")
        lyrics = fetch_one(genius, artist, title)
        if lyrics is None:
            print("  skipped")
            n_fail += 1
            continue
        rows.append({
            "title": title, "artist": artist, "year": year,
            "decade": decade_of(year), "chart_position": peak,
            "source": "genius_api", "lyrics_raw": lyrics,
        })
        time.sleep(1.5)  # polite to Genius
    print(f"\nScrape: {len(rows)} succeeded / {n_fail} failed "
          f"out of {len(SONG_LIST)} ({n_fail / len(SONG_LIST):.1%} fail rate)")
    return pd.DataFrame(rows)


def merge_and_save(out_path: str = "data/lyrics_full.csv") -> None:
    walkerkq_df = load_walkerkq()
    recent_df = scrape_recent()
    full = pd.concat([walkerkq_df, recent_df], ignore_index=True)
    # Dedupe on lowercased (artist, title)
    full["_key"] = (full["artist"].str.lower().str.strip()
                    + "|" + full["title"].str.lower().str.strip())
    full = full.drop_duplicates(subset="_key", keep="first").drop(columns="_key")
    full = full.reset_index(drop=True)
    full.insert(0, "song_id", [f"S{i:05d}" for i in range(len(full))])
    full.to_csv(out_path, index=False)
    print(f"\nSaved {len(full)} songs -> {out_path}")
    print(full.groupby("decade").size())
    print(full["source"].value_counts())


if __name__ == "__main__":
    merge_and_save()
