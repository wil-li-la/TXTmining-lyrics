"""Download external datasets."""
import os
import urllib.request

WALKERKQ_URL = "https://raw.githubusercontent.com/walkerkq/musiclyrics/master/billboard_lyrics_1964-2015.csv"
# Original Ghent URL is dead (404). Springer hosts the same dataset as
# supplementary material to Brysbaert et al. 2014 (BRM 46:904-911).
BRYSBAERT_URL = "http://crr.ugent.be/papers/Concreteness_ratings_Brysbaert_et_al_BRM.txt"
BRYSBAERT_XLSX_URL = (
    "https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-013-0403-5/"
    "MediaObjects/13428_2013_403_MOESM1_ESM.xlsx"
)

def download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"  exists: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def download_brysbaert(dest: str = "data/raw/brysbaert.txt") -> None:
    """Download the Brysbaert concreteness norms.

    The original Ghent mirror is dead, so we fall back to the Springer-hosted
    supplementary xlsx and convert it to a tab-separated text file matching
    the historical Ghent schema (Word, Bigram, Conc.M, Conc.SD, ...).
    """
    if os.path.exists(dest):
        print(f"  exists: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    try:
        print(f"  trying primary: {BRYSBAERT_URL}")
        urllib.request.urlretrieve(BRYSBAERT_URL, dest)
        return
    except Exception as e:
        print(f"  primary failed ({e}); falling back to Springer xlsx")

    import pandas as pd  # local import to avoid hard dep for callers
    xlsx_path = dest + ".xlsx"
    print(f"  downloading {BRYSBAERT_XLSX_URL}")
    urllib.request.urlretrieve(BRYSBAERT_XLSX_URL, xlsx_path)
    df = pd.read_excel(xlsx_path)
    df.to_csv(dest, sep="\t", index=False)
    os.remove(xlsx_path)
    print(f"  converted xlsx -> {dest} ({len(df)} rows)")


def main() -> None:
    download(WALKERKQ_URL, "data/raw/walkerkq.csv")
    download_brysbaert("data/raw/brysbaert.txt")


if __name__ == "__main__":
    main()
