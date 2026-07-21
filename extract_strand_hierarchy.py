"""Recover Strand lexicon hierarchy using the IDs assigned by strand.py.

This is an evidence-extraction helper: it does not modify the source CSVs.
It downloads the same alphabet pages and records every source head together
with its nearest parent, preserving the original ``nN`` ID sequence.
"""

import csv
import re
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

BASE_URL = (
    "http://nuristan.info/Nuristani/Nuristani/Nuristani/"
    "NuristaniLanguage/Lexicon/alph-{char}.html"
)
CATEGORIES = [
    "p", "b", "bAsp", "f", "v", "w", "m", "uFrn", "u", "o", "oFrn", "uTns", "oTns",
    "cDen", "zDen", "t", "d", "dAsp", "s", "z", "l", "lVls", "n", "cRet", "jRet",
    "tRet", "dRet", "dRetAsp", "sRet", "zRet", "r", "rFlp", "lBak", "rApx", "nApx",
    "nRet", "rVoc", "cLam", "jLam", "jLamAsp", "sLam", "zLam", "y", "i", "e", "aLam",
    "iTns", "eTns", "kPal", "gPal", "gPalAsp", "k", "g", "gAsp", "x", "gSpi", "nasVel",
    "iBak", "a", "aOpn", "aRnd", "kLab", "gLab", "gLabAsp", "nas", "q", "hPhyr", "Ayn",
    "AgltStp", "h", "hPglt", "hPgltRnd",
]
CACHE_DIR = Path(".cache/strand")
OUTPUT = Path("data/strand_hierarchy.csv")


def page(char):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"alph-{char}.html"
    if not path.exists():
        request = Request(BASE_URL.format(char=char), headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request) as response:
            path.write_bytes(response.read())
    return BeautifulSoup(path.read_bytes(), "html5lib")


def main():
    rows = []
    counter = 0
    stack = []

    for char in CATEGORIES:
        try:
            soup = page(char)
        except HTTPError:
            continue

        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                if not (row.find(class_="lng1") or row.find(class_="lng2")):
                    continue
                cells = row.find_all("td")
                comment = cells[-1].find(class_="mid")
                text = cells[-1].get_text().replace("\n", "")
                definitions = re.findall(r"‘(.*?)’", text)
                level = int(cells[0].get("colspan", 1) or 1 if row.find(class_="lng2") else 0)
                while stack and level <= stack[-1]["level"]:
                    stack.pop()

                turner = None
                if comment:
                    match = re.search(r"T\. (\d+(?:\.\d+)?)", comment.get_text())
                    if match:
                        turner = match.group(1)
                if not turner:
                    turner = next((item["id"] for item in stack if not item["id"].startswith("n")), None)
                if not turner:
                    counter += 1

                item = {
                    "source_index": len(rows) + 1,
                    "id": turner or f"n{counter}",
                    "parent_index": stack[-1]["source_index"] if stack else "",
                    "parent_id": stack[-1]["id"] if stack else "",
                    "level": level,
                    "language_id": cells[-2].find("em").get_text(),
                    "form": cells[-1].find("em").get_text(),
                    "gloss": definitions[0] if definitions else "",
                    "comment": comment.get_text() if comment else "",
                    "page": char,
                }
                rows.append(item)
                stack.append(item)

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} source heads to {OUTPUT}; final generated ID n{counter}")


if __name__ == "__main__":
    main()
