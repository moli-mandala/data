"""Print reproducible random CDIAL source/parse comparisons for manual review."""

from __future__ import annotations

import argparse
import csv
import pickle
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
from tags import extract_tags


def source_entries(pages: list[str]) -> dict[str, list[str]]:
    entries: dict[str, list[str]] = defaultdict(list)
    for page in pages:
        soup = BeautifulSoup(page, "html.parser")
        for node in soup.find_all("hw"):
            number = node.find("number")
            if number:
                entries[number.get_text(strip=True)].append(str(node))
    return entries


def compact_source(fragments: list[str]) -> str:
    chunks = []
    for fragment in fragments:
        soup = BeautifulSoup(fragment, "html.parser")
        for br in soup.find_all("br"):
            br.replace_with("\n")
        chunks.append(re.sub(r"[ \t]+", " ", soup.get_text()).strip())
    return "\n--- addendum/continuation ---\n".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ids", help="comma-separated IDs instead of a random sample")
    args = parser.parse_args()

    with (HERE / "cdial.pickle").open("rb") as handle:
        sources = source_entries(pickle.load(handle))

    parsed: dict[str, list[list[str]]] = defaultdict(list)
    with (HERE / "cdial.csv").open(newline="") as handle:
        for row in csv.reader(handle):
            parsed[row[1]].append(row)

    if args.ids:
        selected = [value.strip() for value in args.ids.split(",") if value.strip()]
    else:
        population = sorted(set(sources) & set(parsed))
        selected = random.Random(args.seed).sample(population, args.count)

    for number in selected:
        print(f"\n{'=' * 20} CDIAL {number} {'=' * 20}")
        print("SOURCE:")
        print(compact_source(sources[number]))
        print("\nPARSED: language | form | gloss | tags | residual notes | source | cognateset")
        for row in parsed[number]:
            tags, residual = extract_tags(row[6], language_id=row[0])
            print(" | ".join([row[0], row[2], row[3], tags, residual, row[7], row[8]]))


if __name__ == "__main__":
    main()
