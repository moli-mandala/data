"""Create a provisional per-head review grid for the Kor–English reverse index.

OCR and string matching are triage aids only; review_status defaults to pending.
"""

import csv
import difflib
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def plain(value):
    value = unicodedata.normalize("NFD", value.casefold())
    return re.sub(r"[^a-z0-9]", "", "".join(c for c in value if unicodedata.category(c) != "Mn"))


def main():
    forward = []
    for file in ("reviewed_inventory.tsv", "remaining_inventory.tsv"):
        with (ROOT / file).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                forward.append((row["printed_response"], row["gloss"]))
    with (ROOT / "held_resolution.tsv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["form"]:
                forward.append((row["form"], row["gloss"]))
    page = column = None
    parsed = []
    for raw in (ROOT / "reverse_column_ocr.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("# printed p"):
            match = re.fullmatch(r"# printed p(\d+), (left|right) column", line)
            page, column = int(match[1]), match[2]
        elif not line:
            continue
        elif ", " in line:
            head, gloss = line.split(", ", 1)
            parsed.append([page, column, head, gloss, line])
        elif parsed:
            parsed[-1][3] += " " + line
            parsed[-1][4] += " " + line
    output = []
    counts = {}
    for page, column, head, gloss, raw in parsed:
        counts[(page, column)] = counts.get((page, column), 0) + 1
        key = plain(head)
        candidates = sorted(
            forward,
            key=lambda pair: max(
                difflib.SequenceMatcher(None, key, plain(segment)).ratio()
                for segment in re.split(r"\s+or\s+|\s+and\s+", pair[0])
            ),
            reverse=True,
        )[:2]
        output.append((page, column, counts[(page, column)], head, gloss, raw,
                       candidates[0][0], candidates[0][1], candidates[1][0],
                       candidates[1][1], "", "", "pending", ""))
    with (ROOT / "reverse_inventory_candidates.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("page", "column", "ordinal", "ocr_head", "ocr_gloss", "ocr_line",
                         "closest_forward_response", "closest_forward_gloss",
                         "second_forward_response", "second_forward_gloss",
                         "reviewed_form", "reviewed_gloss", "review_status", "review_note"))
        writer.writerows(output)
    print(f"{len(output)} provisional OCR heads; all require visual review")


if __name__ == "__main__":
    main()
