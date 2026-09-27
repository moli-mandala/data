"""Provisional exact-form screen of Norton’s reverse index OCR.

This is a triage aid, not a visual source audit or an installation script.
"""

import csv
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def plain(value):
    value = unicodedata.normalize("NFD", value.casefold())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z0-9]", "", value)


def main():
    forward = []
    for name in ("reviewed_inventory.tsv", "remaining_inventory.tsv"):
        with (ROOT / name).open(encoding="utf-8", newline="") as handle:
            forward.extend(csv.DictReader(handle, delimiter="\t"))
    forward_strings = [(plain(row["printed_response"]), row["gloss"]) for row in forward]
    page = side = None
    rows = []
    for raw in (ROOT / "reverse_ocr.txt").read_text(encoding="utf-8").splitlines():
        if raw.startswith("# printed p"):
            match = re.match(r"# printed p(\d+), (left|right) column", raw)
            page, side = int(match[1]), match[2]
            continue
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        if ", " in raw:
            head, gloss = raw.split(", ", 1)
        else:
            head, gloss = raw, ""
        key = plain(head)
        matches = [g for response, g in forward_strings if len(key) >= 3 and key in response]
        rows.append((page, side, head, gloss, "yes" if matches else "no", "; ".join(matches[:3]), raw))
    with (ROOT / "reverse_ocr_screen.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("page", "column", "ocr_head", "ocr_gloss", "forward_string_match", "forward_glosses", "ocr_line"))
        writer.writerows(rows)
    print(f"{len(rows)} OCR lines; {sum(row[4] == 'no' for row in rows)} without a forward string match; visually review all")


if __name__ == "__main__":
    main()
