"""Analysis-only inventory of a 2020 Attapady Kurumba numeral page.

No installed CSV is produced: a reuse licence for this post-2019 page has not
been identified. Supply a locally downloaded Kurumba-Attapady.htm to reproduce
the audit, e.g. from https://lingweb.eva.mpg.de/channumerals/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup


PACKAGE = Path(__file__).resolve().parent
SHA256 = "447111e36ae5c890bdaf5cf0b29fd73f2ca8125276f485bd3a62262dc314c990"
MARKER = re.compile(r"(?<!\d)(\d[\d ]*)\.")
REVIEW = {1, 2, 15, 17, 60, 80, 500, 700}


def inventory(html: Path) -> list[dict]:
    raw = html.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("The Attapady Kurumba page differs from the 2026-09-25 snapshot")
    tables = BeautifulSoup(raw, "html.parser").find_all("table")
    if len(tables) != 4 or len(tables[1].find_all("tr")) != 20:
        raise ValueError("Unexpected table layout")
    out = []
    for row_index, tr in enumerate(tables[1].find_all("tr"), 1):
        cells = tr.find_all("td")
        if len(cells) != 2:
            raise ValueError(f"Row {row_index} does not have two cells")
        for column, td in enumerate(cells, 1):
            cell = re.sub(r"\s+", " ", td.get_text(" ", strip=True)).strip()
            matches = list(MARKER.finditer(cell))
            if not matches or matches[0].start() != 0:
                raise ValueError(f"No initial numeral in cell {row_index}/{column}")
            for item_index, match in enumerate(matches):
                end = matches[item_index + 1].start() if item_index + 1 < len(matches) else len(cell)
                number = int(match.group(1).replace(" ", ""))
                form = cell[match.end():end].strip().strip(",").strip()
                out.append({
                    "number": number,
                    "source_cell": cell,
                    "source_item": cell[match.start():end].strip(),
                    "source_form": form,
                    "table_row": row_index,
                    "table_column": column,
                    "item_in_cell": item_index + 1,
                    "status": "held_pending_reuse_review" if form else "source_blank",
                    "transcription_review": "internal spacing/hyphenation" if number in REVIEW else "",
                    "source_lect": "Attapady Kurumba (Pal Kurumba)",
                    "proposed_language_id": "PaluKurumba",
                    "source_cell_key": f"bsi2020attapadykurumba:number:{number}",
                })
    out.sort(key=lambda row: row["number"])
    if len(out) != 46 or len({row["number"] for row in out}) != 46:
        raise ValueError("Expected 46 explicit numbered prompts")
    if {row["number"] for row in out if not row["source_form"]} != set(range(21, 30)):
        raise ValueError("Changed blank-cell inventory")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    parser.add_argument("--write-audit", action="store_true")
    args = parser.parse_args()
    rows = inventory(args.html)
    print(f"{len(rows)} numbered prompts; 37 held readings; 9 blank cells")
    if args.write_audit:
        with (PACKAGE / "analysis-audit.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
