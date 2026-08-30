#!/usr/bin/env python3
"""Compare frozen items 51-55 manual ledgers with audit-only legacy data."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
LINES = HERE / "manual_chunks" / "items_051_055_lines.tsv"
CELLS = HERE / "manual_chunks" / "items_051_055_cells.tsv"
LEGACY_AUDIT = HERE.parent / "20260826-sil-garobd-audit.csv"
OUTPUT = HERE / "manual_chunks" / "items_051_055_reconciliation.tsv"
FROZEN_LINES_SHA256 = "4342065d879483cc2e6197f14122c2cfe94638f8db1252835e413e01e3263e89"
FROZEN_CELLS_SHA256 = "5158ca7583f94121b11c0d98427a7e7b4f7b0b9f6d23beb8c0fda94e05c0ceec"
FIELDS = [
    "Item", "Site_Code", "Similarity_Group", "Source_Line_ID",
    "Manual_Transcription", "Manual_Line_Status", "Cell_Review_Status",
    "Legacy_Raw_Form", "Legacy_Status", "Legacy_Reason", "Exact_Codepoint_Equal",
    "Reconciliation_Disposition", "Frozen_Manual_Lines_SHA256",
    "Frozen_Manual_Cells_SHA256", "Independence_Note",
]
INDEPENDENCE_NOTE = (
    "legacy comparison performed only after ledger freeze; legacy data did not "
    "supply, alter, or verify the manual reading"
)


def read_csv(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconciliation_rows() -> list[dict[str, str]]:
    assert digest(LINES) == FROZEN_LINES_SHA256
    assert digest(CELLS) == FROZEN_CELLS_SHA256
    lines = read_csv(LINES, delimiter="\t")
    cells = read_csv(CELLS, delimiter="\t")
    cell_status = {(row["Item"], row["Site_Code"]): row["Review_Status"] for row in cells}
    legacy_rows = [row for row in read_csv(LEGACY_AUDIT) if 51 <= int(row["Item"]) <= 55]
    assert len(legacy_rows) == 93
    legacy: defaultdict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in legacy_rows:
        legacy[(row["Item"], row["Site_Code"], row["Group"])].append(row)

    consumed: Counter[tuple[str, str, str]] = Counter()
    out: list[dict[str, str]] = []
    for line in lines:
        for code in line["Bracket_Codes"]:
            key = (line["Item"], code, line["Similarity_Group"])
            prior = legacy[key][consumed[key]]
            consumed[key] += 1
            form = line["Manual_Transcription"]
            if line["Line_Status"] == "source_blank":
                disposition = "manual source blank; legacy audit also records a printed gap"
            elif prior["Status"] == "excluded":
                disposition = "manually recovered formerly excluded glyph sequence"
            elif form == prior["Raw_Form"]:
                disposition = "legacy exact comparison match (audit-only; not verification)"
            else:
                disposition = "legacy differs at codepoint level (audit-only; manual unchanged)"
            out.append({
                "Item": line["Item"], "Site_Code": code,
                "Similarity_Group": line["Similarity_Group"], "Source_Line_ID": line["Line_ID"],
                "Manual_Transcription": form, "Manual_Line_Status": line["Line_Status"],
                "Cell_Review_Status": cell_status[(line["Item"], code)],
                "Legacy_Raw_Form": prior["Raw_Form"], "Legacy_Status": prior["Status"],
                "Legacy_Reason": prior["Reason"],
                "Exact_Codepoint_Equal": "yes" if form == prior["Raw_Form"] else "no",
                "Reconciliation_Disposition": disposition,
                "Frozen_Manual_Lines_SHA256": FROZEN_LINES_SHA256,
                "Frozen_Manual_Cells_SHA256": FROZEN_CELLS_SHA256,
                "Independence_Note": INDEPENDENCE_NOTE,
            })
    assert len(out) == 93
    assert consumed == Counter({key: len(rows) for key, rows in legacy.items()})
    return out


def main() -> None:
    rows = reconciliation_rows()
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} post-freeze comparisons to {OUTPUT}")


if __name__ == "__main__":
    main()
