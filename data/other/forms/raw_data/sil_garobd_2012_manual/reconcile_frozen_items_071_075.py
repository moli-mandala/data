#!/usr/bin/env python3
"""Compare frozen items 71-75 manual ledgers with audit-only legacy data."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
LINES = HERE / "manual_chunks" / "items_071_075_lines.tsv"
CELLS = HERE / "manual_chunks" / "items_071_075_cells.tsv"
LEGACY_AUDIT = HERE.parent / "20260826-sil-garobd-audit.csv"
OUTPUT = HERE / "manual_chunks" / "items_071_075_reconciliation.tsv"
FROZEN_LINES_SHA256 = "1e2cd3d7b78ef0b3ae75dcd66a9c2ff37ab36fc1fa18a4232248f511aba24d07"
FROZEN_CELLS_SHA256 = "70bcde80a33a1e42596edc23ada60acb4aa1af5143770f3caf18aa871f58a0bf"
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
    legacy_rows = [row for row in read_csv(LEGACY_AUDIT) if 71 <= int(row["Item"]) <= 75]
    assert len(legacy_rows) == 97
    legacy: defaultdict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in legacy_rows:
        legacy[(row["Item"], row["Site_Code"], row["Group"])].append(row)

    consumed: Counter[tuple[str, str, str]] = Counter()
    out: list[dict[str, str]] = []
    for line in lines:
        for code in line["Bracket_Codes"]:
            key = (line["Item"], code, line["Similarity_Group"])
            candidates = legacy[key]
            prior = candidates[consumed[key]] if consumed[key] < len(candidates) else None
            if prior is not None:
                consumed[key] += 1
            form = line["Manual_Transcription"]
            if prior is None:
                disposition = "manual source record absent from legacy audit"
            elif line["Line_Status"] == "source_blank":
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
                "Legacy_Raw_Form": "" if prior is None else prior["Raw_Form"],
                "Legacy_Status": "missing" if prior is None else prior["Status"],
                "Legacy_Reason": "no matching legacy audit record" if prior is None else prior["Reason"],
                "Exact_Codepoint_Equal": "yes" if prior is not None and form == prior["Raw_Form"] else "no",
                "Reconciliation_Disposition": disposition,
                "Frozen_Manual_Lines_SHA256": FROZEN_LINES_SHA256,
                "Frozen_Manual_Cells_SHA256": FROZEN_CELLS_SHA256,
                "Independence_Note": INDEPENDENCE_NOTE,
            })
    assert len(out) == 98
    assert Counter(row["Legacy_Status"] for row in out)["missing"] == 1
    assert [(row["Item"], row["Site_Code"], row["Similarity_Group"]) for row in out if row["Legacy_Status"] == "missing"] == [("72", "j", "1")]
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
