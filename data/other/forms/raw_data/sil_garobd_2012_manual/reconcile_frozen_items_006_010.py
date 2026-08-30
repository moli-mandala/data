#!/usr/bin/env python3
"""Compare the frozen items 6-10 manual ledger with the legacy audit.

This script is intentionally downstream of manual transcription. It refuses to
run unless the cell ledger has its frozen SHA-256. Legacy values are copied only
into an audit comparison and never feed back into the manual generator or ledger.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
CELLS = HERE / "manual_chunks" / "items_006_010_cells.tsv"
LEGACY_AUDIT = HERE.parent / "20260826-sil-garobd-audit.csv"
OUTPUT = HERE / "manual_chunks" / "items_006_010_reconciliation.tsv"
FROZEN_CELLS_SHA256 = "5aecba4aae2f5edbb8422c797a72815529ac12d755d985e900ae2dd83db6b337"
FIELDS = [
    "Item", "Site_Code", "Similarity_Group", "Response_Order",
    "Manual_Transcription", "Manual_Status", "Legacy_Raw_Form", "Legacy_Status",
    "Legacy_Reason", "Exact_Codepoint_Equal", "Reconciliation_Disposition",
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
    assert digest(CELLS) == FROZEN_CELLS_SHA256
    manual = read_csv(CELLS, delimiter="\t")
    legacy_rows = [
        row for row in read_csv(LEGACY_AUDIT)
        if 6 <= int(row["Item"]) <= 10
    ]
    assert len(legacy_rows) == 88
    legacy: defaultdict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in legacy_rows:
        legacy[(row["Item"], row["Site_Code"], row["Group"])].append(row)

    out: list[dict[str, str]] = []
    consumed: Counter[tuple[str, str, str]] = Counter()
    for cell in manual:
        forms = cell["Manual_Transcription"].split(" | ") if cell["Manual_Transcription"] else [""]
        groups = cell["Similarity_Groups"].split("|")
        assert len(forms) == len(groups)
        for order, (form, group) in enumerate(zip(forms, groups, strict=True), start=1):
            key = (cell["Item"], cell["Site_Code"], group)
            prior = legacy[key][consumed[key]]
            consumed[key] += 1
            if cell["Review_Status"] == "source_blank":
                disposition = "manual source blank; legacy audit also records a printed gap"
            elif prior["Status"] == "excluded":
                disposition = "manually recovered formerly excluded glyph sequence"
            elif form == prior["Raw_Form"]:
                disposition = "legacy exact comparison match (audit-only; not verification)"
            else:
                disposition = "legacy differs at codepoint level (audit-only; manual unchanged)"
            out.append({
                "Item": cell["Item"],
                "Site_Code": cell["Site_Code"],
                "Similarity_Group": group,
                "Response_Order": str(order),
                "Manual_Transcription": form,
                "Manual_Status": cell["Review_Status"],
                "Legacy_Raw_Form": prior["Raw_Form"],
                "Legacy_Status": prior["Status"],
                "Legacy_Reason": prior["Reason"],
                "Exact_Codepoint_Equal": "yes" if form == prior["Raw_Form"] else "no",
                "Reconciliation_Disposition": disposition,
                "Frozen_Manual_Cells_SHA256": FROZEN_CELLS_SHA256,
                "Independence_Note": INDEPENDENCE_NOTE,
            })
    assert len(out) == 88
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
