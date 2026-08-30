#!/usr/bin/env python3
"""Reconcile five 1989 Bhumij lists republished in Varenkamp 2024.

The two reports reproduce the same elicitation events but use materially
different publication transcription conventions. Identity is established by
the metadata, not by forcing the two diplomatic transcriptions to agree.
"""

import csv
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
HO = HERE.parent / "sil_ho_2024" / "staged_audit.tsv"
REGISTRY = HERE / "overlap_registry.tsv"
OUT = HERE / "ho_2024_overlap_reconciliation.tsv"
FIELDS = [
    "Item", "Gloss", "Bhumij_Site_Code", "Ho2024_Site_Code", "Locality",
    "Elicitation_Date", "Durable_List_ID", "Durable_Cell_ID",
    "Bhumij_2015_Status", "Bhumij_2015_Forms", "Bhumij_2015_Cognate_Labels",
    "Bhumij_2015_Citation", "Ho_2024_Status", "Ho_2024_Transcription",
    "Ho_2024_Citation", "Status_Parity", "Representation_Comparison",
    "Canonical_Publication", "Disposition",
]


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def strip_similarity_labels(text):
    value = re.sub(r"(^|,\s*)\d+(?=\s|,|\()\s*", r"\1", text)
    return re.sub(r"^,\s*", "", value).strip()


def load_bhumij():
    rows = []
    for path in sorted((HERE / "manual_chunks").glob("items_*_hand_keyed.tsv")):
        rows.extend(read_tsv(path))
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert len(rows) == 3780 and len(by_key) == 3780
    return by_key


def build_rows():
    registry = read_tsv(REGISTRY)
    assert len(registry) == 5
    assert len({row["Bhumij_Site_Code"] for row in registry}) == 5
    assert len({row["Ho2024_Site_Code"] for row in registry}) == 5
    bhumij = load_bhumij()
    ho_rows = read_tsv(HO)
    ho = {(row["Item"], row["Site_Code"]): row for row in ho_rows}
    output = []
    for spec in registry:
        for item in range(1, 211):
            key_b = (str(item), spec["Bhumij_Site_Code"])
            key_h = (str(item), spec["Ho2024_Site_Code"])
            primary, later = bhumij[key_b], ho[key_h]
            primary_blank = primary["Review_Status"] == "source_blank"
            later_blank = later["Review_Status"] == "blank"
            status_parity = "yes" if primary_blank == later_blank else "no"
            later_form = strip_similarity_labels(later["Manual_Transcription"])
            if primary_blank and later_blank:
                comparison = "blank-parity"
            elif primary["Manual_Transcription"] == later_form:
                comparison = "unicode-exact-after-label-removal"
            else:
                comparison = "publication-transcription-differs"
            row = {
                "Item": str(item), "Gloss": primary["Gloss"],
                "Bhumij_Site_Code": spec["Bhumij_Site_Code"],
                "Ho2024_Site_Code": spec["Ho2024_Site_Code"],
                "Locality": spec["Locality"],
                "Elicitation_Date": spec["Elicitation_Date"],
                "Durable_List_ID": spec["Durable_List_ID"],
                "Durable_Cell_ID": f"{spec['Durable_List_ID']}-i{item:03d}",
                "Bhumij_2015_Status": primary["Review_Status"],
                "Bhumij_2015_Forms": primary["Manual_Transcription"],
                "Bhumij_2015_Cognate_Labels": primary["Source_Cognate_Labels"],
                "Bhumij_2015_Citation": (
                    "baileymaggard2015bhumij[Appendix B.3, printed p. "
                    f"{primary['Printed_Page']}, item {item}, list "
                    f"{spec['Bhumij_Site_Code']}]"
                ),
                "Ho_2024_Status": later["Review_Status"],
                "Ho_2024_Transcription": later_form,
                "Ho_2024_Citation": later["Citation"],
                "Status_Parity": status_parity,
                "Representation_Comparison": comparison,
                "Canonical_Publication": spec["Canonical_Publication"],
                "Disposition": (
                    "install-primary-bhumij-2015; "
                    "exclude-ho-2024-same-elicitation-republication"
                ),
            }
            assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
            output.append(row)
    assert len(output) == 1050
    assert all(row["Status_Parity"] == "yes" for row in output)
    counts = {
        label: sum(row["Representation_Comparison"] == label for row in output)
        for label in {
            "blank-parity", "unicode-exact-after-label-removal",
            "publication-transcription-differs",
        }
    }
    assert counts == {
        "blank-parity": 11,
        "unicode-exact-after-label-removal": 221,
        "publication-transcription-differs": 818,
    }
    return output


def main():
    rows = build_rows()
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        "wrote 1,050 same-elicitation reconciliation rows: "
        "11 blank parity; 221 Unicode-exact after label removal; "
        "818 publication-transcription differences"
    )


if __name__ == "__main__":
    main()
