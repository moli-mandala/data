#!/usr/bin/env python3
"""Create the stable human-review sheet for the eleven Irula word lists."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
PALAKKAD = REPO / "data/other/forms/20260826-sil-palakkad.csv"
FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
FIELDS = [
    "Item", "Gloss", "Site", "Response", "Group", "PDF_Page", "Printed_Page", "Column",
    "Raw_OCR", "Palakkad_Kunjapana_Control", "Transcription", "Review", "Uncertainty",
]


def gloss_key(value: str) -> str:
    value = value.lower().replace("chili", "chilli").replace("teeth", "tooth")
    value = value.replace("nail", "fingernail")
    value = re.sub(r"\([^)]*\)", "", value)
    value = re.sub(r"\bhe is\b|\bhe\b|\bit\b|\bdon't\b", "", value)
    value = value.split(",", 1)[0]
    return re.sub(r"[^a-z?]+", " ", value).strip()


def palakkad_controls() -> dict[str, str]:
    result = {}
    with PALAKKAD.open(encoding="utf-8", newline="") as stream:
        for values in csv.reader(stream):
            row = dict(zip(FORM_FIELDS, values))
            if row["Language_ID"] == "Irula" and "palakkad-irula-kunjapana" in row["Tags"]:
                result.setdefault(gloss_key(row["Gloss"]), row["Phonemic"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scaffold", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.scaffold.read_text(encoding="utf-8"))
    controls = palakkad_controls()
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            if not row["target"]:
                continue
            missing = row["ocr"].lower() == "missing data"
            writer.writerow(
                {
                    "Item": row["item"],
                    "Gloss": row["gloss"],
                    "Site": row["site"],
                    "Response": row["response"],
                    "Group": row["group"],
                    "PDF_Page": row["pdf_page"],
                    "Printed_Page": row["printed_page"],
                    "Column": row["column"],
                    "Raw_OCR": row["ocr"],
                    "Palakkad_Kunjapana_Control": controls.get(gloss_key(row["gloss"]), ""),
                    "Transcription": "",
                    "Review": "missing" if missing else "pending",
                    "Uncertainty": "",
                }
            )
    print(f"wrote {sum(row['target'] for row in rows)} target response records to {args.output}")


if __name__ == "__main__":
    main()
