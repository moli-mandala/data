#!/usr/bin/env python3
"""Build the complete source-local preservation-profile character inventory."""
from __future__ import annotations

import csv
import importlib.util
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMPORTER = HERE / "import_ho.py"
OUT = HERE / "symbol_inventory.tsv"


def main() -> None:
    spec = importlib.util.spec_from_file_location("ho_importer", IMPORTER)
    module = importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(module)
    rows = module.overlay_manual_chunks(module.validate_base())
    module.require_complete(rows)
    forms, _ = module.build(rows, module.validate_registry())
    counts = Counter(character for row in forms for character in row[2])
    with OUT.open("w", encoding="utf-8", newline="") as stream:
        fields = ["Codepoint", "Symbol", "Unicode_Name", "Count", "Decision"]
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for character in sorted(counts, key=ord):
            if character.isspace(): decision = "preserve word boundary"
            elif character.isdigit(): decision = "preserve lexical numeral; similarity labels already removed"
            elif unicodedata.category(character).startswith("P"): decision = "preserve reviewed source punctuation"
            else: decision = "preserve NFC diplomatic transcription"
            writer.writerow({
                "Codepoint": f"U+{ord(character):04X}", "Symbol": character,
                "Unicode_Name": unicodedata.name(character, "UNNAMED"),
                "Count": counts[character], "Decision": decision,
            })
    if "�" in counts:
        raise ValueError("Replacement character present in staged Ho forms")
    print(f"wrote {len(counts)} source symbols covering {len(forms)} staged forms")


if __name__ == "__main__": main()
