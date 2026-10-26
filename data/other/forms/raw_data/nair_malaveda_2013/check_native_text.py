"""Crosscheck provisional literal witnesses against pinned native page text.

This catches transcription/spacing drift, not visual glyph or source-scope errors.
Separate drawn lowlines are ignored only for this text-layer comparison.
"""
import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent


def plain(value):
    value = value.replace("\uf019", "\u0303").replace("\uf02c", "\u0323").replace("\u0332", "")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


blocks = [json.loads(line) for line in (HERE / "physical-units-first-reading.jsonl").read_text().splitlines()]
pages = {page: plain((HERE / f"pdf{page}-native.txt").read_text()) for page in range(40, 45)}
matches = []
missing = []
for block in blocks:
    for witness in block["witnesses"]:
        form = plain(witness["literal_first_reading"])
        check = {"witness_key": witness["witness_key"], "literal": witness["literal_first_reading"],
                 "text_layer_comparison": form, "page": block["printed_and_pdf_page"]}
        (matches if form in pages[block["printed_and_pdf_page"]] else missing).append(check)
(HERE / "native-text-crosscheck.json").write_text(json.dumps({"status": "provisional text-layer crosscheck; not a visual/source audit",
    "matched": len(matches), "missing": missing, "notes": "Lowlines are separate PDF drawing objects and are ignored only in this comparison. Native legacy tilde/dot slots are decoded; literal dash/space differences are not ignored."}, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"matched": len(matches), "missing": len(missing), "missing_literals": [m["literal"] for m in missing]}, ensure_ascii=False))
