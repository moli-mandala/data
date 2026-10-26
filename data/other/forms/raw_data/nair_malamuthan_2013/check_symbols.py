"""Literal source-spelling inventory; do not infer phonetic values."""
import collections
import csv
import json
import unicodedata
from pathlib import Path

P = Path(__file__).resolve().parent
rows = list(csv.reader((P / "proposed.csv").open()))
forms = [r[2] for r in rows]
counts = collections.Counter(c for form in forms for c in form)
report = {"row_count": len(rows),
          "all_nfc": all(unicodedata.is_normalized("NFC", f) for f in forms),
          "replacement_count": counts["\ufffd"],
          "codepoints": [{"character": c, "U+": f"U+{ord(c):04X}",
                         "name": unicodedata.name(c, "UNKNOWN"), "occurrences": n}
                        for c, n in sorted(counts.items(), key=lambda x: ord(x[0]))],
          "routing": "literal Roman source spelling, conversion disabled, hyphens retained",
          "source_phonemic_layer": "not separately supplied; Phonemic blank",
          "source_native_script_layer": "not supplied; Native blank"}
assert report["all_nfc"] and report["replacement_count"] == 0
(P / "symbol-inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(len(report["codepoints"]), len(forms))
