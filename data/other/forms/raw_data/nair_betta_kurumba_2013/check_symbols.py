"""Complete literal-route input-symbol coverage without invented IPA."""
import collections
import csv
import hashlib
import json
import unicodedata
from pathlib import Path

P = Path(__file__).resolve().parent
rows = list(csv.reader((P / "proposed.csv").open()))
symbols = collections.Counter(c for r in rows for c in r[2])
assert all(unicodedata.is_normalized("NFC", r[2]) for r in rows)
assert not any(unicodedata.category(c) in {"Co", "Cc"} or c == "\ufffd" for c in symbols)
out = {"status": "complete literal-route symbol inventory",
       "routing": "convert: false, preserve_hyphens: true; Roman source spelling is preserved, with no inferred phonemic tier",
       "csv_sha256": hashlib.sha256((P / "proposed.csv").read_bytes()).hexdigest(),
       "rows": len(rows), "symbol_count": len(symbols),
       "symbols": [{"symbol": c, "codepoint": f"U+{ord(c):04X}",
                    "name": unicodedata.name(c, "UNNAMED"), "occurrences": count}
                   for c, count in sorted(symbols.items())],
       "lowline_policy": "Short vector rectangles are retained as U+0332 on the selected glyphs; source underdots are distinct U+0323/precomposed letters",
       "dash_policy": "Printed en dash and ASCII hyphen remain distinct in Original/Form; structural significance is provided in tags/audit",
       "phonemic": "blank on every proposed row because the consulted chapter provides one Roman tier, not a distinct IPA tier"}
(P / "symbol-inventory.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "symbols"}, ensure_ascii=False))
