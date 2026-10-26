"""Lightweight source-local completeness/profile checks; no CLDF build."""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "physical-units-first-reading.jsonl"
CSV = HERE / "review-candidate.csv"
AUDIT = HERE / "review-audit.jsonl"
PROFILE = HERE / "symbol-inventory.json"


def main() -> None:
    units = [json.loads(x) for x in SOURCE.read_text().splitlines()]
    audit = [json.loads(x) for x in AUDIT.read_text().splitlines()]
    with CSV.open(newline="") as inp:
        rows = list(csv.reader(inp))
    assert len(units) == len(audit) == 42
    assert len(rows) == 36 and all(len(row) == 15 for row in rows)
    assert [u["source_unit_key"] for u in units] == [a["source_unit_key"] for a in audit]
    witnesses = [w for a in audit for w in a["witnesses"]]
    assert len(witnesses) == 66 and len({w["entry_key"] for w in witnesses}) == 66
    assert {row[10] for row in rows} == {w["entry_key"] for w in witnesses if w["status"] == "proposed_uninstalled"}
    assert all(row[0] == "Kanikkaran" and row[7].startswith("nair2013kani[p. ") for row in rows)
    assert all("dialect:Kanikkaran:nair2013-kerala-kani:Kerala%20Kani" in row[14].split() for row in rows)
    assert all(not any(unicodedata.category(ch) == "Co" or ch == "�" for ch in row[2]) for row in rows)
    assert all(unicodedata.is_normalized("NFC", row[2]) for row in rows)
    assert all(not any(row[i] for i in (1, 4, 5, 8, 9, 11, 12, 13)) for row in rows)
    assert all(w["source_scope"] == "Kerala Kani target" for w in witnesses if w["entry_key"] in {r[10] for r in rows})
    forms = [row[2] for row in rows]
    assert "naṭakkaan parraa" in forms and "meya peyyine" in forms
    negative_suffix = next(row for row in rows if row[10] == "nair2013kani:p33:negative:0:w1")
    assert negative_suffix[2] == "aa", "The source prints aa without a leading hyphen"
    assert {"suffix", "neg"} <= set(negative_suffix[14].split())
    assert "malampaala" not in forms and "vaalu" not in forms and "netti" not in forms
    # Printed identity/name evidence is accounted for, not silently made a lexeme.
    assert any(w["printed_form"] == "malampaala" for w in witnesses)
    # The source uses double-vowel spelling. Converting it before Original would lose evidence.
    assert any("aa" in form for form in forms) and any("ee" in form for form in forms)
    assert not any("ā" in form or "ē" in form for form in forms)
    tag_by_key = {row[10]: set(row[14].split()) for row in rows}
    assert "sentential" in tag_by_key["nair2013kani:p33:present:1:w1"]
    assert "sentential" not in tag_by_key["nair2013kani:p33:negative:1:w1"]
    assert "multiword-expression" in tag_by_key["nair2013kani:p33:negative:1:w1"]
    assert "noun" not in tag_by_key["nair2013kani:p33:native-words:7:w1"]
    characters = Counter(ch for form in forms for ch in unicodedata.normalize("NFD", form))
    PROFILE.write_text(json.dumps({
        "status": "source-local literal-preservation proposal; no phonemic conversion",
        "input_sha256": hashlib.sha256(CSV.read_bytes()).hexdigest(),
        "rows_checked": len(rows),
        "distinct_nfd_codepoints": len(characters),
        "nfd_codepoints": [{"char": ch, "codepoint": f"U+{ord(ch):04X}", "unicode_name": unicodedata.name(ch, "UNKNOWN"), "occurrences": n} for ch, n in sorted(characters.items())],
        "no_private_use_or_replacement": True,
        "original_policy": "Keep complete source Roman form with doubled vowels; conversion false; native/phonemic blank.",
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"PASS 42 units, 66 witness slots, 36 candidate rows, {len(characters)} NFD codepoints")


if __name__ == "__main__":
    main()
