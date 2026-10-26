"""Pin a stratified source-unit sample before original-page inspection."""
import csv
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
PDF = next(p for p in HERE.parents if p.name == "jambu-all") / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf"
SEED = 20260927340

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    inputs = {name: PACKAGE / name for name in (
        "physical-units-first-reading.jsonl", "review-candidate.csv", "review-audit.jsonl")}
    audit = [json.loads(s) for s in inputs["review-audit.jsonl"].read_text().splitlines()]
    rows = {r[10]: r for r in csv.reader(inputs["review-candidate.csv"].open(newline=""))}
    by_key = {r["source_unit_key"]: r for r in audit}
    groups = {
        "p32-vowel": ("p32:vowel-shift:", 4),
        "p32-vconsonant": ("p32:v-consonant:", 3),
        "p32-retroflex": ("p32:retroflex-to-cc:", 1),
        "p32-lto y": ("p32:l-to-y:", 1),
        "p32-morphology": ("p32:noun-increment:", 1),
        "p33-accusative": ("p33:noun-increment:", 2),
        "p33-present": ("p33:present:", 3),
        "p33-negative": ("p33:negative:", 2),
        "p33-native": ("p33:native-words:", 3),
    }
    rng = random.Random(SEED)
    selected = []
    for stratum, (part, count) in groups.items():
        eligible = sorted(k for k, a in by_key.items() if part in k and a["emitted_entry_keys"])
        assert len(eligible) >= count, (stratum, len(eligible))
        for key in sorted(rng.sample(eligible, count)):
            a = by_key[key]
            selected.append({"stratum": stratum, "source_unit_key": key,
                             "printed_page": a["printed_page"], "kind": a["kind"],
                             "actual_rows": [{"entry_key": entry,
                                              "columns": rows[entry]}
                                             for entry in a["emitted_entry_keys"]],
                             "source_witnesses": a["witnesses"]})
    assert len(selected) == 20 and len({x["source_unit_key"] for x in selected}) == 20
    controls = ["nair2013kani:p32:identity:1", "nair2013kani:p32:vowel-shift:0",
                "nair2013kani:p33:present:0", "nair2013kani:p33:negative:0"]
    result = {"seed": SEED, "status": "selected, originals not yet checked",
              "source_pdf_sha256": digest(PDF),
              "input_sha256": {name: digest(path) for name, path in inputs.items()},
              "selected": selected,
              "edge_controls": [{"source_unit_key": k, "audit": by_key[k]} for k in controls]}
    (HERE / "selection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(len(selected), "source units /", sum(len(x["actual_rows"]) for x in selected), "rows")

if __name__ == "__main__":
    main()
