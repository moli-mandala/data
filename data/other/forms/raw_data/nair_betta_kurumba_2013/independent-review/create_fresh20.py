"""Seeded source-page-balanced original-vs-proposal selection; no input writes."""
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

P = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
seed = 20260927423
csv_path = P / "proposed.csv"
audit_path = P / "proposal-audit.jsonl"
rows = list(csv.reader(csv_path.open()))
audit = [json.loads(s) for s in audit_path.read_text().splitlines()]
by_unit = {a["unit_key"]: a for a in audit}
previous = HERE / "root-original-review-pass1-20260927421.json"
excluded_keys = {s["entry_key"] for s in json.loads(previous.read_text())["sample"]}
assert len(excluded_keys) == 20
previous2 = HERE / "fresh20-selection-pass2.json"
excluded_keys |= {s["key"] for s in json.loads(previous2.read_text())["sample"]}
assert len(excluded_keys) == 40
excluded_units = {k.split(":variant:")[0] for k in excluded_keys}
grouped = defaultdict(list)
for r in rows:
    unit_key = r[10].split(":variant:")[0]
    a = by_unit[unit_key]
    if unit_key not in excluded_units:
        grouped[a["printed_page"]].append((r, a))
assert excluded_units <= set(by_unit)
draw = random.Random(seed)
sample = []
for page, n in ((12, 2), (13, 3), (14, 3), (15, 3), (16, 3), (17, 3), (18, 3)):
    picks = draw.sample(grouped[page], n)
    for r, a in picks:
        sample.append({"key": r[10], "page": page, "section": a["section"],
                       "physical_unit": a, "proposed_row": r,
                       "review_form": None, "review_gloss": None,
                       "review_tags_and_notes": None, "material_error": None,
                       "original_evidence": None})
out = {"status": "selection-only-pending-independent-original-review",
       "seed": seed, "sampled": len(sample),
       "excluded_previous_keys": sorted(excluded_keys),
       "excluded_physical_units": sorted(excluded_units),
       "previous_audit_sha256": hashlib.sha256(previous.read_bytes()).hexdigest(),
       "previous_selection2_sha256": hashlib.sha256(previous2.read_bytes()).hexdigest(),
       "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
       "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
       "page_counts": {str(p): sum(x["page"] == p for x in sample) for p in range(12, 19)},
       "source_pdf_sha256": audit[0]["source_pdf_sha256"],
       "sample": sample,
       "targeted_controls": ["p. 12 diagnostic -ooda vs p. 13 ooḍa",
                             "p. 16 plain -lpad- prose and relative example vs underdotted negative examples",
                             "p. 16 missing purposive-infinitive target form remains audit-only",
                             "p. 17 printed ba-nd-um vs prose suffix -un contradiction",
                             "p. 18 all 12 suffix cells and 3 slash alternatives"]}
HERE.mkdir(exist_ok=True)
assert len({s["key"].split(":variant:")[0] for s in sample}) == 20
(HERE / "fresh20-selection-pass3.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "sample"}, ensure_ascii=False))
