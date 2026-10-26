"""Select original-backed witness rows before independent inspection."""
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
CSV = PACKAGE / "proposed.csv"
AUDIT = PACKAGE / "proposal-audit.jsonl"
FREEZE = PACKAGE / "review-freeze.json"
SEED = 20260927411


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


assert sha(CSV)=="38574c8a5365c956852a13cfe75632cfb2d900f2546750f69184d58a2c84bee5"
freeze = json.loads(FREEZE.read_text())
for name,digest in freeze["files_sha256"].items(): assert sha(PACKAGE / name)==digest,name
rows = {r[10]:r for r in csv.reader(CSV.open(newline=""))}
units = defaultdict(list)
for a in [json.loads(s) for s in AUDIT.read_text().splitlines()]:
    key = a.get("witness",{}).get("witness_key")
    if key in rows: units[a["source_unit_key"]].append((a,rows[key]))
by_page = defaultdict(list)
for key,entries in units.items(): by_page[entries[0][0]["printed_page"]].append(key)
rng = random.Random(SEED)
selected = []
for page,count in ((40,4),(41,6),(42,4),(43,4),(44,2)):
    for key in rng.sample(sorted(by_page[page]),count):
        record,row = rng.choice(units[key])
        selected.append(dict(sample_index=len(selected)+1,source_unit_key=key,printed_page=page,
            witness_key=row[10],actual_csv_row=row,actual_audit_record=record,
            original_image=f"pdf{page}-original.png",source_locator=f"printed p. {page}: {key.split(':')[-1]}"))
assert len(selected)==len({s["source_unit_key"] for s in selected})==20
output = dict(seed=SEED,status="selected before fresh original inspection",source_pdf_sha256=freeze["source_pdf_sha256"],
    inputs={"review-freeze.json":sha(FREEZE),"proposed.csv":sha(CSV),"proposal-audit.jsonl":sha(AUDIT),"graph-proposal.json":sha(PACKAGE / "graph-proposal.json")},
    selection=selected,method="20 distinct physical source blocks, four pronoun blocks40/six41/four42/four43/two44; one actual emitted witness drawn per block")
(HERE / "selection-20260927.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
for s in selected: print(s["sample_index"],s["printed_page"],s["actual_csv_row"])
