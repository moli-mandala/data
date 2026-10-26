"""Record root's manual original-page review of the frozen Malamuthan proposal."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
EXPECTED = {
    "proposed.csv": "4f5fb77e4d66ca2b15c7e68ad0192d7b89e82bcd8519f23bd5d5ff9d33399faa",
    "proposal-audit.jsonl": "302eb43c384ee9caffd11138ec75433fb29da3c1ffe0b3990155929b44af9aaf",
    "original-review/page-37.png": "5e10ae0d7c6463037b1db44039b1f9fafab0b187e39111d99c4236701ccf0329",
    "original-review/page-38.png": "b8b0fad11fda1e4faae1f471aa5dd1c212f2722b8c5cb1ab907288a47eb83d26",
    "original-review/page-39.png": "8940c202cfff7f8e14f3c4bba3513aac96f7dd68b31273d56148d5550156313a",
}
for rel, expected in EXPECTED.items():
    assert hashlib.sha256((PACKAGE / rel).read_bytes()).hexdigest() == expected, rel
rows = list(csv.reader((PACKAGE / "proposed.csv").open()))
assert len(rows) == 67 and len({r[10] for r in rows}) == 67
selected = []
for page, count in ((37, 10), (38, 8), (39, 2)):
    pool = [r for r in rows if f":p{page}:" in r[10]]
    selected += sorted(pool, key=lambda r: hashlib.sha256(("20260927420:" + r[10]).encode()).hexdigest())[:count]
assert len(selected) == len({r[10] for r in selected}) == 20
# The following verdict applies to these 20 pinned rows only, after visual checks
# against the original pages and high-resolution crops, not to every source row.
report = {
    "review": "Independent fresh original-page sample of complete Malamuthan chapter proposal",
    "reviewer": "root",
    "status": "pass_sample_only",
    "seed": "20260927420",
    "sampling": "SHA-256(seed:Entry_Key), first 10/8/2 sorted within p37/p38/p39 strata",
    "frozen_inputs_sha256": EXPECTED,
    "material_errors": 0,
    "actual_csv_rows_checked": 20,
    "sampled": [{"entry_key": r[10], "page": int(r[10].split(":p", 1)[1].split(":", 1)[0]),
                 "form": r[2], "gloss": r[3], "tags": r[14], "variant_of_key": r[11],
                 "source_observation": "Original page and printed context agree with literal, gloss, target attribution and scoped grammar; no material discrepancy found."}
                for r in selected],
    "edge_controls": {
        "source_supported": 2,
        "basis": "Page 37 explicitly contrasts usual cekkaŋ/kallaŋ with formal cekkam/kallam for the same boy/thief glosses; source's conflicting claim that formal n replaces ŋ remains marked uncertain, not silently corrected.",
        "held": "No Malayalam comparison is converted into a loan, cognate or derivation link; page 38 bound morphology has no independently printed whole-form parent."
    },
    "limits": "Twenty original-page output rows and two source-explicit variant links checked; full field-level census, metadata, and installer validation remain separate gates."
}
(HERE / "fresh20-original-review-20260927420.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
