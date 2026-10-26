"""Reconcile every compiled Alu Kurumba Varghese form with research ownership."""
import csv
import json
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P.parents[2]
LAB = P.parent

compiled = {}
with (DATA / "cldf/forms.csv").open() as fh:
    for rec in csv.DictReader(fh):
        if rec["Language_ID"] == "AluKurumba" and rec["Source"].startswith("varghese2015palakkad["):
            assert rec["ID"] not in compiled
            compiled[rec["ID"]] = {"Form": rec["Form"], "Gloss": rec["Gloss"], "Source": rec["Source"]}

prior = {}
for path in sorted((LAB / "AluKurumba").glob("batch-*.json")):
    if int(path.stem.split("-")[1]) > 29:
        continue
    manifest = json.loads(path.read_text())
    for proposal in manifest["proposals"]:
        for record in proposal["records"]:
            if record["Language_ID"] == "AluKurumba" and record["Source"].startswith("varghese2015palakkad["):
                assert record["ID"] not in prior
                prior[record["ID"]] = path.name

legacy_holds = {}
for item in json.loads((LAB / "dravidian-surveys-20260921/research-holds.json").read_text()):
    record = item["record"]
    if record["Language_ID"] == "AluKurumba" and record["Source"].startswith("varghese2015palakkad["):
        assert record["ID"] not in legacy_holds
        legacy_holds[record["ID"]] = item["reason"]

current = json.loads((P / "record-coverage-001.json").read_text())["recordOwnership"]
assert not (set(prior) & set(legacy_holds))
assert not (set(prior) & set(current))
assert not (set(legacy_holds) & set(current))
ownership = {rec_id: "prior-proposed:" + batch for rec_id, batch in prior.items()}
ownership.update({rec_id: "prior-hold" for rec_id in legacy_holds})
ownership.update({rec_id: "current-" + disposition for rec_id, disposition in current.items()})
assert set(ownership) == set(compiled), (len(compiled), len(ownership), sorted(set(compiled) - set(ownership)))
assert len(compiled) == len(prior) + len(legacy_holds) + len(current) == 227

report = {"source": "varghese2015palakkad", "language": "AluKurumba", "totalCompiledRecords": len(compiled),
          "priorProposedRecords": len(prior), "priorHeldRecords": len(legacy_holds),
          "currentProposedRecords": sum(v == "proposed" for v in current.values()),
          "currentHeldRecords": sum(v == "held" for v in current.values()),
          "unexaminedRecords": 0, "uniqueOwnership": True, "recordOwnership": dict(sorted(ownership.items())),
          "legacyHoldReasons": legacy_holds}
(P / "source-coverage-001.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}))
