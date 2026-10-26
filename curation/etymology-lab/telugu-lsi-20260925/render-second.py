"""Render checkpoint 002 as pending review manifests; never save assignments."""
import collections
import csv
import json
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P.parents[2]
DEST = P.parent / "Telugu"
decisions = json.loads((P / "proposed-decisions-002.json").read_text())
articles = json.loads((P / "primary-evidence-002.json").read_text())
accepted = decisions["accepted"]
assert decisions["status"] == "pending-review"
ids = {x["parent"] for x in accepted}
parents = {r["ID"]: r for r in csv.DictReader((DATA / "cldf/forms.csv").open()) if r["ID"] in ids}
assert set(parents) == ids
groups = collections.OrderedDict()
for item in accepted:
    groups.setdefault((item["parent"], item["evidence"]), []).append(item)
old_owned = json.loads((P / "review-manifest-paths-002.json").read_text()).get("Telugu", []) if (P / "review-manifest-paths-002.json").exists() else []
existing = [int(f.stem.split("-")[1]) for f in DEST.glob("batch-*.json") if f.stem.split("-")[1].isdigit()]
start = max(existing, default=0) + 1
owned, links = [], []
for offset in range(0, len(groups), 5):
    batch_index = offset // 5
    number = int(Path(old_owned[batch_index]).stem.split("-")[1]) if batch_index < len(old_owned) else start + batch_index - len(old_owned)
    subset = list(groups.items())[offset:offset + 5]
    path = DEST / f"batch-{number:03d}.json"
    if path.exists():
        prior = json.loads(path.read_text())
        assert prior.get("researchDirectory") == str(P) and prior.get("checkpoint") == "002", path
    props = []
    lines = [f"**Next {len(subset)} proposals, ready for review. Nothing in this batch saved yet.**", "",
             f"Telugu · LSI 1928 remaining records · batch {number:03d}. Pending review.", "",
             "| # | Telugu | Proposed etymology | Evidence |", "|---|---|---|---|"]
    for j, ((parent, evidence), items) in enumerate(subset, 1):
        article = articles[parent]
        url = f'https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page={article["Page"]}'
        form_label = ", ".join("**" + x["record"]["Form"].replace("|", "\\|") + "** ‘" + x["record"]["Gloss"] + "’" for x in items)
        lines.append(f"| {j} | {form_label} | Proposed reflex of [**DEDR {parent[1:]}**]({url}) | {evidence} ({len(items)} record" + ("s" if len(items) != 1 else "") + ".) |")
        assignments = [dict(Form_ID=x["record"]["ID"], Etymon_ID=parent, Kind="reflex", Rank="1", Status="proposed",
                            Source=x["citation"], Notes=x["evidence"], Pos="") for x in items]
        props.append(dict(number=j, status="pending-review", parentId=parent, parentForm=parents[parent]["Form"],
                          kind="reflex", citation=items[0]["citation"], evidence=evidence,
                          formIds=sorted(x["record"]["ID"] for x in items), records=[x["record"] for x in items],
                          assignments=assignments))
    affected = sum(len(p["records"]) for p in props)
    lines += ["", f"Proposed assignment rows: **{affected}**. Affected records: **{affected}**. Relation kind: **reflex**.",
              "", f"Exact persistent IDs and complete records: [{path.name}]({path}).", "",
              "Complete clauses, opaque loans, and uncertain source transcriptions remain in [research holds](../telugu-lsi-20260925/research-holds-002.json)."]
    path.write_text(json.dumps(dict(language="Telugu", batch=number, status="pending-review", date="2026-09-25",
                                    checkpoint="002", researchDirectory=str(P), proposals=props), ensure_ascii=False, indent=2) + "\n")
    review = path.with_name(path.stem + "-review.md")
    review.write_text("\n".join(lines) + "\n")
    owned.append(str(path))
    links.append(f"[{number:03d}]({review})")
(P / "review-manifest-paths-002.json").write_text(json.dumps({"Telugu": owned}, indent=2) + "\n")

allrows = json.loads((P / "survey-inventory.json").read_text())
prior_p = json.loads((P / "previously-proposed-records.json").read_text())
prior_h = json.loads((P / "previously-held-records.json").read_text())
first = json.loads((P / "proposed-decisions.json").read_text())
second = decisions
owned_ids = [x["record"]["ID"] for x in prior_p + prior_h + first["accepted"] + first["held"] + second["accepted"] + second["held"]]
assert len(owned_ids) == len(set(owned_ids)) == len(allrows) == 167
assert set(owned_ids) == {r["ID"] for r in allrows}
(P / "record-coverage-002.json").write_text(json.dumps({"sourceRecords": len(allrows), "earlierProposed": len(prior_p),
    "earlierHeld": len(prior_h), "firstProposed": len(first["accepted"]), "firstHeld": len(first["held"]),
    "secondProposed": len(second["accepted"]), "secondHeld": len(second["held"]), "unexamined": 0,
    "uniqueOwnership": True}, indent=2) + "\n")
(P / "unexamined-inventory.json").write_text("[]\n")
index = ["# Telugu LSI 1928 residual research — checkpoint 002", "",
         "**Pending review. No new assignments saved. No rebuild run.**", "",
         f"The remaining 83 records have {len(second['accepted'])} new proposals and {len(second['held'])} holds. "
         f"All 167 source records now have one research disposition: {len(prior_p)} earlier pending proposals, "
         f"{len(first['accepted'])} checkpoint-001 proposals, {len(second['accepted'])} checkpoint-002 proposals, "
         f"{len(first['held']) + len(second['held'])} holds, and zero unexamined.", "",
         "New review batches: " + ", ".join(links) + ".", "",
         "[Exact-form search](discovery-002-results.json) and [DEDR/gloss search](followup-002-results.json) used the existing homepage query code against the read-only database. "
         "The generic ‘He’ gloss returned the first 1,000 of 1,389 hits; the exact pronoun search and full DEDR 1 article resolved those proposed forms. "
         "[Full primary articles](primary-evidence-002.json), [decisions](proposed-decisions-002.json), [holds](research-holds-002.json), and [coverage](record-coverage-002.json) preserve the basis.", "",
         "Full subject-plus-predicate clauses remain held until a complete component analysis exists. No source record, parent heading, or accepted sidecar was edited."]
(P / "REVIEW-002.md").write_text("\n".join(index) + "\n")
print(json.dumps({"proposed": len(second["accepted"]), "held": len(second["held"]), "batches": len(owned), "unexamined": 0}))
