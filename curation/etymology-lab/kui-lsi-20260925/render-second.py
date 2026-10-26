"""Render pending final Kui LSI checkpoint without modifying accepted data."""
import collections
import csv
import json
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P.parents[2]
DEST = P.parent / "Kui"
decisions = json.loads((P / "proposed-decisions-002.json").read_text())
articles = json.loads((P / "primary-evidence-002.json").read_text())
accepted, held = decisions["accepted"], decisions["held"]
assert decisions["status"] == "pending-review"
ids = {part for x in accepted for part in x.get("components", [x["parent"]])}
parents = {r["ID"]: r for r in csv.DictReader((DATA / "cldf/forms.csv").open()) if r["ID"] in ids}
assert set(parents) == ids
groups = collections.OrderedDict()
for item in accepted:
    groups.setdefault((tuple(item.get("components", [item["parent"]])), item["evidence"]), []).append(item)
old_owned = json.loads((P / "review-manifest-paths-002.json").read_text()).get("Kui", []) if (P / "review-manifest-paths-002.json").exists() else []
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
             f"Kui · LSI 1928 remaining records · batch {number:03d}. Pending review.", "",
             "| # | Kui | Proposed etymology | Evidence |", "|---|---|---|---|"]
    for j, ((parts, evidence), items) in enumerate(subset, 1):
        parent = parts[0]
        form_label = ", ".join("**" + x["record"]["Form"].replace("|", "\\|") + "** ‘" + x["record"]["Gloss"] + "’" for x in items)
        kind = items[0]["kind"]
        if kind == "component":
            proposed = "Ordered components: " + " + ".join(
                f'[**DEDR {part[1:]}**](https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page={articles[part]["Page"]})' for part in parts)
        else:
            proposed = f'[**DEDR {parent[1:]}**](https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page={articles[parent]["Page"]}) reflex family'
        lines.append(f"| {j} | {form_label} | {proposed} | {evidence} ({len(items)} record" + ("s" if len(items) != 1 else "") + ".) |")
        assignments = [dict(Form_ID=x["record"]["ID"], Etymon_ID=part, Kind=kind, Rank="1", Status="proposed",
                            Source=x["citation"], Notes=x["evidence"], Pos=str(pos) if kind == "component" else "")
                       for x in items for pos, part in enumerate(parts, 1)]
        prop = dict(number=j, status="pending-review", parentId=parent, parentForm=parents[parent]["Form"], kind=kind,
                    citation=items[0]["citation"], evidence=evidence,
                    formIds=sorted(x["record"]["ID"] for x in items), records=[x["record"] for x in items],
                    assignments=assignments)
        if kind == "component":
            prop.update(componentIds=list(parts), componentForms=[parents[part]["Form"] for part in parts])
        props.append(prop)
    affected = sum(len(p["records"]) for p in props)
    edge_count = sum(len(p["assignments"]) for p in props)
    lines += ["", f"Proposed assignment rows: **{edge_count}**. Affected records: **{affected}**. Relation kinds: **{', '.join(sorted({p['kind'] for p in props}))}**.",
              "", f"Exact persistent IDs and complete records: [{path.name}]({path}).", "",
              "Complete clauses and unsourced loan or compound components remain in [research holds](../kui-lsi-20260925/research-holds-002.json)."]
    path.write_text(json.dumps(dict(language="Kui", batch=number, status="pending-review", date="2026-09-25",
                                    checkpoint="002", researchDirectory=str(P), proposals=props), ensure_ascii=False, indent=2) + "\n")
    review = path.with_name(path.stem + "-review.md")
    review.write_text("\n".join(lines) + "\n")
    owned.append(str(path))
    links.append(f"[{number:03d}]({review})")
(P / "review-manifest-paths-002.json").write_text(json.dumps({"Kui": owned}, indent=2) + "\n")

allrows = json.loads((P / "survey-inventory.json").read_text())
prior_p = json.loads((P / "previously-proposed-records.json").read_text())
prior_h = json.loads((P / "previously-held-records.json").read_text())
first = json.loads((P / "proposed-decisions.json").read_text())
owned_ids = [x["record"]["ID"] for x in prior_p + prior_h + first["accepted"] + first["held"] + accepted + held]
assert len(owned_ids) == len(set(owned_ids)) == len(allrows) == 175
assert set(owned_ids) == {r["ID"] for r in allrows}
(P / "record-coverage-002.json").write_text(json.dumps(dict(sourceRecords=len(allrows), earlierProposed=len(prior_p),
    earlierHeld=len(prior_h), firstProposed=len(first["accepted"]), firstHeld=len(first["held"]),
    secondProposed=len(accepted), secondHeld=len(held), unexamined=0, uniqueOwnership=True), indent=2) + "\n")
(P / "unexamined-inventory.json").write_text("[]\n")
index = ["# Kui LSI 1928 residual research — checkpoint 002", "", "**Pending review. No new assignments saved. No rebuild run.**", "",
         f"The remaining 101 records have {len(accepted)} pending proposals ({sum(len(x.get('components', [x['parent']])) for x in accepted)} assignment rows) and {len(held)} holds. "
         f"All 175 source records now have one disposition: {len(prior_p)} earlier pending proposals, {len(prior_h)} earlier hold, "
         f"{len(first['accepted'])} checkpoint-001 proposals, {len(accepted)} checkpoint-002 proposals, "
         f"{len(first['held']) + len(held)} new holds, and zero unexamined.", "",
         "New review batches: " + ", ".join(links) + ".", "",
         "[Exact-form search sets](discovery-002-results.json), [DEDR/gloss search sets](followup-002-results.json), "
         "[full primary articles](primary-evidence-002.json), [decisions](proposed-decisions-002.json), "
         "[holds](research-holds-002.json), and [coverage](record-coverage-002.json) preserve the research basis. "
         "Some extremely broad pronoun queries reached the search tool's 1,000-row pagination cap; the proposed pronouns were resolved by precise form matches and full DEDR articles.", "",
         "Complete clauses remain held until all parts can be analysed. No source text, parent heading, or accepted sidecar was edited."]
(P / "REVIEW-002.md").write_text("\n".join(index) + "\n")
print(json.dumps(dict(proposed=len(accepted), rows=sum(len(x.get("components", [x["parent"]])) for x in accepted), held=len(held), batches=len(owned), unexamined=0)))
