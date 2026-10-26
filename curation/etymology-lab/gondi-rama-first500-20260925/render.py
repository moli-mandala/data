"""Render Rama et al. first-500 Gondi research-only numbered review manifests."""
import collections
import csv
import json
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P.parents[2]
DEST = P.parent / "Gondi"
decisions = json.loads((P / "proposed-decisions.json").read_text())
rows, holds = decisions["accepted"], decisions["held"]
articles = json.loads((P / "primary-evidence.json").read_text())
parent_ids = {p for r in rows for p in r.get("components", [r["parent"]])}
parents = {}
with (DATA / "cldf/forms.csv").open() as stream:
    for record in csv.DictReader(stream):
        if record["ID"] in parent_ids:
            parents[record["ID"]] = record
assert set(parents) == parent_ids

groups = collections.OrderedDict()
for row in rows:
    key = (tuple(row.get("components", [row["parent"]])), row["evidence"])
    groups.setdefault(key, []).append(row)
grouped = list(groups.items())
old_paths = json.loads((P / "review-manifest-paths.json").read_text()) if (P / "review-manifest-paths.json").exists() else []
next_number = max((int(f.stem.split("-")[1]) for f in DEST.glob("batch-*.json") if f.stem.split("-")[1].isdigit()), default=0) + 1
paths, links = [], []

def esc(value):
    return str(value).replace("|", "\\|").replace("\n", " ")

for offset, start in enumerate(range(0, len(grouped), 5)):
    subset = grouped[start:start + 5]
    if offset < len(old_paths):
        path = Path(old_paths[offset])
        previous = json.loads(path.read_text())
        assert previous["researchDirectory"] == str(P)
        number = previous["batch"]
    else:
        number = next_number
        next_number += 1
        path = DEST / f"batch-{number:03d}.json"
        assert not path.exists(), path
    proposals = []
    lines = [f"**Next {len(subset)} proposals, ready for review. Nothing in this batch saved yet.**", "",
             f"Gondi · Rama et al. 2017 first-500 residual records · batch {number:03d}. Pending user review.", "",
             "| # | Gondi | Proposed etymology | Evidence |", "|---|---|---|---|"]
    for position, ((component_ids, evidence), items) in enumerate(subset, 1):
        forms = list(dict.fromkeys((x["record"]["Form"], x["record"]["Gloss"]) for x in items))
        label = ", ".join(f"**{esc(form)}** ‘{esc(gloss)}’" for form, gloss in forms)
        if len(component_ids) > 1:
            proposed = "Ordered components: " + " + ".join(
                f"[**DEDR {pid[1:]}**]({articles[pid]['URL']})" for pid in component_ids)
        else:
            pid = component_ids[0]
            proposed = f"Proposed reflex of [**DEDR {pid[1:]}**]({articles[pid]['URL']})"
        lines.append(f"| {position} | {label} | {proposed} | {esc(evidence)} ({len(items)} record{'s' if len(items) != 1 else ''}.) |")
        assignments = []
        for item in items:
            for component_position, pid in enumerate(component_ids, 1):
                assignments.append({"Form_ID": item["record"]["ID"], "Etymon_ID": pid,
                                    "Kind": item["kind"], "Rank": "1", "Status": "proposed",
                                    "Source": item["citation"], "Notes": item["evidence"],
                                    "Pos": str(component_position) if len(component_ids) > 1 else ""})
        proposal = {"number": position, "status": "pending-review", "parentId": component_ids[0],
                    "parentForm": parents[component_ids[0]]["Form"], "kind": items[0]["kind"],
                    "citation": items[0]["citation"], "evidence": evidence,
                    "formIds": sorted(x["record"]["ID"] for x in items),
                    "records": [x["record"] for x in items], "assignments": assignments}
        if len(component_ids) > 1:
            proposal["componentIds"] = list(component_ids)
            proposal["componentForms"] = [parents[pid]["Form"] for pid in component_ids]
        proposals.append(proposal)
    edge_count = sum(len(x["assignments"]) for x in proposals)
    record_count = sum(len(x["records"]) for x in proposals)
    lines += ["", f"Proposed assignment rows: **{edge_count}**. Affected records: **{record_count}**. Relation kinds: **{', '.join(sorted({x['kind'] for x in proposals}))}**.", "",
              f"Exact persistent IDs, source locators and ordered assignments: [{path.name}]({path}).", "",
              "Excluded ambiguous responses are in [research holds](../gondi-rama-first500-20260925/research-holds.json). No new lexical material or donor nodes are proposed."]
    manifest = {"language": "Gondi", "batch": number, "status": "pending-review", "date": "2026-09-25",
                "researchDirectory": str(P), "proposals": proposals}
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    review = path.with_name(path.stem + "-review.md")
    review.write_text("\n".join(lines) + "\n")
    paths.append(str(path))
    links.append(f"[{number:03d}]({review})")

assert len(old_paths) <= len(paths)
(P / "review-manifest-paths.json").write_text(json.dumps(paths, indent=2) + "\n")
edges = sum(len(x.get("components", [x["parent"]])) for x in rows)
index = ["# Gondi Rama et al. 2017 first-500 residual records — research review", "",
         "**Pending review. No new assignments have been saved. No rebuild has been run.**", "",
         f"The exact source inventory has **10,264 records**: **4,708 linked**, **309 earlier pending**, **373 earlier held**, and **4,874 previously unexamined**. This bounded pass selected 500 eligible records; **{len(rows)} records** have proposals producing **{edges} assignment rows**, **{len(holds)}** have explicit research holds, and **4,374 remain explicitly deferred and unexamined**. Ant, meat, and broom cases are among the deferred IDs.", "",
         "New pending batches: " + ", ".join(links) + ".", "",
         "The 15 [exact-form homepage search chunks](discovery-001-035-results.json) covered all 500 selected records using the current Jambu query implementation; [discovery coverage](discovery-coverage.json) confirms 3,078 returned rows and exact query alignment. The [full cached DEDR articles](primary-evidence.json), [draft decisions](proposed-decisions.json), [holds](research-holds.json), [deferred records](deferred-inventory.json), and [source-wide coverage](source-coverage-001.json) preserve evidence and exact ownership.", "",
         "The source combines site-specific cognate-list rows across Gondi varieties. Identical surface forms and repeated glosses remain distinct by persistent ID and original site citation. This first alphabetical slice includes 91 cabbage/cauliflower records and many other regional-loan candidates. No immediate donor is assumed merely because a form resembles Hindi or Marathi.", "",
         "Three **goḍēl** ‘axe’ records match full DEDR addendum da32 directly, but that addendum lacks a linkable parent node in the current graph. **naktur** ‘blood’ resembles DEDR 3748 Gondi nattur, but the consonant cluster is unexplained. Both remain held. A DEDR ID denotes an entire article; its displayed head may differ from the attested comparison inside the article. Existing source records and accepted sidecars remain unchanged."]
(P / "REVIEW.md").write_text("\n".join(index) + "\n")
print(json.dumps({"batches": len(paths), "proposals": len(grouped), "records": len(rows), "rows": edges, "holds": len(holds)}))
