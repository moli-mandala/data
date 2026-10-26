"""Render Alu Kurumba Blair et al. pending review batches; no accepted graph writes."""
import collections
import csv
import json
from pathlib import Path

P = Path(__file__).resolve().parent
LAB = P.parent
ROOT = P.parents[2]
DEST = LAB / "AluKurumba"
data = json.loads((P / "proposed-decisions.json").read_text())
rows, held = data["accepted"], data["held"]
articles = json.loads((P / "primary-evidence.json").read_text())
parent_ids = {p for r in rows for p in r.get("components", [r["parent"]])}
parents = {}
with (ROOT / "cldf/forms.csv").open() as fh:
    for rec in csv.DictReader(fh):
        if rec["ID"] in parent_ids:
            parents[rec["ID"]] = rec
assert set(parents) == parent_ids

groups = collections.OrderedDict()
for row in rows:
    key = (tuple(row.get("components", [row["parent"]])), row["evidence"])
    groups.setdefault(key, []).append(row)
grouped = list(groups.items())

old_paths = json.loads((P / "review-manifest-paths.json").read_text()) if (P / "review-manifest-paths.json").exists() else []
max_existing = max((int(f.stem.split("-")[1]) for f in DEST.glob("batch-*.json") if f.stem.split("-")[1].isdigit()), default=0)
next_number = max_existing + 1
new_paths = []
links = []

def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")

for start in range(0, len(grouped), 5):
    subset = grouped[start:start + 5]
    offset = start // 5
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
    lines = [f"**Next {len(subset)} proposal{'s' if len(subset) != 1 else ''}, ready for review. Nothing in this batch saved yet.**", "",
             f"AluKurumba · Blair et al. 2012 residual records · batch {number:03d}. Pending user review.", "",
             "| # | AluKurumba | Proposed etymology | Evidence |", "|---|---|---|---|"]
    for pos, ((component_ids, evidence), items) in enumerate(subset, 1):
        forms = list(dict.fromkeys((x["record"]["Form"], x["record"]["Gloss"]) for x in items))
        form_label = ", ".join(f"**{esc(f)}** ‘{esc(g)}’" for f, g in forms)
        if len(component_ids) > 1:
            proposed = "Ordered components: " + " + ".join(
                f"[**DEDR {pid[1:]}**]({articles[pid]['URL']})" for pid in component_ids)
        else:
            pid = component_ids[0]
            proposed = f"Proposed reflex of [**DEDR {pid[1:]}**]({articles[pid]['URL']})"
        lines.append(f"| {pos} | {form_label} | {proposed} | {esc(evidence)} ({len(items)} record{'s' if len(items) != 1 else ''}.) |")
        assignments = []
        for item in items:
            for part_pos, pid in enumerate(component_ids, 1):
                assignments.append({"Form_ID": item["record"]["ID"], "Etymon_ID": pid,
                                    "Kind": item["kind"], "Rank": "1", "Status": "proposed",
                                    "Source": item["citation"], "Notes": item["evidence"],
                                    "Pos": str(part_pos) if len(component_ids) > 1 else ""})
        proposal = {"number": pos, "status": "pending-review", "parentId": component_ids[0],
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
              f"Exact persistent IDs, source locators and ordered assignments: [{path.name}]({path}).",
              "", "Excluded uncertain responses are documented in [research holds](../alukurumba-blair-20260925/research-holds.json). No new lexical material or donor nodes are proposed."]
    manifest = {"language": "AluKurumba", "batch": number, "status": "pending-review", "date": "2026-09-25",
                "researchDirectory": str(P), "proposals": proposals}
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    review = path.with_name(path.stem + "-review.md")
    review.write_text("\n".join(lines) + "\n")
    new_paths.append(str(path))
    links.append(f"[{number:03d}]({review})")

assert len(old_paths) <= len(new_paths)
(P / "review-manifest-paths.json").write_text(json.dumps(new_paths, indent=2) + "\n")
edge_total = sum(len(x.get("components", [x["parent"]])) for x in rows)
index = ["# Alu Kurumba Blair et al. 2012 residual records — research review", "",
         "**Pending review. No new assignments have been saved. No rebuild has been run.**", "",
         f"All **328 residual records** have an initial research disposition: **{len(rows)} proposed records**, producing **{edge_total} assignment rows**, and **{len(held)} research holds**. Zero residual records remain unexamined. Unresolved loan donors and multiword elicitation responses are held for further analysis.", "",
         "New pending batches: " + ", ".join(links) + ".", "",
         "The exact-form homepage search covered all 328 residual source records in ten retained chunks using the existing Jambu query code; 328 global comparisons returned 4,671 rows. The [cached full DEDR articles](primary-evidence.json), [draft decisions](proposed-decisions.json), [holds](research-holds.json), and [exact whole-source coverage](source-coverage-001.json) preserve the research basis.", "",
         "Ambiguous source spellings and contact forms with unresolved immediate donors remain held. DEDR entries are whole articles: a survey reflex may occur within an article whose displayed head differs. Existing source records, dictionary heads, and accepted sidecars remain unchanged."]
(P / "REVIEW.md").write_text("\n".join(index) + "\n")
print(json.dumps({"batches": len(new_paths), "proposals": len(grouped), "records": len(rows), "rows": edge_total, "holds": len(held)}))
