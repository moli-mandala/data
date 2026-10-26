"""Pin source-local Adiya review artifacts without installing them."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
files = [
    "scope/physical-unit-census.jsonl", "scope/identity-proposal.json",
    "first-reading.jsonl", "proposed.csv", "proposal-audit.jsonl",
    "proposal-status.json", "candidate-validation.json", "notation-evidence.json",
    "graph-proposal.json", "graph-approved.json", "revision4-graph-continuity.json", "metadata/proposed-settings.yaml",
    "metadata/proposed-dialect.csv", "metadata/proposed-references.bib",
    "prepare_first_reading.py", "prepare_candidate.py", "prepare_graph_proposal.py",
    "record_notation_evidence.py", "validate_candidate.py",
    "record_marker_revision3.py", "revision3-marker-continuity.json", "metadata/formatter-review.json",
]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
for file in files: assert (HERE / file).is_file(), file
rows = list(csv.reader((HERE / "proposed.csv").open(newline="")))
audit = [json.loads(s) for s in (HERE / "proposal-audit.jsonl").read_text().splitlines()]
assert len(rows) == 171 and len(audit) == 29
assert len({r[10] for r in rows}) == len(rows)
assert {r[10] for r in rows} == {k for a in audit for k in a["entry_keys"]}
pdf = HERE.parents[5] / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf"
result = {
    "status": "owner review candidate frozen for independent original audit; no canonical installation",
    "source_pdf": str(pdf), "source_pdf_sha256": sha(pdf),
    "source_chapter_printed_pages": [4,5,6,7,8],
    "original_pages_visually_reviewed_by_owner": [4,5,6,7,8],
    "physical_units": 29, "physical_witnesses": 174,
    "output_rows": 171, "Ravula_Adiya_rows": 136,
    "Malayalam_Kannada_comparison_rows": 35, "documentary_name_holds": 3,
    "graph_edges_applied": 12, "graph_edges_proposed": 0,
    "files_sha256": {name: sha(HERE / name) for name in files},
    "review_priorities": [
        "Every original p4–7 lexical block, particularly underlined l/n/rr, stacked lower dots and source spacing",
        "p5 paradigm column/gender/honorific alignment and plain prose niinŋka versus later niiŋka",
        "p6–7 full-clause gloss/tense/person and p7 kinship address versus reference distinctions",
        "p8 four reported theses and Adiya-to-existing-Ravula identity routing",
        "12 proposed source-explicit variant edges and five held non-edge classes",
    ],
    "deferred_gates": ["independent original/output audit", "graph adjudication",
                       "approval-guarded source-stage installer", "canonical parser/metadata tests",
                       "full CLDF build", "compiled graph/identity tests", "browser QA"],
    "canonical_changes": False,
}
(HERE / "review-freeze.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print("Adiya 171-row source-local review freeze pinned")
