"""Record the independent original-page review; never modify owner proposals."""
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = HERE.parent / "interlinear-review"
ROOT = next(p for p in HERE.parents if (p / "data/SOURCE_INGESTION_CHECKLIST.md").exists())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load = lambda p: [json.loads(line) for line in p.read_text().splitlines()]
selection = json.loads((HERE / "selection-20260927329.json").read_text())
for name, digest in selection["inputs"].items():
    assert sha(OWNER / name) == digest, name
whole = load(OWNER / "whole-expression-proposal.jsonl")
columns = load(OWNER / "c5-column-proposal.jsonl")
rows = {r["source_unit_key"]: r for r in whole + columns}
assert len(whole) == 117 and len(columns) == 192 and len(rows) == 309
assert all(rows[r["source_unit_key"]] == r for r in selection["sample"])
coverage = json.loads((OWNER / "c5-column-coverage.json").read_text())
covered = [k for block in coverage["coverage"] for k in block["children"]]
assert len(covered) == len(set(covered)) == 192
assert set(covered) == {r["source_unit_key"] for r in columns}
for block in coverage["coverage"]:
    indices = []
    for key in block["children"]:
        r = rows[key]
        assert r["physical_p_key"] == block["physical_p_key"]
        assert r["physical_g_key"] == block["physical_g_key"]
        a, b = r["target_token_range"]
        indices.extend(range(a, b + 1))
    assert indices == list(range(1, block["target_tokens"] + 1))
status = json.loads((OWNER / "grouping-status.json").read_text())
assert status["all_p_blocks_accounted"]
prefix = "behera2022korwakodaku:"
findings = [
    {"id": "introductory-fragment", "keys": [prefix + "C.5:expression:1"],
     "old": {"tags": rows[prefix + "C.5:expression:1"]["tags"]},
     "proposed": {"tags": ["multiword-expression"]},
     "evidence": "PDF105/printed95 n1: free translation 'Once, when I was small'; introductory temporal fragment, followed by separately numbered main narrative n2. Preserve separate physical unit; do not claim a complete sentence."},
    {"id": "printed-column-conflict", "keys": [prefix + "C.5:column:1:3", prefix + "C.5:column:1:4"],
     "old": {"typed_uncertainty": []},
     "proposed": {"typed_uncertainty": ["source-word-alignment"],
                  "note": "The printed columns pair tʃotʰeɾɐ with 'was' and holo with 'small'; the anomalous pairing is preserved without reassignment or inferred part of speech."},
     "evidence": "PDF105/printed95 n1: directly aligned original p/g columns. No literal correction authorized."},
    {"id": "performance-note-in-gloss", "keys": [prefix + "C.5:column:19.2:4"],
     "old": {"gloss": "entered (drawn out for emphasis)", "source_analysis": ""},
     "proposed": {"gloss": "entered", "source_analysis": "drawn out for emphasis"},
     "evidence": "PDF108/printed98 n19.2 explicitly supplies performance parenthetical after lexical 'entered'. Preserve complete raw_gloss and all source length signs."},
    {"id": "literal-baseline-dot", "keys": [prefix + "C.4:expression:6"],
     "related_physical_key": prefix + "pdf100:line4",
     "proposed": {"typed_uncertainty": ["source-notation"],
                  "note": "The source prints sa.b with a baseline dot; its phonetic interpretation is not specified. The dot is retained literally."},
     "evidence": "PDF100/printed90 n6 original visibly prints sa.b, not an extraction artifact. Apply the same qualification to its continuous-alignment child; that child is outside this report's owned proposal set."}
]
assert rows[prefix + "C.5:column:19.2:4"]["gloss"] == findings[2]["old"]["gloss"]
report = {
    "status": "changes_required",
    "review_scope": "Independent field/grammar review of all117 whole-expression/answer rows and all192 C5 column/group rows; all31 C5 physical p/g pairs visually compared on four original pages; fresh20 exact-row original audit. Not a claim that all117 whole-expression glyph strings received a full independent rereading.",
    "source": {"path": "tmp/pdfs/korwa-kodaku-2022/source.pdf", "sha256": sha(ROOT / "tmp/pdfs/korwa-kodaku-2022/source.pdf")},
    "inputs": selection["inputs"],
    "selection_sha256": sha(HERE / "selection-20260927329.json"),
    "original_pages_viewed": [93, 96, 97, 99, 100, 103, 104, 105, 106, 107, 108],
    "render_policy": "One overwritten original-page PNG; fitz page index PDF page minus1, matrix1.6–1.8; original PDF remains authoritative. No bulk render corpus.",
    "coverage": {"whole_rows": 117, "column_rows": 192, "c5_physical_pairs": 31,
                 "whole_by_section_role": dict(Counter(r["section"] + ":" + r["role"] for r in whole)),
                 "all192_column_keys_unique_and_covered": True,
                 "all_column_target_ranges_contiguous_and_nonoverlapping": True,
                 "whole_p_accounting": status["physical_p_blocks"],
                 "whole_p_accounting_scope": "Checked owner physical-key transfer and complete key census; root separately verifies full v5 transfer.",
                 "unresolved_internal_spans_preserved": coverage["unresolved_internal_spans"]},
    "fresh_sample": {"seed": selection["seed"], "units": 20, "passed": 20, "material_sample_errors": 0,
                     "rows": [{"source_unit_key": r["source_unit_key"], "source_pages": r["source_pages"], "result": "pass", "checked": ["literal form", "printed gloss/free-translation scope", "source location", "output metadata"]} for r in selection["sample"]]},
    "findings": findings,
    "class_decisions": [
        "All10 target questions and10 answers compared with original C5 pages; source English-only Q/A controls in other chapters must not be invented as target answers.",
        "C4n3 blank lexical Gloss is correct: original PDF99 has a p/g block but no separate f translation. Its interlinear analysis remains auditable.",
        "C5 pause/emph-only rows correctly have blank lexical Gloss and retained source_analysis; no adjective/noun inferred from those labels.",
        "C5n6.1 and12.1 unresolved internal spans remain intact; printed layout does not support a forced one-to-one internal assignment.",
        "C5n5.1 sticks versus whole-translation spears is a printed layer difference, preserved without harmonization.",
        "C5n17.2 printed tiger-not-is versus separate target tokens remains a source alignment; preserve the leading hyphen in raw_gloss without treating it as target morphology.",
        "No lexical POS or tense inferred solely from unqualified English free translations. Structured feature review of the790 continuous children is a separate agent's task."
    ],
    "installation_approval": False,
    "owner_files_modified": False
}
(HERE / "independent-review-pass1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": report["status"], "fresh_sample": "20/20 passed", "finding_classes": len(findings), "reviewed_rows": 309}))
