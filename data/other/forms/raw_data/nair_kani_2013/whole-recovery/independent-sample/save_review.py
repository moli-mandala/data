"""Record the independent original-image check of the preselected sample."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

selection_path = HERE / "selection.json"
selection = json.loads(selection_path.read_text())
for name, expected in selection["input_sha256"].items():
    assert digest(PACKAGE / name) == expected, name
assert digest(next(p for p in HERE.parents if p.name == "jambu-all") /
              "tmp/pdfs/tribal-languages-kerala-nair2013.pdf") == selection["source_pdf_sha256"]
source_images = {page: PACKAGE / f"originals/page-{page}.png" for page in (32, 33)}
for path in source_images.values():
    assert path.exists()

checked = []
for item in selection["selected"]:
    checked.append({"source_unit_key": item["source_unit_key"],
                    "printed_page": item["printed_page"],
                    "checked_entry_keys": [r["entry_key"] for r in item["actual_rows"]],
                    "result": "pass",
                    "fields_checked": ["Form", "Gloss", "Language_ID", "Source", "Tags",
                                       "literal/grammar scope and source comparison attribution"],
                    "evidence": "Original Nair page visually read at full-page and detail crop; the actual CSV row was compared to the printed target witness and the audit's attributed control."})

report = {
    "status": "fresh sampled output pass; targeted edge-control failure requires source repair before acceptance",
    "reviewer": "norton_finish",
    "authorship": "Independent of owner first reading and candidate generator; selected before opening original images",
    "selection_sha256": digest(selection_path),
    "source_pdf_sha256": selection["source_pdf_sha256"],
    "original_image_sha256": {str(page): digest(path) for page, path in source_images.items()},
    "input_sha256": selection["input_sha256"],
    "sample_seed": selection["seed"],
    "sampled_source_units": len(checked),
    "sampled_actual_rows": sum(len(x["checked_entry_keys"]) for x in checked),
    "sampled_material_errors": 0,
    "selected_unit_reviews": checked,
    "targeted_edge_controls": [
        {"source_unit_key": "nair2013kani:p32:identity:1", "result": "pass",
         "finding": "Community and self-name strings remain documentary; no target lexical gloss was manufactured."},
        {"source_unit_key": "nair2013kani:p32:vowel-shift:0", "result": "pass",
         "finding": "The Malayalam-to-Kani vowel comparison is a source claim, not a source-local etymology edge or an independently attested Tamil Nadu Kani row."},
        {"source_unit_key": "nair2013kani:p33:present:0", "result": "pass",
         "finding": "Printed Malayalam -unnu and Kani -ine are visibly bound forms. The Malayalam first-reading control retains both n underlines."},
        {"source_unit_key": "nair2013kani:p33:negative:0", "result": "fail",
         "field": "Form", "entry_key": "nair2013kani:p33:negative:0:w1",
         "staged": "-aa", "printed": "aa",
         "finding": "The original calls aa a negative suffix but prints the token as plain aa, without a leading hyphen. Retain the suffix analysis in Tags/Notes; do not insert a literal bound marker into Form. This does not affect the separately sampled negative expressions."},
    ],
    "follow_up": "Owner should preserve this report as a failed-control historical freeze, repair the one literal, regenerate its audit/CSV, then run a fresh disjoint output sample before installation.",
    "canonical_changes": False,
}
(HERE / "independent-original-review-pass1.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print("sample", len(checked), "pass; targeted controls", sum(x["result"] == "fail" for x in report["targeted_edge_controls"]), "fail")
