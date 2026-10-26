"""Pin root's manual independent original-page findings for Adiya revision2."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
CSV_SHA = "a337b89f9f13cfbd70d32286b7f68916d708530a05d32e683d91a2e65585739b"
AUDIT_SHA = "4b330670fc5f61aa19039ef9e3c0774fe1af8f91df7c90701656aca82ea98e11"
assert hashlib.sha256((PACKAGE / "proposed.csv").read_bytes()).hexdigest() == CSV_SHA
assert hashlib.sha256((PACKAGE / "proposal-audit.jsonl").read_bytes()).hexdigest() == AUDIT_SHA
rows = list(csv.reader((PACKAGE / "proposed.csv").open()))
selected = []
for page in range(4, 8):
    pool = [r for r in rows if f":p{page}:" in r[10]]
    selected += sorted(pool, key=lambda r: hashlib.sha256(("20260927430:" + r[10]).encode()).hexdigest())[:5]
assert len(selected) == 20
failures = {
    "nair2013adiya:p5:plural-morphology:slot4": "Original p5 prints –kaḷ (en dash); candidate prints -kaḷ (ASCII hyphen). Literal dash was normalized despite literal profile.",
    "nair2013adiya:p6:affinal-address-mi:slot1": "Original p6 prints –mi (en dash); candidate prints -mi (ASCII hyphen).",
}
report = {
    "reviewer": "root",
    "status": "literal repair required before acceptance",
    "csv_sha256": CSV_SHA,
    "audit_sha256": AUDIT_SHA,
    "source_pdf_sha256": "4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af",
    "render_manifest": json.loads((HERE / "render-manifest.json").read_text()),
    "seed": "20260927430",
    "sampling": "SHA-256(seed:Entry_Key), first five sorted within each p4/p5/p6/p7 stratum",
    "actual_csv_rows_checked": 20,
    "sample_passes": 18,
    "sample_material_errors": 2,
    "material_errors_including_extra_control": 3,
    "sampled": [{"entry_key": r[10], "form": r[2], "gloss": r[3], "tags": r[14],
                 "verdict": "repair" if r[10] in failures else "pass",
                 "original_observation": failures.get(r[10], "Original printed form, language column, gloss and scoped grammar agree; no material discrepancy.")}
                for r in selected],
    "additional_control": {"entry_key": "nair2013adiya:p5:plural-morphology:slot1",
                           "candidate": "-a", "printed": "a in a-suffix",
                           "finding": "No leading hyphen is printed on the source marker. Preserve bound-suffix tags without inventing a leading dash."},
    "graph_review": {"proposed_edges": 12, "source_support": "All twelve proposals are supported by original slash groups or explicit free variation on p5/p7. Live endpoints are in the frozen CSV; held address/reference and repeated paradigm cells remain prudent non-edges.",
                     "approval": "Source relation scope accepted; apply only after source-literal repair and final freeze."},
    "metadata_review": {"source_identity": "Adiya self-designation and Wayanad scope agree with existing Ravula identity; use a source-qualified lect under Ravula without a fabricated point.",
                        "dialect_glottocode": "ravu1237 establishes the parent Ravula language, not a separately indexed Wayanad source dialect; leave proposed dialect code blank.",
                        "p8_bibliography": "All four auxiliary author/title/year records agree with printed p8 and remain reported-only; brace title capitalization and check rendered references."},
    "required_followup": ["Version revision2 before edits", "Complete original bound-marker/dash sweep", "Targeted independent reread of all changed literal fields", "Final graph-bearing fresh-output sample", "Guarded source-stage installation and canonical checks"],
    "limits": "Twenty output rows, one extra source marker control, all twelve proposed variant groups, identity and bibliography reviewed; this does not certify unsampled fields or full-build/browser gates."
}
(HERE / "original-review-pass1-20260927430.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
