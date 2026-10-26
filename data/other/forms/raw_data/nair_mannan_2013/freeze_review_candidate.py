"""Pin complete source-local proposal and its focused evidence for review."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "data/SOURCE_INGESTION_CHECKLIST.md").exists())


def main():
    files = [
        "physical-units-first-reading.jsonl","first-reading-status.json","prepare_first_reading.py",
        "proposed.csv","proposal-audit.jsonl","proposal-status.json","prepare_candidate.py",
        "symbol-inventory.json","proposed-settings.yaml","proposed-references.bib",
        "metadata-and-graph-proposal.json","actual-parser-review.json","validate_actual_parser.py",
        "scope/page-and-font-census.json","scope/underline-vector-census.json","scope/native-text-crosscheck.json",
    ]
    files.extend(["metadata-repair-continuity.json","apply_metadata_review.py","independent-review/complete-original-review.json"])
    test = ROOT / "data/tests/test_nair_mannan_proposal.py"
    log = ROOT / "tmp/jambu-next-source/mannan-focused.log"
    assert "5 passed" in log.read_text(), log.read_text()
    validation = json.loads((HERE / "actual-parser-review.json").read_text())
    assert validation["checks"]["NFC"]["rows"]==validation["checks"]["NFD"]["rows"]==154
    pins = {name:hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in files}
    for name,digest in validation["inputs"].items(): assert pins[name]==digest
    freeze = dict(status="frozen complete owner source-local proposal, not accepted or installed",
        counts=dict(audit_groups=140,source_witness_records=208,rows=154,Mannan_rows=123,Malayalam_comparisons=31,audit_only_witness_records=54),
        files_sha256=pins,test_sha256=hashlib.sha256(test.read_bytes()).hexdigest(),
        focused_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
        scope_note="Five chapter pages plus original p50 outside commentary; context symbol inventory and one shared –e scope expansion mean 208 records are not 208 independently printed lexical occurrences.",
        canonical_writes=False,
        pending=["root independent original/final-output review","independent complete literal review when reviewer available","guarded acceptance/installer","canonical metadata/parser verification","full build/full suite/compiled identity/graph/browser gates"])
    (HERE / "review-freeze.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(freeze["counts"]))


if __name__=="__main__": main()
