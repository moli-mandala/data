"""Exact guarded metadata revision; preserve the complete failed r1 freeze."""
import csv
import hashlib
import io
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE / "revision1"
REVIEW = HERE / "independent-review/complete-original-review.json"
REVIEW_SHA = "81911bba97c05c5f53793cd732e9ac1377bf5f67d4b616f8def6ee42c8a621e8"


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert sha(REVIEW)==REVIEW_SHA
    review = json.loads(REVIEW.read_text())
    if not OLD.exists():
        freeze = json.loads((HERE / "review-freeze.json").read_text())
        for name,digest in freeze["files_sha256"].items(): assert sha(HERE / name)==digest,name
        OLD.mkdir()
        for name in [*freeze["files_sha256"],"review-freeze.json","root-original-review-pass1-20260927451.json"]:
            src = HERE / name
            if not src.exists(): continue
            target = OLD / name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,target)
    freeze = json.loads((OLD / "review-freeze.json").read_text())
    for name,digest in freeze["files_sha256"].items(): assert sha(OLD / name)==digest,name
    for name,digest in review["inputs_sha256"].items():
        path = OLD / name if (OLD / name).exists() else HERE / name
        assert sha(path)==digest,name
    rows = list(csv.reader((OLD / "proposed.csv").open(newline="")))
    audit = [json.loads(s) for s in (OLD / "proposal-audit.jsonl").read_text().splitlines()]
    by_key = {r[10]:r for r in rows}
    decisions = review["metadata_decisions"]
    assert len(decisions)==19
    extra = [
        dict(entry_key="nair2013mannan:p46:kin-plural:0:w1",add_tags=["kinship"],evidence="p46 calls –laam a plural suffix occurring after kinship terms; semantic-domain classification, not a fabricated noun.")
    ]
    # Separate loop keeps the suffix and both language witnesses explicit.
    extra = [extra[0]] + [dict(entry_key=f"nair2013mannan:p46:final-l:2:w{i}",add_tags=["kinship"],evidence="p46 explicit daughter comparison: kinship meaning, no grammatical gender inference.") for i in (1,2)]
    delta = []
    for decision in decisions + extra:
        row = by_key[decision["entry_key"]]
        old = row[14]
        if "old_tags" in decision:
            assert old==decision["old_tags"] and row[2]==decision["form_unchanged"]
        new = " ".join(dict.fromkeys(old.split()+decision["add_tags"]))
        assert new!=old
        if "new_tags" in decision: assert set(new.split())==set(decision["new_tags"].split())
        row[14]=new
        delta.append(dict(entry_key=row[10],field="Tags",old=old,new=new,source_evidence=decision["evidence"],
            review_origin="independent exact decision" if "old_tags" in decision else "owner full-class scope extension"))
    by_delta = {d["entry_key"]:d for d in delta}
    assert len(by_delta)==22
    for record in audit:
        for witness in record["witness_dispositions"]:
            if witness["witness_key"] in by_delta:
                witness["metadata_review"] = by_delta[witness["witness_key"]]
        if any(k in by_delta for k in record["proposed_entry_keys"]):
            record["metadata_revision"] = "r2 exact independent/source-class tag overlay; original source_record unchanged"
    prior = list(csv.reader((OLD / "proposed.csv").open(newline="")))
    assert [r[10] for r in prior]==[r[10] for r in rows]
    assert all(a[:14]==b[:14] for a,b in zip(prior,rows))
    assert sum(a[14]!=b[14] for a,b in zip(prior,rows))==22
    out=io.StringIO(newline="");csv.writer(out).writerows(rows)
    (HERE / "proposed.csv").write_text(out.getvalue())
    (HERE / "proposal-audit.jsonl").write_text("".join(json.dumps(a,ensure_ascii=False,sort_keys=True)+"\n" for a in audit))
    continuity=dict(status="22 exact Tags-only source-grounded repairs; all14otherfields and154orderedkeys unchanged",
        independent_report_sha256=REVIEW_SHA,before_csv_sha256=sha(OLD / "proposed.csv"),after_csv_sha256=sha(HERE / "proposed.csv"),
        before_audit_sha256=sha(OLD / "proposal-audit.jsonl"),after_audit_sha256=sha(HERE / "proposal-audit.jsonl"),
        independent_decisions=19,owner_class_extensions=3,changes=delta,first_reading_preserved=sha(HERE / "physical-units-first-reading.jsonl")==freeze["files_sha256"]["physical-units-first-reading.jsonl"])
    (HERE / "metadata-repair-continuity.json").write_text(json.dumps(continuity,ensure_ascii=False,indent=2)+"\n")
    status = json.loads((OLD / "proposal-status.json").read_text())
    status["status"] = "r2 source-grounded metadata repair; independent fresh output acceptance pending"
    status["inputs"]["metadata_reconciler"] = sha(Path(__file__))
    status["inputs"]["independent_complete_review"] = REVIEW_SHA
    status["outputs"].update({name:sha(HERE / name) for name in ("proposed.csv","proposal-audit.jsonl","metadata-repair-continuity.json")})
    (HERE / "proposal-status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n")
    print(continuity["status"])


if __name__=="__main__": main()
