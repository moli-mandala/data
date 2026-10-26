"""Install only the exact independently approved Juang source-stage freeze."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "whole-source-recovery"
DATA = ROOT.parents[4]
STEM = "20260925-dalton-haldar-juang"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(require_approval=True):
    manifest_path = OUT / "frozen-manifest-20260927.json"
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        assert path.is_file() and sha(path) == expected, f"Changed or missing freeze input: {relative}"
    assert sha(DATA / manifest["test_file"]) == manifest["test_sha256"], "Changed frozen regression tests"
    if require_approval:
        approval = json.loads((OUT / "root-installation-approval-20260927.json").read_text())
        assert approval["status"] == "approved"
        assert approval["frozen_manifest_sha256"] == sha(manifest_path)
    with (OUT / "proposal.csv").open() as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 226 and all(len(r) == 15 for r in rows)
    assert len({r[10] for r in rows}) == 226
    historical = list(csv.reader((ROOT / f"{STEM}.csv").open()))
    assert len(historical) == 196
    assert [r[10] for r in rows[:196]] == [r[10] for r in historical]
    actual = DATA / "data/other/forms" / f"{STEM}.csv"
    assert sha(actual) in (manifest["canonical_before_sha256"], sha(OUT / "proposal.csv")), "Canonical input changed after freeze"
    return manifest


def install():
    manifest = validate()
    targets = {
        OUT / "proposal.csv": DATA / "data/other/forms" / f"{STEM}.csv",
        OUT / f"{STEM}.yaml": DATA / "data/other/forms" / f"{STEM}.yaml",
        OUT / "proposal-profile.txt": DATA / "conversion/dalton-haldar-juang-1872.txt",
        OUT / "proposal-audit.jsonl": OUT / "installed-audit.jsonl",
    }
    for source, target in targets.items():
        shutil.copyfile(source, target)
        assert sha(source) == sha(target)
    report = {"status":"exact approved source-stage files installed", "rows":226,
              "frozen_manifest_sha256":sha(OUT / "frozen-manifest-20260927.json"),
              "installed":{str(target.relative_to(DATA)):sha(target) for target in targets.values()},
              "shared_metadata":"Bibliography scope patch and authored checklist/ledger still require source-owner reconciliation; no DB/CLDF build performed."}
    (OUT / "installation-result-20260927.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--install", action="store_true")
    p.add_argument("--check-without-approval", action="store_true")
    args = p.parse_args()
    if args.install:
        assert not args.check_without_approval
        install()
    else:
        validate(require_approval=not args.check_without_approval)
        print("Freeze and complete-source identity checks passed; no canonical writes.")
