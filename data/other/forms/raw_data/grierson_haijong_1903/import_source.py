"""Reproduce the reviewed full Haijong chapter and comparative tables."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE.parents[4]
sys.path.insert(0, str(PACKAGE))
from prepare_full_source import prepare_native_assembly

OUTPUT = DATA / "data/other/forms/20260925-grierson-haijong.csv"
AUDIT = PACKAGE / "audit.jsonl"
PROFILE = DATA / "conversion/grierson-haijong-1903.txt"
SCAN = DATA.parent / "tmp/pdfs/lsi-v5-1/LSI-V5-1.djvu"
SCAN_SHA256 = "c1588bc4fb594caee445583402461e86c463a13fe3561dcdbc533a766ae0db67"


def generate():
    return prepare_native_assembly()


def approved_bytes():
    approval = json.loads((PACKAGE / "independent-final-output-audit-20260926.json").read_text())
    if approval["status"] != "pass" or approval["material_errors"] != 0 or len(approval["sample"]) != 20:
        raise ValueError("A passing fresh20 output audit is required")
    for name, digest in approval["input_sha256"].items():
        if hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Audited proposal changed: {name}")
    rows, audit = generate()
    stream = io.StringIO(newline="")
    csv.writer(stream).writerows(rows)
    csv_bytes = stream.getvalue().encode("utf-8")
    audit_bytes = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in audit).encode("utf-8")
    if csv_bytes != (PACKAGE / "full-proposed.csv").read_bytes() or audit_bytes != (PACKAGE / "full-proposed-audit.jsonl").read_bytes():
        raise ValueError("Regeneration differs from reviewed proposal")
    return csv_bytes, audit_bytes, (PACKAGE / "full-profile-proposed.txt").read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--check-scan", action="store_true")
    args = parser.parse_args()
    if args.check_scan:
        digest = hashlib.sha256()
        with SCAN.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != SCAN_SHA256:
            raise ValueError("Original scan hash mismatch")
    csv_bytes, audit_bytes, profile_bytes = approved_bytes()
    if args.install:
        OUTPUT.write_bytes(csv_bytes)
        AUDIT.write_bytes(audit_bytes)
        PROFILE.write_bytes(profile_bytes)
    print("896 rows, 907 audit units, 385 Native rows; full-build gates deferred")


if __name__ == "__main__":
    main()
