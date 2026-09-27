"""Reproduce the fully reviewed Crooke glossary; install only reviewed bytes."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[4]
STEM = "20260925-crooke-korwa-mirzapur"
DIALECT = "dialect:kw:crooke-1892-mirzapur:Mirzapur%20Korwa"
spec = importlib.util.spec_from_file_location("crooke_complete", ROOT / "prepare_full_source.py")
complete = importlib.util.module_from_spec(spec)
spec.loader.exec_module(complete)
build = complete.build


def write(install=False):
    review = json.loads((ROOT / "final-output-review-20260926.json").read_text())
    assert review["status"] == "pass" and review["material_errors"] == 0
    for name, digest in {**review["hashes"], **review["original_images"]}.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    rows, audit = build()
    buf = io.StringIO(newline="")
    csv.writer(buf).writerows(rows)
    assert buf.getvalue().encode() == (ROOT / "full-proposed.csv").read_bytes()
    audit_bytes = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in audit).encode()
    assert audit_bytes == (ROOT / "full-proposed-audit.jsonl").read_bytes()
    if install:
        targets = {
            "full-proposed.csv": DATA / f"data/other/forms/{STEM}.csv",
            "full-proposed-audit.jsonl": ROOT / "audit.jsonl",
            "full-proposed-profile.txt": DATA / "conversion/crooke-korwa-1892.txt",
            "full-proposed-source.yaml": DATA / f"data/other/forms/{STEM}.yaml",
        }
        for name, target in targets.items():
            target.write_bytes((ROOT / name).read_bytes())
    print(f"123 complete responses verified; installed={install}; no database built")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    write(parser.parse_args().install)
