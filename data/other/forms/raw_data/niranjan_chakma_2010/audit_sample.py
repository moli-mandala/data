"""Select a fresh reproducible transcription-review sample; never installs/builds."""
import argparse
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent


def sample(seed, count):
    payload = (ROOT / "visual-review.jsonl").read_bytes()
    reviews = [json.loads(line) for line in payload.splitlines()]
    candidates = {(r["pdf_page"], r["candidate_row"]): r for r in
                  map(json.loads, (ROOT / "candidate-cells.jsonl").read_text().splitlines())}
    selected = sorted(random.Random(seed).sample(reviews, count),
                      key=lambda r: (r["pdf_page"], r["candidate_row"]))
    return {
        "seed": seed, "sample_size": count,
        "population": len(reviews),
        "review_sha256": hashlib.sha256(payload).hexdigest(),
        "scope": "Main-wordlist transcription only; not installed-output acceptance",
        "status": "awaiting visual comparison",
        "records": [{"review_snapshot": r,
                     "candidate_snapshot": candidates[r["pdf_page"], r["candidate_row"]],
                     "audit_decision": "pending"} for r in selected],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Preserve completed audit evidence rather than overwriting it on reruns.
    with args.output.open("x") as handle:
        json.dump(sample(args.seed, args.count), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
