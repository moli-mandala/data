"""Emit source rows and audit; --install updates source files only, never builds."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parent
STEM = "20260921-niranjan-chakma"


def read_jsonl(name):
    return [json.loads(line) for line in (ROOT / name).read_text().splitlines()]


def locator(record):
    return record["pdf_page"], record["candidate_row"]


def prepare():
    reviews = read_jsonl("visual-review.jsonl")
    candidates = {locator(r): r for r in read_jsonl("candidate-cells.jsonl")}
    identities = {locator(r): r for r in json.loads((ROOT / "entry-locators.json").read_text())["records"]}
    multiples = {locator(r): r for r in json.loads((ROOT / "multiple-form-review.json").read_text())}
    policy = json.loads((ROOT / "row-policy.json").read_text())
    withheld = {locator(r): r for r in policy["withheld"]}
    if len(reviews) != len({locator(r) for r in reviews}) or set(map(locator, reviews)) != set(candidates) or set(candidates) != set(identities):
        raise ValueError("Review, raw candidate and frozen identity coverage disagree")
    output, audit = [], []
    for review in sorted(reviews, key=locator):
        key = locator(review)
        identity = identities[key]
        record = {"entry_key": identity["entry_key"], "locator": identity,
                  "review": review, "candidate": candidates[key],
                  "language": policy["language"], "dialect": None,
                  "uncertainty_types": [], "emitted_rows": []}
        if key in withheld:
            record.update(status="withheld", reason=withheld[key]["reason"])
            record["uncertainty_types"].append(withheld[key]["type"])
            audit.append(record)
            continue
        forms = [review["reviewed_chakma"]]
        if key in multiples:
            decision = multiples[key]
            if decision["source_cell"] != forms[0] or decision["decision"] != "independent-coequivalents":
                raise ValueError(f"Multiple-form decision needs review: {key}")
            forms = decision["literal_segments"]
            record["multiple_form_decision"] = decision
        if review["review_status"] == "glyph-uncertain":
            record["uncertainty_types"].append("transcription")
        gloss = review["reviewed_english"]
        if review.get("gloss_review"):
            gloss = review["gloss_review"]["editorial_suggestion"]
            record["uncertainty_types"].append("gloss")
        for number, form in enumerate(forms, 1):
            form = unicodedata.normalize("NFC", form)
            if not form.strip() or "\ufffd" in form:
                raise ValueError(f"Invalid headword at {key}")
            entry_key = f'{identity["entry_key"]}:response:{number}'
            citation = f'{policy["source_key"]}[p. {identity["printed_page"]}, Chakma column, scan y{identity["original_bbox"][1]}]'
            row = [policy["language"], "", form, gloss, form, "", "", citation,
                   "", "", entry_key, "", "", "", "uncertain" if record["uncertainty_types"] else ""]
            output.append(row)
            record["emitted_rows"].append(row)
        record.update(status="draft", reason="Source spelling retained; no phonemic or ancestry claim.")
        audit.append(record)
    keys = [r[10] for r in output]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate emitted identities")
    return output, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if args.output is None and not args.install:
        parser.error("Provide --output or --install")
    if args.output is None:
        args.output = ROOT
    rows, audit = prepare()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "draft.csv").open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    (args.output / "audit.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in audit))
    counts = Counter(ch for row in rows for ch in row[2])
    inventory = {"policy": "Preserve every source symbol; no sound values inferred; conversion disabled.",
                 "symbols": [{"character": ch, "codepoint": f"U+{ord(ch):04X}",
                              "unicode_name": unicodedata.name(ch), "count": count,
                              "examples": [row[10] for row in rows if ch in row[2]][:3]}
                             for ch, count in sorted(counts.items())]}
    (args.output / "symbol-inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
    summary = {"source_records": len(audit), "draft_rows": len(rows),
               "withheld_records": sum(r["status"] == "withheld" for r in audit),
               "uncertain_draft_rows": sum(bool(r[14]) for r in rows),
               "installed_rows": len(rows) if args.install else 0, "relationship_edges": 0}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    if args.install:
        forms_dir = ROOT.parent.parent
        with (forms_dir / f"{STEM}.csv").open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)
        (forms_dir / f"{STEM}.yaml").write_bytes((ROOT / f"{STEM}.yaml").read_bytes())
        # Bibliography and language registry are reviewed separately. No shared
        # registry, compiled data or subprocess is mutated by the importer.


if __name__ == "__main__":
    main()
