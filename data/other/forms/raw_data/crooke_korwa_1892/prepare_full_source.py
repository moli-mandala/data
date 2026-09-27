"""Stage complete Crooke responses for independent review; never install or build."""

import csv
import hashlib
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIALECT = "dialect:kw:crooke-1892-mirzapur:Mirzapur%20Korwa"


def build():
    inventory_path = ROOT / "reviewed_inventory.tsv"
    review = json.loads((ROOT / "root-full-literal-review-20260926.json").read_text())
    assert hashlib.sha256(inventory_path.read_bytes()).hexdigest() == review["inventory_sha256"]
    inventory = list(csv.DictReader(inventory_path.open(), delimiter="\t"))
    decisions = {(r["page"], r["item"]): r for r in review["records"]}
    independent_path = ROOT / "independent-full-literal-comparison-review-20260926.json"
    independent = json.loads(independent_path.read_text())
    for name in ("root-full-literal-review-20260926.json", "comparison-first-reading.jsonl"):
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == independent["input_hashes"][name]
    for name, digest in independent["original_images"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    second_read = {(r["printed_page"], r["item"]): r for r in independent["records"]}
    assert second_read.keys() == decisions.keys()
    comparisons = {
        (r["printed_page"], r["item"]): r
        for r in map(json.loads, (ROOT / "comparison-first-reading.jsonl").read_text().splitlines())
    }
    rows, audit = [], []
    for old in inventory:
        page, item = int(old["page"]), int(old["item"])
        decision = decisions[page, item]
        assert decision["before"] == old["form"]
        second = second_read[page, item]
        assert second["reviewed_form"] == decision["after"]
        assert second["reviewed_gloss"] == old["gloss"]
        form = unicodedata.normalize("NFC", second.get("after", decision["after"]))
        gloss = old["gloss"]
        tags = [DIALECT]
        if " " in form:
            tags.append("multiword-expression")
        if gloss.startswith("to ") or gloss in ("come", "the rice is cooking"):
            tags.append("verb")
        # Comparison attribution is source evidence; old exclusion reasons are not Notes.
        notes = old["note"] if old["note"].startswith("Crooke ") else ""
        if not notes and "Hindi comparison" in old["note"]:
            notes = "Crooke gives a Hindi comparison; no donor relation is inferred."
        comparison = comparisons.get((page, item))
        if comparison:
            language = comparison["comparison_language"]
            comparator = comparison["comparison_form"]
            if comparator:
                qualifier = "tentatively compares" if comparison["tentative_source_comparison"] else "compares"
                notes = f"Crooke {qualifier} {language} {comparator}"
                if comparison["comparison_gloss"]:
                    notes += f" ‘{comparison['comparison_gloss']}’"
                notes += "."
            else:
                notes = f"Crooke marks a {language} comparison without a separate comparison form."
        uncertainties = []
        if (page, item) == (126, 17):
            uncertainties.append("transcription: tiny speck beneath final m cannot securely be distinguished from an intentional diacritic; plain m retained provisionally after two original-image readings")
            notes = (notes + " The source has a tiny mark beneath final m; plain m is provisional because the image does not distinguish a diacritic from a blemish.").strip()
        if uncertainties:
            tags.append("uncertain")
        key = decision["entry_key"]
        row = ["kw", "", form, gloss, "", "", notes,
               f"crooke1892korwa[p. {page}, item {item}]", "", "", key,
               "", "", "", " ".join(tags)]
        rows.append(row)
        audit.append({**decision, "status": "selected", "source_form": old["form"],
                      "printed_page": page, "scan_page": page + 10,
                      "selected_form": form, "previous_reason": old["note"],
                      "uncertainty_types": uncertainties, "entry_keys": [key],
                      "comparison_evidence": comparison,
                      "independent_review": second,
                      "independent_review_sha256": hashlib.sha256(independent_path.read_bytes()).hexdigest(),
                      "source_image": f"printed-p{page}.png",
                      "source_image_sha256": review["original_images"][f"printed-p{page}.png"],
                      "review_state": "two complete original-image readings reconciled"})
    assert len(rows) == len(audit) == len(decisions) == 123
    assert len({r[10] for r in rows}) == 123
    legacy = list(csv.reader((ROOT / "20260925-crooke-korwa-mirzapur.csv").open()))
    assert len(legacy) == 93 and {r[10] for r in legacy} <= {r[10] for r in rows}
    # Keep the existing stem-based row order; append recovered attestations.
    by_key = {r[10]: r for r in rows}
    legacy_keys = {r[10] for r in legacy}
    rows = [by_key[r[10]] for r in legacy] + [r for r in rows if r[10] not in legacy_keys]
    return rows, audit


def main():
    rows, audit = build()
    with (ROOT / "full-proposed.csv").open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    (ROOT / "full-proposed-audit.jsonl").write_text("".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in audit))
    # Retain established conversion rules; cover the two newly recovered underdots
    # literally rather than guessing a different phonological interpretation.
    profile = ROOT.parents[4] / "conversion/crooke-korwa-1892.txt"
    text = profile.read_text()
    for glyph in ("ṛ", "ṭ"):
        if not any(line.startswith(glyph + "\t") for line in text.splitlines()):
            text += f"{glyph}\t{glyph}\n"
    (ROOT / "full-proposed-profile.txt").write_text(text)
    summary = {"status": "proposal_only_independent_review_and_metadata_pending",
               "units": 123, "forms": 123, "legacy_keys_preserved": 93,
               "recovered_structural_holds": 30, "literal_corrections": 13,
               "remaining_transcription_uncertainties": 1,
               "pending": ["Final metadata", "Profile coverage and scoped parser", "Fresh final output audit"],
               "hashes": {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()
                          for n in ["full-proposed.csv", "full-proposed-audit.jsonl",
                                    "full-proposed-profile.txt"]}}
    (ROOT / "full-proposed-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("123 complete responses staged; 93 legacy keys preserved; canonical files unchanged")


if __name__ == "__main__":
    main()
