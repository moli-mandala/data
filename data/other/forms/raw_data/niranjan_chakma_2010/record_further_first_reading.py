"""Record every physical row and embedded comparator before identity mapping."""
import csv
import json
from collections import Counter
from pathlib import Path
from record_brokskat_first_reading import sha

ROOT = Path(__file__).resolve().parent
EMBEDDED = {
    (100, 19): ("সং.", "জাড্য", "শীত"),
    (100, 21): ("সং.", "দোরক", "দড়ি"),
    (101, 8): ("উর্দু", "টবা", "রান্নার হাঁড়ি"),
    (101, 16): ("সেরবাদিয়া", "সিব", "সিদ্ধ"),
    (101, 27): ("রাজবংশী", "দাফনা", "উরু"),
}


def main():
    manual = ROOT / "further-manual-first-reading.tsv"
    raw_path = ROOT / "whole-remaining-raw-lines.jsonl"
    raw_pages = {r["pdf_page"]: r for r in map(json.loads, raw_path.read_text().splitlines())}
    with manual.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    counts = Counter(int(r["pdf_page"]) for r in rows)
    assert counts == {100: 24, 101: 27, 102: 22, 103: 11}, counts
    output = []
    for page, count in sorted(counts.items()):
        page_rows = [r for r in rows if int(r["pdf_page"]) == page]
        assert [int(r["row"]) for r in page_rows] == list(range(1, count + 1))
        for r in page_rows:
            row = int(r["row"])
            assert r["chakma"] and r["comparison"] and r["comparison_label"] and r["printed_meaning"]
            assert bool(r["chakma_meaning"]) == bool(r["comparison_meaning"])
            key = f"niranjan2010chakma:p{page-2}:further-comparison:row{row}"
            witnesses = []
            for column in ("chakma", "comparison"):
                ambiguous = (page, row) == (101, 9)
                witnesses.append({
                    "witness_key": f"{key}:{column}",
                    "source_language_label": "চাকমা" if column == "chakma" else r["comparison_label"],
                    "proposed_language_id": "Chakma" if column == "chakma" else None,
                    "identity_status": "existing base language" if column == "chakma" else "source label awaiting explicit mapping",
                    "raw_owner_first_reading": r[column],
                    "scoped_meaning_first_reading": None if ambiguous else r[column + "_meaning"] or r["printed_meaning"],
                    "meaning_scope": "unresolved repeated abbreviation চা" if ambiguous else "separate source meanings" if r[column + "_meaning"] else "shared printed definition",
                    "native_script": "Bengali; explicit Roman parentheses retained where present",
                    "phonemic_asserted": False, "accepted_for_installation": False,
                })
            embedded = []
            if (page, row) in EMBEDDED:
                label, form, meaning = EMBEDDED[page, row]
                embedded.append({
                    "witness_key": f"{key}:embedded:1", "source_language_label": label,
                    "raw_owner_first_reading": form, "scoped_meaning_first_reading": meaning,
                    "proposed_language_id": None,
                    "review_status": "embedded comparison first reading; glyph and identity review pending",
                    "accepted_for_installation": False,
                })
            output.append({
                "source_unit_key": key, "pdf_page": page, "printed_page": page - 2,
                "physical_row": row, "printed_meaning_first_reading": r["printed_meaning"],
                "witnesses": witnesses, "embedded_comparisons": embedded, "review_note": r["review_note"],
                "review_status": "owner original-page first reading; independent complete review pending",
                "raw_ocr_reference": {"file": raw_path.name, "pdf_page": page,
                                      "raw_tsv_sha256": raw_pages[page]["raw_tsv_sha256"]},
                "relation_policy": "comparison only; no ancestry or borrowing inferred",
                "accepted_for_installation": False,
            })
    assert len({r["source_unit_key"] for r in output}) == 84
    assert sum(len(r["embedded_comparisons"]) for r in output) == 5
    target = ROOT / "further-first-reading.jsonl"
    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output))
    status = {
        "status": "complete physical chapter first reading; identity and independent review pending",
        "physical_rows": 84, "paired_witness_cells": 168, "additional_embedded_comparisons": 5,
        "page_counts": dict(counts),
        "source_comparator_label_counts": dict(Counter(r["comparison_label"] for r in rows)),
        "rows_with_review_notes": sum(bool(r["review_note"]) for r in rows),
        "ambiguous_meaning_scope_rows": ["niranjan2010chakma:p99:further-comparison:row9"],
        "source_pdf_sha256": "f022a2a448445cdc9840891478bd5864e2ded1ae98985332b51f0938c075842d",
        "inputs": {manual.name: sha(manual), raw_path.name: sha(raw_path)},
        "output": {target.name: sha(target)},
        "pending": ["independent complete original review and glyph resolution",
                    "all language labels mapped with primary evidence; target/control classification",
                    "historical stages, generic labels and named varieties kept distinct",
                    "alternate/embedded witnesses, register and reference metadata",
                    "whole-book integration, profiles, focused checks and authorized compiled gates"],
        "installed_rows_changed": 0,
    }
    (ROOT / "further-first-reading-status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
