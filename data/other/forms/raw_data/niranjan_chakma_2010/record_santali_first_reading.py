"""Preserve all Santali–Chakma comparison cells without installing proposals."""
import csv
import json
from collections import Counter
from pathlib import Path
from record_brokskat_first_reading import sha

ROOT = Path(__file__).resolve().parent


def main():
    manual = ROOT / "santali-manual-first-reading.tsv"
    raw_path = ROOT / "whole-remaining-raw-lines.jsonl"
    raw_pages = {r["pdf_page"]: r for r in map(json.loads, raw_path.read_text().splitlines())}
    with manual.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    counts = Counter(int(r["pdf_page"]) for r in rows)
    assert counts == {96: 26, 97: 26, 98: 27, 99: 17}, counts
    output = []
    for page, count in sorted(counts.items()):
        page_rows = [r for r in rows if int(r["pdf_page"]) == page]
        assert [int(r["row"]) for r in page_rows] == list(range(1, count + 1))
        for r in page_rows:
            assert r["santali"] and r["chakma"] and r["printed_meaning"]
            assert bool(r["santali_meaning"]) == bool(r["chakma_meaning"])
            key = f"niranjan2010chakma:p{page-2}:santali-comparison:row{r['row']}"
            witnesses = []
            for column, language, registry in [("santali", "সাঁওতালি", "sa"), ("chakma", "চাকমা", "Chakma")]:
                meaning = r[column + "_meaning"] or r["printed_meaning"]
                if page == 96 and r["row"] == "24":
                    meaning = "পতিত জমি"
                witnesses.append({
                    "witness_key": f"{key}:{column}", "source_language_label": language,
                    "proposed_language_id": registry, "raw_owner_first_reading": r[column],
                    "scoped_meaning_first_reading": meaning,
                    "meaning_scope": "language-specific sense scope" if r[column + "_meaning"] else "shared printed definition",
                    "native_script": "Bengali", "phonemic_asserted": False,
                    "accepted_for_installation": False,
                })
            output.append({
                "source_unit_key": key, "pdf_page": page, "printed_page": page - 2,
                "physical_row": int(r["row"]), "printed_meaning_first_reading": r["printed_meaning"],
                "witnesses": witnesses, "review_note": r["review_note"],
                "review_status": "owner original-page first reading; independent complete review pending",
                "raw_ocr_reference": {"file": raw_path.name, "pdf_page": page,
                                      "raw_tsv_sha256": raw_pages[page]["raw_tsv_sha256"]},
                "relation_policy": "comparison only; no ancestry or borrowing inferred",
                "accepted_for_installation": False,
            })
    assert len({r["source_unit_key"] for r in output}) == 96
    target = ROOT / "santali-first-reading.jsonl"
    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output))
    status = {
        "status": "complete chapter first reading; not an installed or certified ingestion",
        "physical_rows": 96, "paired_witness_cells": 192, "page_counts": dict(counts),
        "language_specific_meaning_rows": sum(bool(r["santali_meaning"]) for r in rows),
        "rows_with_review_notes": sum(bool(r["review_note"]) for r in rows),
        "source_pdf_sha256": "f022a2a448445cdc9840891478bd5864e2ded1ae98985332b51f0938c075842d",
        "inputs": {manual.name: sha(manual), raw_path.name: sha(raw_path)},
        "output": {target.name: sha(target)},
        "pending": ["independent original review of all 96 rows and glyph flags",
                    "source-supported alternate expansions and grammatical scope",
                    "resolve প্রা. বাং. comparison in printed page 94 row 24",
                    "complete whole-book integration, metadata, profiles, focused checks and authorized compiled gates"],
        "installed_rows_changed": 0,
    }
    (ROOT / "santali-first-reading-status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
