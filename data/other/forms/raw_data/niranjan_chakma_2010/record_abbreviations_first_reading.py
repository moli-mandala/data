"""Preserve every printed declaration, including repeats and ambiguous labels."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from record_brokskat_first_reading import sha

ROOT = Path(__file__).resolve().parent


def main():
    manual = ROOT / "abbreviations-manual-first-reading.tsv"
    with manual.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    counts = Counter((int(r["pdf_page"]), r["column"]) for r in rows)
    assert counts == {(77, "left"): 23, (77, "middle"): 23, (77, "right"): 18,
                      (78, "left"): 11, (78, "right"): 13}, counts
    records = []
    index = defaultdict(list)
    for page, column in counts:
        column_rows = [r for r in rows if int(r["pdf_page"]) == page and r["column"] == column]
        assert [int(r["row"]) for r in column_rows] == list(range(1, counts[page, column] + 1))
        for r in column_rows:
            assert r["abbreviation"] and r["expansion"]
            key = f"niranjan2010chakma:p{page-2}:abbreviations:{column}:row{r['row']}"
            records.append({
                "source_unit_key": key, "pdf_page": page, "printed_page": page-2,
                "column": column, "physical_row": int(r["row"]),
                "abbreviation_first_reading": r["abbreviation"],
                "expansion_first_reading": r["expansion"], "review_note": r["review_note"],
                "disposition": "source metadata declaration, not a lexical attestation",
                "review_status": "owner original reading; independent glyph and identity review pending",
                "canonical_language_id": None, "automatic_expansion_allowed": False,
            })
            index[r["abbreviation"]].append(key)
    assert len(records) == len({r["source_unit_key"] for r in records}) == 88
    assert len(index["বাং"]) == 2
    output = ROOT / "abbreviations-first-reading.jsonl"
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    status = {
        "status": "complete two-page declaration first reading; contextual mapping unapproved",
        "physical_declarations": 88, "distinct_literal_abbreviations": len(index),
        "column_counts": {f"pdf{p}:{c}": n for (p, c), n in counts.items()},
        "repeated_abbreviations": {a: keys for a, keys in index.items() if len(keys) > 1},
        "context_collision": {"চা": "declared চাক here; used for Chakma meaning scopes elsewhere; resolve each occurrence"},
        "source_pdf_sha256": "f022a2a448445cdc9840891478bd5864e2ded1ae98985332b51f0938c075842d",
        "input": {manual.name: sha(manual)}, "output": {output.name: sha(output)},
        "pending": ["independent original review of all declarations",
                    "occurrence-level reconciliation in all101 comparison groups",
                    "language/dialect/historical-stage mapping from primary evidence",
                    "separate work citations from language abbreviations"],
        "installed_rows_changed": 0,
    }
    (ROOT / "abbreviations-first-reading-status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
