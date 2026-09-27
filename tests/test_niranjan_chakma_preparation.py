"""Review-evidence checks only; these do not certify lexical transcription."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1] / "data/other/forms/raw_data/niranjan_chakma_2010"


def test_crop_coordinates_stay_in_original_page_frame():
    spec = importlib.util.spec_from_file_location("niranjan_chakma_extract", ROOT / "extract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tsv = ("level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
           "5\t1\t1\t1\t1\t1\t20\t60\t80\t30\t90\tSea\n")
    result = module.read_lines(tsv, 900)
    assert result[0]["bbox"] == [920, 60, 1000, 90]
    assert result[0]["words"][0]["left"] == 920
    assert result[0]["status"] == "unreviewed"


def test_review_preview_is_reproducible_without_building(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "preview.py"), "--output", str(tmp_path)],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == {
        "candidate_records": 733, "unpaired_native_lines": 89,
        "irregular_cell_counts": 0, "installed_rows": 0,
    }
    for name in ("candidate-cells.jsonl", "unpaired-lines.jsonl"):
        assert (tmp_path / name).read_bytes() == (ROOT / name).read_bytes()


def test_missing_ocr_cells_are_not_silently_dropped():
    rows = [json.loads(line) for line in (ROOT / "candidate-cells.jsonl").read_text().splitlines()]
    spring = next(r for r in rows if r["pdf_page"] == 22 and r["english_raw"] == "Spring")
    previous = [json.loads(line) for line in (ROOT / "v2-candidate-cells.jsonl").read_text().splitlines()]
    old_spring = next(r for r in previous if r["pdf_page"] == 22 and r["english_raw"] == "Spring")
    assert old_spring["cells"]["chakma"] == []
    assert "chakma-cell-count:0" in old_spring["review_type"]
    assert len(spring["cells"]["chakma"]) == 1
    repairs = json.loads((ROOT / "missing-cell-review.json").read_text())
    assert len(repairs) == 14
    assert all(len(r["v4_recovered_cell"]) == 1 for r in repairs)
    assert any(r["pdf_page"] == 22 and r["english_raw"] == "Spring" for r in repairs)
    assert all(r["status"] == "unreviewed" for r in rows)
    botanical = next(r for r in rows if r["pdf_page"] == 25 and r["candidate_row"] == 3)
    assert botanical["english_raw"] == "Depterocarpus turbinatus"
    assert botanical["english_line_indices"] == [3, 4]


def test_bengali_model_comparison_preserves_english_anchors():
    before = [json.loads(line) for line in (ROOT / "v2-raw-lines.jsonl").read_text().splitlines()]
    after = [json.loads(line) for line in (ROOT / "raw-lines.jsonl").read_text().splitlines()]
    assert [r for r in before if r["column"] == "english"] == [r for r in after if r["column"] == "english"]
    manifest = json.loads((ROOT / "ocr-manifest.json").read_text())
    assert manifest["bengali_sublanguages"] == "disabled"
    assert manifest["bengali_model_sha256"] == "1cd0129288d1f74f6661c35e638e487bd146a66bb732ecd9e011e48dcb0df623"


def test_english_omission_does_not_drop_attested_native_cells():
    rows = [json.loads(line) for line in (ROOT / "candidate-cells.jsonl").read_text().splitlines()]
    row = next(r for r in rows if r.get("english_recovery_id") == "printed26-dancing-hall")
    assert row["pdf_page"] == 28 and row["candidate_row"] == 3
    assert row["english_raw"] == "Dancing hall"
    assert row["english_line_indices"] == []  # No invented full-column OCR record.
    assert row["cells"]["chakma"][0]["raw"] == "নাটঘর"
    assert row["cells"]["bengali"][0]["raw"] == "নাচঘর"
    linguistic = next(r for r in rows if r.get("english_recovery_id") == "printed35-linguistic")
    assert linguistic["pdf_page"] == 37 and linguistic["candidate_row"] == 23
    assert linguistic["english_raw"] == "Linguistic"
    assert linguistic["english_line_indices"] == []
    assert len(linguistic["cells"]["chakma"]) == len(linguistic["cells"]["bengali"]) == 1


def test_visual_review_is_bound_to_source_rows_and_preserves_uncertainty():
    import unicodedata
    candidates = {(r["pdf_page"], r["candidate_row"]): r for r in
                  (json.loads(line) for line in (ROOT / "candidate-cells.jsonl").read_text().splitlines())}
    review = [json.loads(line) for line in (ROOT / "visual-review.jsonl").read_text().splitlines()]
    keys = [(r["pdf_page"], r["candidate_row"]) for r in review]
    assert len(keys) == len(set(keys))
    assert set(keys) == set(candidates)
    for row in review:
        candidate = candidates[row["pdf_page"], row["candidate_row"]]
        assert row["english_anchor"] == candidate["english_raw"]
        assert row["original_ocr_cells"] == candidate["cells"]["chakma"]
        assert unicodedata.is_normalized("NFC", row["reviewed_chakma"])
        if row["review_status"] == "glyph-uncertain":
            assert row["uncertainty_type"] == "transcription" and row["review_note"]
    worm = next(r for r in review if r["pdf_page"] == 23 and r["candidate_row"] == 16)
    assert worm["reviewed_english"] == "Worm"
    assert worm["gloss_review"]["bengali_control"] == "গরম"
    assert worm["gloss_review"]["editorial_suggestion"] == "warm"


def test_repeated_printed_pain_entries_remain_distinct_source_records():
    review = [json.loads(line) for line in (ROOT / "visual-review.jsonl").read_text().splitlines()]
    repeated = [r for r in review if r["pdf_page"] == 35 and r["reviewed_english"] == "Pain"]
    assert [r["candidate_row"] for r in repeated] == [3, 19]
    assert [r["reviewed_chakma"] for r in repeated] == ["শুলোনি", "শুলোনি"]
    assert all(r["source_structure_note"] for r in repeated)


def test_unpaired_exclusions_preserve_every_original_line():
    raw = [json.loads(line) for line in (ROOT / "unpaired-lines.jsonl").read_text().splitlines()]
    review = [json.loads(line) for line in (ROOT / "unpaired-line-review.jsonl").read_text().splitlines()]
    assert [r["original_ocr_line"] for r in review] == raw
    for row in review:
        original = row["original_ocr_line"]
        for field in ("pdf_page", "column", "ocr_line_index"):
            assert row[field] == original[field]
        assert row["reason"] in {"section-heading", "column-heading", "page-footer"}
        assert row["decision"] == "exclude-from-lexical-rows" and row["evidence"]
    footers = [r for r in review if r["reason"] == "page-footer"]
    assert [r["printed_page"] for r in footers] == list(range(19, 47))
    assert all(r["original_ocr_line"]["bbox"][1] >= 2125 for r in footers)


def test_second_review_retains_history_and_does_not_clear_unresolved_flags():
    rows = {(r["pdf_page"], r["candidate_row"]): r for r in
            map(json.loads, (ROOT / "visual-review.jsonl").read_text().splitlines())}
    followup = list(map(json.loads, (ROOT / "second-glyph-review.jsonl").read_text().splitlines()))
    keys = [(r["pdf_page"], r["candidate_row"]) for r in followup]
    assert len(keys) == len(set(keys))
    for record, key in zip(followup, keys):
        row = rows[key]
        assert record["previous_reading"] and record["previous_note"] and record["evidence"]
        assert record["reading"] == row["reviewed_chakma"]
        assert record["decision"] == row["second_review"]["decision"]
        if record["decision"] == "retain-uncertainty":
            assert row["review_status"] == "glyph-uncertain"
            assert row["uncertainty_type"] == "transcription"
        else:
            assert record["decision"] == "resolved" and record["note"]
            assert row["review_status"] == "visually-reviewed"


def test_failed_audit_corrections_preserve_original_evidence():
    rows = {(r["pdf_page"], r["candidate_row"]): r for r in
            map(json.loads, (ROOT / "visual-review.jsonl").read_text().splitlines())}
    audit = json.loads((ROOT / "transcription-audit-2026092108.json").read_text())
    failures = [r for r in audit["records"] if r["audit_decision"] == "material-error"]
    assert audit["material_errors"] == len(failures) == 2
    assert rows[23, 12]["reviewed_chakma"] == "দিবুচ্যা"
    assert rows[40, 14]["reviewed_english"] == "Sclera"
    assert rows[40, 14]["english_anchor"] == "| Sclera"
    assert all("|" not in r["reviewed_english"] for r in rows.values())
    assert failures[0]["review_snapshot"]["reviewed_chakma"] == "দ্বিবুচ্যা"
    assert failures[1]["review_snapshot"]["reviewed_english"] == "| Sclera"
