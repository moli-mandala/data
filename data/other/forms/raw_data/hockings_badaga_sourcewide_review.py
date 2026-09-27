#!/usr/bin/env python3
"""Audit all cached Badaga OCR articles and prepare a page-indexed image queue.

This does not certify a transcription. It retains every unreviewed source key
for a later image-backed decision and flags records likely to need extra care.
No PDF render, database build, or network access is needed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import hockings_badaga as importer


ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / "data/other/forms/raw_data"
AUDIT = RAW / "20260818-hockings-badaga-audit.csv"
CORRECTIONS = RAW / "20260818-hockings-badaga-corrections.csv"
CACHE = ROOT / ".cache/ocr/hockings-badaga/pages"
QUEUE = RAW / "20260925-hockings-badaga-image-review-queue.tsv"
PAGES = RAW / "20260925-hockings-badaga-page-coverage.tsv"
REPORT = RAW / "20260925-hockings-badaga-sourcewide-audit.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def printed_page(pdf_page: int) -> int:
    assert pdf_page not in {443, 444}
    return pdf_page - (20 if pdf_page < 443 else 22)


def flags(row: dict[str, str]) -> list[str]:
    form, head = row["Form"], row["Raw_Head"]
    confidence = float(row["OCR_Confidence"])
    result: list[str] = []
    if confidence < 60:
        result.append("confidence_below_60")
    elif confidence < 80:
        result.append("confidence_60_to_79")
    if any(c.isdigit() for c in form):
        result.append("digit_in_form")
    if any(c in form for c in "$£{}[]_`!�"):
        result.append("suspicious_form_symbol")
    if form.casefold() not in head.casefold():
        result.append("form_not_literal_in_ocr_head")
    if "/" in head:
        result.append("printed_alternates")
    if row["Unresolved_DEDR_IDs"].strip():
        result.append("unresolved_printed_dedr")
    if len(row["Raw_OCR"]) > 1000:
        result.append("long_article_over_1000_chars")
    if int(row["Top"]) < 85 or int(row["Top"]) > 570:
        result.append("page_edge")
    if form.count("(") != form.count(")"):
        result.append("unbalanced_form_parentheses")
    return result


def main() -> None:
    audit = read_csv(AUDIT)
    corrections = read_csv(CORRECTIONS)
    reviewed = {row["Entry_Key"] for row in corrections}
    assert len(audit) == 9993 and len(reviewed) == len(corrections) == 20
    assert len({row["Entry_Key"] for row in audit}) == len(audit)
    assert reviewed <= {row["Entry_Key"] for row in audit}
    assert {row["Status"] for row in audit} == {"ingested"}
    expected_pdf_pages = set(range(21, 644)) - {443, 444}
    actual_cache_pages = {int(path.stem.removeprefix("page-")) for path in CACHE.glob("page-*.json")}
    assert actual_cache_pages == expected_pdf_pages
    cached_pages = [
        json.loads((CACHE / f"page-{page:03d}.json").read_text(encoding="utf-8"))
        for page in sorted(expected_pdf_pages)
    ]
    replay_entries, layout_exclusions = importer.extract_entries(cached_pages)
    replay_rows, replay_audit = importer.build_rows(
        replay_entries, importer.read_valid_dedr(ROOT / "data/dedr/params.csv")
    )
    assert replay_audit == audit
    assert len(replay_rows) == 16706
    assert len(layout_exclusions) == 1
    assert layout_exclusions[0]["Printed_Page"] == "1"

    all_pages: dict[int, list[dict[str, str]]] = defaultdict(list)
    queue: list[dict[str, str]] = []
    risk_counts: Counter[str] = Counter()
    for row in audit:
        page = int(row["PDF_Page"])
        assert page in expected_pdf_pages
        assert int(row["Printed_Page"]) == printed_page(page)
        assert row["Entry_Key"].endswith(
            f":p{row['Printed_Page']}:c{row['Column']}:y{int(row['Top']):04d}"
        )
        all_pages[page].append(row)
        if row["Entry_Key"] in reviewed:
            continue
        problems = flags(row)
        risk_counts.update(problems)
        critical = {"confidence_below_60", "digit_in_form", "suspicious_form_symbol", "form_not_literal_in_ocr_head"}
        high = {"confidence_60_to_79", "unresolved_printed_dedr", "long_article_over_1000_chars", "page_edge", "unbalanced_form_parentheses"}
        priority = "critical" if critical.intersection(problems) else "high" if high.intersection(problems) else "routine"
        queue.append({
            "Entry_Key": row["Entry_Key"],
            "PDF_Page": row["PDF_Page"],
            "Printed_Page": row["Printed_Page"],
            "Column": row["Column"],
            "Top": row["Top"],
            "Priority": priority,
            "Risk_Flags": "|".join(problems),
            "OCR_Confidence": row["OCR_Confidence"],
            "Raw_Head": row["Raw_Head"],
            "Form": row["Form"],
            "Review_State": "pending_image_review",
        })
    queue.sort(key=lambda row: (int(row["PDF_Page"]), int(row["Column"]), int(row["Top"])))
    assert len(queue) == 9973
    assert set(all_pages) == {page for page in expected_pdf_pages if printed_page(page) >= 3}

    with QUEUE.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(queue[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(queue)

    queue_by_page: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in queue:
        queue_by_page[int(row["PDF_Page"])].append(row)
    page_rows = []
    for pdf_page in sorted(all_pages):
        rows = all_pages[pdf_page]
        pending = queue_by_page[pdf_page]
        priorities = Counter(row["Priority"] for row in pending)
        page_rows.append({
            "PDF_Page": pdf_page,
            "Printed_Page": printed_page(pdf_page),
            "Audit_Articles": len(rows),
            "Image_Reviewed": len(rows) - len(pending),
            "Pending_Image_Review": len(pending),
            "Critical": priorities["critical"],
            "High": priorities["high"],
            "Routine": priorities["routine"],
            "Column_1": sum(row["Column"] == "1" for row in rows),
            "Column_2": sum(row["Column"] == "2" for row in rows),
            "OCR_JSON": str(CACHE / f"page-{pdf_page:03d}.json"),
        })
    with PAGES.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(page_rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(page_rows)

    report = {
        "source": "Hockings and Pilot-Raichoor 1992 Badaga-English dictionary, printed pp. 1-621",
        "status": "full_source_ocr_structural_audit; image_review_pending_for_provisional_transcriptions",
        "audit_sha256": digest(AUDIT),
        "corrections_sha256": digest(CORRECTIONS),
        "queue_sha256": digest(QUEUE),
        "page_coverage_sha256": digest(PAGES),
        "cached_ocr_pdf_pages": len(actual_cache_pages),
        "dictionary_pages_with_articles": len(all_pages),
        "raw_articles": len(audit),
        "replayed_installed_rows": len(replay_rows),
        "replayed_audit_exact_match": True,
        "layout_only_exclusions": len(layout_exclusions),
        "image_reviewed_articles": len(reviewed),
        "provisionally_installed_articles": len(queue),
        "priority_counts": dict(sorted(Counter(row["Priority"] for row in queue).items())),
        "risk_counts": dict(sorted(risk_counts.items())),
        "structural_assertions": [
            "all 621 expected PDF page OCR caches present, excluding inserted blanks 443-444",
            "all printed dictionary pages 3-621 have article records",
            "all 9,993 article source keys unique and agree with printed/PDF page, column, and y locator",
            "cached OCR replay reproduces all 9,993 checked-in audit records exactly and 16,706 uncorrected rich rows",
            "the only layout exclusion is the printed-page-1 title, not a lexical article",
            "all 20 prior image decisions resolve to an audit key",
            "all 9,973 remaining articles appear once in the page-indexed queue",
        ],
        "caveat": "Risk flags are triage indicators, not image-verified corrections. The cached OCR JSON is not a substitute for the copyrighted PDF scan.",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("raw_articles", "provisionally_installed_articles", "cached_ocr_pdf_pages", "dictionary_pages_with_articles", "priority_counts", "risk_counts")}, indent=2))


if __name__ == "__main__":
    main()
