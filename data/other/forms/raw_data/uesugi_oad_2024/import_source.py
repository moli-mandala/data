#!/usr/bin/env python3
"""Import Uesugi & Catt (eds.), *Old Avestan Dictionary* (ILCAA, AAL 67, 2024; CC BY 4.0).

Part I (Dictionary, printed pp. 1–404 = PDF pp. 30–433) is read from the born-digital text
layer of the publisher's PDF, which is not redistributed. Run from the git root (``data/``):

    python data/other/forms/raw_data/uesugi_oad_2024/import_source.py            # dry run
    python data/other/forms/raw_data/uesugi_oad_2024/import_source.py --install  # write outputs

``--pdf`` (or ``$OAD_PDF``) overrides the default path; the importer fails clearly when the PDF
is absent or its SHA-256 differs. Extracted layout lines are cached (keyed by extractor version
and PDF hash) under the workspace ``tmp/iranic/oad/``.

One installed row per printed lemma (``◊`` non-verb, ``♣`` verb root), plus variant rows for
printed alternant headwords (``X/Y``, ``X or Y``) and ``(or X)`` spellings. Attestations
(Wingdings bullet), ``F`` cross-references, verb-stem sub-headings and section headings are
audit-only. See README.md for the full policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import extract  # noqa: E402
import model  # noqa: E402
import segment  # noqa: E402

ROOT = HERE.parents[4]  # the git root (data/)
STEM = "20261025-uesugi-oad"
PDF_DEFAULT = Path("/Users/aryamanarora/Documents/Linguistics/Indo-European/Iranic/"
                   "B542_AAL67_OldAvestan.pdf")
PDF_SHA256 = "7bb39fa3f87a5bc1400bc7243ad6da4d203ea8bad10765d13fe8a8045851b5f9"
PDF_PAGES = 558
DICT_PDF_PAGES = (30, 433)        # Part I Dictionary
PRINTED_PAGES = (1, 404)          # printed page = PDF page - 29
PART2_FIRST_PDF_PAGE = 434        # Part II title page (excluded)
CACHE = ROOT.parent / "tmp/iranic/oad/lines.jsonl"

FORM_OUTPUT = ROOT / f"data/other/forms/{STEM}.csv"
TEXT_OUTPUT = ROOT / f"data/other/entry_texts/{STEM}.csv"
AUDIT_OUTPUT = HERE / "audit.csv"
SAMPLE_OUTPUT = HERE / "sample.csv"
MANIFEST_OUTPUT = HERE / "manifest.json"
EDITORIAL = HERE / "editorial.csv"
SAMPLE_SEED = "uesugi-oad-2026-10-26-r4"
SAMPLE_SIZE = 20


def pdf_path(arg: str | None) -> Path:
    path = Path(arg or os.environ.get("OAD_PDF") or PDF_DEFAULT)
    if not path.exists():
        raise SystemExit(f"Old Avestan Dictionary PDF not found at {path}; pass --pdf or set "
                         "OAD_PDF (the PDF is not redistributed in this repository)")
    return path


def load_pages(path: Path | None = None) -> list[dict]:
    """Return extracted dictionary pages, verifying the PDF identity and page-range layout."""
    if path is None:
        path = pdf_path(None)
    digest = extract.sha256(path)
    if digest != PDF_SHA256:
        raise SystemExit(f"unexpected PDF SHA-256 {digest} for {path}")
    pages = extract.extract(path, *DICT_PDF_PAGES, cache=CACHE, pdf_sha=PDF_SHA256)
    assert len(pages) == DICT_PDF_PAGES[1] - DICT_PDF_PAGES[0] + 1 == 404
    for page in pages:
        printed = page["pdf_page"] - 29
        # p. 1 has no running head; every other page prints its number centred at the top
        assert page["printed_number"] in (printed, None), page["pdf_page"]
        assert page["printed_number"] == printed or printed == 1
    return pages


def build(pages: list[dict] | None = None):
    pages = pages if pages is not None else load_pages()
    units = segment.segment(pages)
    editorial = model.load_editorial(EDITORIAL)
    lemmas, forms, texts, audit = model.build_rows(units, editorial)
    unknown = set(editorial) - {l["uid"] for l in lemmas}
    assert not unknown, f"editorial rows without a lemma unit: {sorted(unknown)}"
    return units, lemmas, forms, texts, audit


def seeded_sample(audit: list[dict]) -> list[dict]:
    """20 installed lemmata, ordered by a seeded hash of their Unit_ID (reproducible)."""
    import hashlib

    pool = [r for r in audit if r["Status"].startswith("installed_lemma")]
    return sorted(pool, key=lambda r: hashlib.sha256(
        f"{SAMPLE_SEED}:{r['Unit_ID']}".encode()).hexdigest())[:SAMPLE_SIZE]


def write_csv(path: Path, fields: list[str], rows: list[dict], header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if header:
            writer.writeheader()
        writer.writerows(rows)


def summary(units, forms, texts, audit) -> dict:
    return {
        "unit_kinds": dict(sorted(Counter(u["kind"] for u in units).items())),
        "audit_statuses": dict(sorted(Counter(r["Status"] for r in audit).items())),
        "form_count": len(forms),
        "lemma_rows": sum(1 for f in forms if not f["Variant_Of_Key"]),
        "variant_rows": sum(1 for f in forms if f["Variant_Of_Key"]),
        "root_linked_rows": sum(1 for f in forms if f["Derivation_Parent_Keys"]),
        "entry_text_blocks": len(texts),
        "audit_count": len(audit),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    pages = load_pages(pdf_path(args.pdf))
    units, lemmas, forms, texts, audit = build(pages)
    keys = [f["Entry_Key"] for f in forms]
    assert len(keys) == len(set(keys)), "duplicate Entry_Key"
    stats = summary(units, forms, texts, audit)
    if not args.install:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return
    import manifest

    write_csv(FORM_OUTPUT, model.FORM_FIELDS, forms, header=False)
    write_csv(TEXT_OUTPUT, model.TEXT_FIELDS, texts, header=True)
    write_csv(AUDIT_OUTPUT, model.AUDIT_FIELDS, audit, header=True)
    sample = seeded_sample(audit)
    manifest.write_sample(SAMPLE_OUTPUT, sample, forms)
    manifest.write_manifest(MANIFEST_OUTPUT, stats, sample, root=ROOT, outputs={
        "forms": FORM_OUTPUT, "entry_texts": TEXT_OUTPUT, "audit": AUDIT_OUTPUT,
        "sample": SAMPLE_OUTPUT, "editorial": EDITORIAL})
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
