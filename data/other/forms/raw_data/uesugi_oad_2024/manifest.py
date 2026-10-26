"""Seeded raw-vs-output sample and the package manifest."""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVIEW = HERE / "sample-review.csv"
# Earlier seeded rounds (20 installed lemmata each); every error class found was fixed in the
# parser or the editorial table before the next fresh draw.
AUDIT_ROUNDS = [
    {"seed": "uesugi-oad-2026-10-25", "material_errors": 1,
     "classes": ["unquoted primary definition before the quoted gloss (haoma-): 8 entries "
                 "fixed in editorial.csv"]},
    {"seed": "uesugi-oad-2026-10-26-r2", "material_errors": 1,
     "classes": ["printed unclosed parenthesis hid a noun sense (arədra-): editorial fix; "
                 "all 7 unbalanced headers flagged in the audit",
                 "non-material: spaced label '(rel. adv.)' not tagged; parser fixed"]},
    {"seed": "uesugi-oad-2026-10-26-r3", "material_errors": 1,
     "classes": ["quotes glossing a word cited in unparenthesised prose were taken as senses "
                 "(maidiiōi.mā̊ŋha-): parser now stops at prose citing a bold word"]},
    {"seed": "uesugi-oad-2026-10-26-r4", "material_errors": 0, "classes": []},
]
SAMPLE_FIELDS = ["Unit_ID", "Printed_Page", "Column", "Raw_Text", "Entry_Key", "Form",
                 "Gloss", "Tags", "Variant_Rows", "Derivation_Parent_Keys",
                 "Material_Error", "Review_Note"]


def _reviews() -> dict[str, dict]:
    if not REVIEW.exists():
        return {}
    with REVIEW.open(encoding="utf-8", newline="") as handle:
        return {row["Unit_ID"]: row for row in csv.DictReader(handle)}


def write_sample(path: Path, sample: list[dict], forms: list[dict]) -> None:
    by_key = {f["Entry_Key"]: f for f in forms}
    variants: dict[str, list[str]] = {}
    for f in forms:
        if f["Variant_Of_Key"]:
            variants.setdefault(f["Variant_Of_Key"], []).append(f["Form"])
    reviews = _reviews()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in sample:
            form = by_key[row["Entry_Key"]]
            review = reviews.get(row["Unit_ID"], {})
            writer.writerow({
                "Unit_ID": row["Unit_ID"], "Printed_Page": row["Printed_Page"],
                "Column": row["Column"], "Raw_Text": row["Raw_Text"][:600],
                "Entry_Key": row["Entry_Key"], "Form": form["Form"], "Gloss": form["Gloss"],
                "Tags": form["Tags"], "Variant_Rows": " | ".join(variants.get(row["Entry_Key"], [])),
                "Derivation_Parent_Keys": form["Derivation_Parent_Keys"],
                "Material_Error": review.get("Material_Error", "unreviewed"),
                "Review_Note": review.get("Review_Note", ""),
            })


def write_manifest(path: Path, stats: dict, sample: list[dict], root: Path,
                   outputs: dict[str, Path]) -> None:
    import import_source as src
    import model

    reviews = _reviews()
    sample_ids = [row["Unit_ID"] for row in sample]
    errors = [u for u in sample_ids if reviews.get(u, {}).get("Material_Error") == "yes"]
    unreviewed = [u for u in sample_ids if u not in reviews]
    manifest = {
        "source_id": model.SOURCE_ID,
        "bibliography": ("Uesugi, Heindio (ed.) & Adam Alvah Catt (editorial supervisor). 2024. "
                         "Old Avestan Dictionary. (Asian and African Lexicon 67.) Tokyo: Research "
                         "Institute for Languages and Cultures of Asia and Africa (ILCAA), Tokyo "
                         "University of Foreign Studies. ISBN 978-4-86337-542-0."),
        "licence": "CC BY 4.0 (http://creativecommons.org/licenses/by/4.0/), PDF p. 3",
        "pdf": {"filename": src.PDF_DEFAULT.name, "sha256": src.PDF_SHA256,
                "pages": src.PDF_PAGES, "redistributed": False,
                "producer": "Adobe Acrobat Pro 24.5 (born-digital text layer)"},
        "page_ranges": {
            "front_matter_pdf": [1, 29],
            "dictionary_pdf": list(src.DICT_PDF_PAGES),
            "dictionary_printed": list(src.PRINTED_PAGES),
            "printed_equals_pdf_minus": 29,
            "part_ii_first_pdf": src.PART2_FIRST_PDF_PAGE,
            "assertions": ("404 dictionary pages; every running head except printed p. 1 "
                           "carries printed page = PDF page - 29"),
        },
        "scope": {
            "included": "Part I Dictionary, printed pp. 1-404: every ◊ and ♣ lemma",
            "excluded": [
                "Part II Text and Translation (PDF pp. 434-558): Yasna 27.13-15, Gāθās Y28-53, "
                "54.1 with interlinear glosses",
                "front matter (acknowledgements, preface, abbreviations, introduction, "
                "references)",
                "inflected attestations, passage citations and translations (audit only)",
                "F cross-references from inflected forms to lemmata (audit only)",
                "verb-stem sub-headings (audit only)",
                "Vedic, Sanskrit, Young Avestan and other comparanda inside entries (prose only)",
            ],
        },
        "extraction": {
            "method": ("pdfplumber character layer: two columns split at the page midpoint, "
                       "lines clustered by top (±5 pt), combining marks re-attached to their "
                       "stream/geometric base, NFC; font classes mark headwords (bold), F-line "
                       "forms (bold italic), and the Wingdings/Verdana structural markers"),
            "extractor_version": __import__("extract").EXTRACTOR_VERSION,
            "ocr": False,
            "editorial_table": str(outputs["editorial"].relative_to(root)),
        },
        "language_model": ("all rows Language_ID Av; Old Avestan lemmata carry the proposed "
                           f"dialect tag {model.OAV_TAG}; lemmata printed in square brackets "
                           "(non-Old-Avestan material, e.g. Y27.15, Y42) carry no dialect tag"),
        "outputs": {name: str(p.relative_to(root)) for name, p in outputs.items()},
        "counts": stats,
        "seeded_sample_history": AUDIT_ROUNDS,
        "seeded_sample": {"seed": src.SAMPLE_SEED, "size": len(sample_ids),
                          "unit_ids": sample_ids, "material_errors": len(errors),
                          "unreviewed": len(unreviewed)},
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
