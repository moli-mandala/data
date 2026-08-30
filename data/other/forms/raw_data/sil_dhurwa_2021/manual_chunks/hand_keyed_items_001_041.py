#!/usr/bin/env python3
"""Write the OCR-blind manual ledger for Dhurwa Appendix B, physical p. 17.

Every lexical value below was independently keyed by visual inspection of the
600-dpi page render and rechecked against targeted 1200-dpi renders.  No PDF
text extraction or OCR value is an input to this script.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_001_041_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
FIELDS = [
    "Item", "Gloss", "Site_Code", "Site_Name", "PDF_Page", "Printed_Page",
    "Column", "Manual_Transcription", "Review_Status", "Confidence",
    "Uncertainty", "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]
SITES = [
    ("TIR", "Tiriya", "1"),
    ("NET", "Nethanar", "2"),
    ("DHA", "Dharba", "3"),
    ("KUK", "Kukanar", "4"),
    ("U5", "Unlabeled fifth printed column", "5"),
]

# Explicit hand-entered page decisions, in printed row and column order.
PAGE_DECISIONS = [
    (1, "body", ("men", "men", "menu", "men", "men")),
    (2, "head", ("ʈɛl", "ʈɛl", "ʈɛl", "ʈɛl:u", "ʈɛl")),
    (3, "face", ("mokom", "mokom", "mokom", "mokom", "mokom")),
    (4, "eye", ("bom:a", "bom:a", "bom:a", "bom:a/kʌɳ", "kʌɳ")),
    (5, "nose", ("muan", "muaɳɖ", "muaɳɖ", "muaɳɖ", "muyʌɳɖ")),
    (6, "mouth", ("coɳɖ", "coɳɖ", "coɳɖ", "coɳɖu", "coɳɖu")),
    (7, "arm", ("budʒ:am", "buj:om", "buj:om", "buj:om", "job:a")),
    (8, "palm", ("po:ɖo:m", "po:ɖom", "po:ɖom", "po:ɖom", "lʌb:a")),
    (9, "finger", ("bʌɳɖa", "vʌɳɖa", "vʌɳɖa", "vʌɳɖa", "vʌɳɖa")),
    (10, "belley [sic]", ("poʈ:a", "poʈ:a", "poʈ:a", "poʈ:a", "poʈ:a")),
    (11, "leg", ("kel", "kel", "kol", "kol", "kelu")),
    (12, "skin", ("ʈo:l", "ʈo:l", "ʈo:l", "ʈo:l", "ʈo:l")),
    (13, "bone", ("bu:la", "bu:la", "bu:la", "bu:log", "bu:la")),
    (14, "blood", ("neʈir", "neʈir", "neʈir", "neʈir", "neʈir")),
    (15, "bangle", ("cu:ɖu", "cu:ɖi", "cu:ɖi", "cu:ɖi", "cu:ɖi")),
    (16, "anklet", ("peɖu", "peɖi", "peɖi", "peɖi", "peɖi")),
    (17, "ring", ("bʌʈ:u", "vʌʈ:u", "vʌʈ", "vʌʈ:u", "vʌʈ:u")),
    (18, "footwear", ("pʌnʌy", "pʌnʌy", "pʌnʌy", "pʌnʌy", "cɛrp")),
    (19, "dhoti", ("pʌʈʌy", "gʌɳɖa", "gʌɳɖa", "gʌɳɖa", "ɖo:ʈi")),
    (20, "saree", ("pʌʈʌy", "gʌɳɖa", "gʌɳɖa", "eadi", "gʌɳɖa")),
    (21, "town", ("geɖa", "geɖa", "geɖa", "", "")),
    (22, "village", ("polu:b", "polu:b", "polu:b", "polu:b", "polu:b")),
    (23, "house", ("olek", "olek", "olek", "ole", "o:le")),
    (24, "door", ("kapaʈ", "kapaʈ", "kʌpaʈ", "kapaʈ", "kapaʈ")),
    (25, "wall", ("biʈi", "biʈi", "biʈi", "biʈi", "biʈi")),
    (26, "window", ("kiɖ kiɖi", "kiɖki", "kiɖki", "kiɖki", "kiɖki")),
    (27, "broom", ("cepiɖ", "cepiɖ", "cepiɖ", "cepiɖ", "cepiɖ")),
    (28, "cow dung", ("cʌɖpi", "cʌɖpi", "cʌɖpi", "ga:bar", "ga:bar")),
    (29, "tree", ("mɛri", "mɛri", "mɛri", "mɛri", "mɛri")),
    (30, "leaf", ("ev", "ev", "ev", "ev", "ev")),
    (31, "firewood", ("kaɖciɖ", "karciɖ", "karciɖ", "karciɖ", "karciɖ")),
    (32, "sickle", ("cɛʈal", "cɛʈal", "cɛʈ:al", "cɛʈ:al", "cɛʈ:al")),
    (33, "axe", ("teŋgya", "teŋgya", "teŋgya", "teŋgya", "teŋgya")),
    (34, "knife", ("kɛɖu:b", "kɛɖu:b", "kɛɖu:b", "kɛɖu:b", "kɛru:b")),
    (35, "rope", ("ʈoɖu", "ʈoɖu", "ʈoɖu", "ʈoɖu", "ʈoɖu")),
    (36, "plough", ("nɛŋgil", "naŋgil", "naŋgil", "naŋgil", "naŋgil")),
    (37, "bow", ("vɪl", "vɪl", "vɪl", "vɪl:u", "vɪl")),
    (38, "arrow", ("ʌmb", "ʌmb", "ʌmb", "ʌmbu", "ʌm:u")),
    (39, "sun", ("po:kal", "po:kal", "po:kal", "po:kal", "po:kal")),
    (40, "moon", ("nɛliŋ", "nɛliŋ", "nɛliŋ", "nɛliŋ", "nɛliñ")),
    (41, "sky", ("baɖor", "baɖo:r", "baɖo:r", "baɖor", "baɖor")),
]


def rows() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item, gloss, forms in PAGE_DECISIONS:
        assert len(forms) == len(SITES) == 5
        for (site_code, site_name, column), form in zip(SITES, forms, strict=True):
            status = "attested" if form else "source_blank"
            uncertainty = "" if form else "source prints double hyphen"
            row = {
                "Item": str(item),
                "Gloss": gloss,
                "Site_Code": site_code,
                "Site_Name": site_name,
                "PDF_Page": "17",
                "Printed_Page": "12",
                "Column": column,
                "Manual_Transcription": form,
                "Review_Status": status,
                "Confidence": "high",
                "Uncertainty": uncertainty,
                "Reviewer_Method": (
                    "manual visual inspection of 600-dpi rendered page; "
                    "difficult glyphs rechecked at 1200 dpi"
                ),
                "Reviewed_At": "2026-08-28",
                "Reviewer_Declaration": DECLARATION,
            }
            assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
            out.append(row)
    return out


def main() -> None:
    output_rows = rows()
    assert len(output_rows) == 205
    assert len({(row["Item"], row["Site_Code"]) for row in output_rows}) == 205
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-keyed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
