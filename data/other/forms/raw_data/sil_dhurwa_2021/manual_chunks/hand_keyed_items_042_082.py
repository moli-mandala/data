#!/usr/bin/env python3
"""Write the OCR-blind manual ledger for Dhurwa Appendix B, physical p. 18.

Every lexical value below was independently keyed by visual inspection of the
600-dpi page render and rechecked against targeted 1200-dpi renders. No PDF
text extraction or OCR value is an input to this script.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_042_082_hand_keyed.tsv"
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
# ` | ` records two responses printed on separate lines in one cell; a slash
# remains a literal printed slash.
PAGE_DECISIONS = [
    (42, "shadow", ("niɖa", "niɖa", "niɖa", "niɖa", "niɖa")),
    (43, "rain", ("vañyi", "vañgi", "vañyi", "vañyi", "vañji")),
    (44, "water", ("niru", "nir", "niru", "niru", "nir")),
    (45, "river", ("pereɖ", "pereɖ", "pereɖ", "pereɖ", "pereɖ")),
    (46, "field", ("baya", "vaya", "vaya", "vaya/kʌmo", "vaya")),
    (47, "hill", ("kop:a", "kop:a", "kop:a", "kop:a/ke:ɳɖi", "kop:a")),
    (48, "path", ("pav", "pav", "pav", "pav", "pavu")),
    (49, "wind", ("vʌl:i", "vʌl:i", "vʌl:i", "vʌl:i", "vʌl:i")),
    (50, "fire", ("kic:u", "kic:u", "kic:u", "kic:u", "kic:u")),
    (51, "smoke", ("gu:ñyikuɖ", "gu:ñyikuɖ", "gu:ñyikuɖ", "gu:ñyi", "gu:ñji")),
    (52, "ash", ("niɖ", "niɖ", "niɖ", "niɖ", "niɖu")),
    (53, "mud", ("dʒo:ba", "jo:ba", "jo:ba", "jo:ba", "jo:ba")),
    (54, "stone", ("kɛl", "kɛl", "kɛl:u", "kɛl:u", "kɛl:u")),
    (55, "dust", ("guɳɖa", "ʈuri/guɳɖa", "guɳɖa", "guɳɖa", "guɳɖa")),
    (56, "gold", ("con", "co:n", "co:n", "co:n", "co:nu")),
    (57, "brass", ("piʈal", "piʈal", "piʈal", "piʈal", "piʈal")),
    (58, "silver", ("ru:p", "ru:p", "ru:p", "caɳʈi", "ru:p")),
    (59, "iron", ("lov:a", "lova", "lov:a", "lov:a", "lov:")),
    (60, "forest", ("mɛram", "mɛram", "ran", "mɛram", "gup:a")),
    (61, "plant", ("para", "para", "para", "para", "para")),
    (62, "thorn", ("caka", "caka", "caka", "caka", "koy:a")),
    (63, "root", ("var", "var", "var", "var", "var")),
    (64, "flower", ("pu:v", "pu:va", "pu:vu", "pu:vu", "pu:v")),
    (65, "fruit", ("pʌl", "pʌl", "pʌl", "pʌlu", "pʌl:")),
    (66, "mango", ("mɛɖi", "mɛɖi", "mɛɖi", "mɛɖi", "mɛɖu")),
    (67, "banana", ("u:lubi", "u:lubi", "u:lupʌl", "u:lu", "u:lu")),
    (68, "tamarind", ("cupari", "cupari", "cupari", "cupar", "cupari")),
    (69, "wheat", ("goŋ", "go:ŋ", "go:ŋ", "go:ŋ", "go:ŋ")),
    (70, "Ragi", ("raʈa", "raʈa", "raʈa", "raʈa", "raʈa")),
    (71, "paddy", ("vɛbɛc:iɖ", "vɛrciɖ", "vɛrciɖ", "vɛrciɖ", "vɛrci")),
    (72, "rice", ("pɛru:k", "pɛru:k", "pɛru:k", "pɛrk", "pɛru:k")),
    (73, "cooludrice", ("vey", "vey", "vey", "vey", "vey")),
    (74, "potato", ("alu", "alu", "alu", "alu", "alu")),
    (75, "brinjal", ("kakaɳɖi | maɖɖu baŋga", "kakandi", "kakaɳɖi", "kakaɳɖi", "kakaɳɖi")),
    (76, "peanut", ("cɛnay", "cɛnay", "cɛnay", "cɛnav", "cɛnav")),
    (77, "chillie", ("miri", "miri", "miri", "miri", "miri")),
    (78, "garlic", ("korul:i | lʌsu:n", "korul:i", "ul:i", "ul:i", "korul:i")),
    (79, "onion", ("ul:i", "ul:i", "go:ɳɖri", "go:ɳɖru", "go:ɳɖri ul:i")),
    (80, "tobacco", ("ɖuŋgya", "ɖuŋgya", "ɖuŋgya", "ɖuŋgya", "ɖuŋgya")),
    (81, "oil", ("nɛy", "nɛy", "nɛy", "nɛyu", "nɛy")),
    (82, "salt", ("cuppu", "cup", "cup", "cup:u", "cup:u")),
]


def rows() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item, gloss, forms in PAGE_DECISIONS:
        assert len(forms) == len(SITES) == 5
        for (site_code, site_name, column), form in zip(SITES, forms, strict=True):
            row = {
                "Item": str(item),
                "Gloss": gloss,
                "Site_Code": site_code,
                "Site_Name": site_name,
                "PDF_Page": "18",
                "Printed_Page": "13",
                "Column": column,
                "Manual_Transcription": form,
                "Review_Status": "attested",
                "Confidence": "high",
                "Uncertainty": "",
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
