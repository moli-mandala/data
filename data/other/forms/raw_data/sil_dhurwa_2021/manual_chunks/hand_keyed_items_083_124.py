#!/usr/bin/env python3
"""Write the OCR-blind manual ledger for Dhurwa Appendix B, physical p. 19.

Every lexical value below was independently keyed by visual inspection of the
600-dpi page render and rechecked against targeted 1200-dpi renders. No PDF
text extraction or OCR value is an input to this script.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_083_124_hand_keyed.tsv"
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
# A slash remains the literal printed alternative separator.
PAGE_DECISIONS = [
    (83, "meat", ("cɛp", "cɛp", "cɛp", "cɛp", "cɛp")),
    (84, "fish", ("mini", "mini", "mini", "mini", "mini")),
    (85, "chicken", ("kor", "kor", "kor", "kor", "kor")),
    (86, "egg", ("kɛrba", "kɛrba", "kɛrba", "kɛrba", "kɛrba")),
    (87, "cow", ("gay", "gay", "gay", "gay", "gaj")),
    (88, "bull", ("bʌɖal", "bʌɖal", "bʌɖar", "bʌɖal", "bʌɖar")),
    (89, "buffalo", ("cɪr", "cɪr", "cɪr", "cɪru", "ciru")),
    (90, "milk", ("pel", "pel", "pel", "pel", "pelu")),
    (91, "tail", ("dʒaʈi", "jaʈi", "neŋgɖa", "purla", "purla")),
    (92, "horn", ("ko:ɖu", "ko:ɖu", "ko:ɖ", "ko:ɖ", "ko:ɖu")),
    (93, "goat", ("meva", "meva", "meva", "meva", "meja")),
    (94, "sheep", ("meɳɖa", "meɳɖa", "meɳɖa", "meɳɖa", "meɳɖa")),
    (95, "dog", ("nɛʈ:a", "nɛʈ:a", "nɛʈ:a", "nɛʈ:a", "nɛʈ:a")),
    (96, "mosquito", ("nurñyi", "kerkot:il", "kergoti", "urñyil", "urñdʒil")),
    (97, "rat", ("ɛl", "ɛl:u", "ɛl", "ɛl:u", "ɛl:u")),
    (98, "ant", ("coɖ:a", "coɖ:a", "coɖ:a", "coɖ:a", "coɖ:a")),
    (99, "spider", ("makɖa ɖaɖi", "makɖa ɖadi", "makɖa ɖaɖi", "pɛlaj baɳdur", "bala")),
    (100, "person", ("mañdʒa", "mañja", "mañja", "mañja", "mʌñe")),
    (101, "man", ("mayid", "mʌyɖ", "mʌyɖ", "mʌyɖ", "mʌyɖ")),
    (102, "woman", ("ayal", "ayal", "ayal", "ayal", "ayal")),
    (103, "child", ("pap", "pap", "pap", "pap", "papu")),
    (104, "father", ("bual", "bual", "bual", "ʈaʈa", "ʈaʈa")),
    (105, "mother", ("iya", "iya", "iya", "iya/ʈʌl", "iya")),
    (106, "brother", ("ʈol:eɖ", "ʈol:eɖ", "ʈol:eɖ", "ʈol:eɖ", "ʈol:eɖ")),
    (107, "sister", ("calal", "calal", "calal", "calal", "calal")),
    (108, "son", ("cinɖu", "cinɖ", "ciɳɖ", "ciɳɖ", "ciɳɖu")),
    (109, "daughter", ("mal", "mal", "mal", "mal", "mal")),
    (110, "husband", ("mayɖ", "mayɖ", "mayɖ", "mayɖ", "ʌreɖ")),
    (111, "wife", ("ayal", "ayal", "ayal", "ayal", "ʌre")),
    (112, "shrine", ("guɖ:i", "guɖ:i", "guɖ:i", "guɖ:i", "guɖ:i")),
    (113, "festival", ("tiyar", "ʈiyar", "ʈiyar", "ʈiyar", "ʈiyar")),
    (114, "spirit (evil)", ("bu:t", "bu:ʈ", "bu:ʈ", "bu:ʈ", "bu:ʈ")),
    (115, "day", ("pʌkʈa", "pʌkʈa", "pʌkʈa", "pʌkʈa", "pʌkʈa")),
    (116, "night", ("ciʈ:a", "ciʈ:a", "ciʈ:a", "ciʈ:a", "ciʈ:a")),
    (117, "morning", ("po:ka", "po:ka", "po:ka", "po:ka", "po:kayi")),
    (118, "noon", ("ʈɪʈ:e ɖɛlkul", "ʈɪʈ:e ɖɛlkul", "ʈɪʈ:eɖɛlkul", "ʈɪʈ:eɖɛlkul", "ʈɛlkul po:kal")),
    (119, "evening", ("aɳɖek", "aɳɖek", "aɳɖek", "aɳɖek", "aɳɖek")),
    (120, "yesterday", ("ari", "ari", "ari", "ari", "ari")),
    (121, "to day", ("ɪne", "ɪne", "ɪne", "ɪne", "ɪneni")),
    (122, "to morrow", ("ʈolli", "ʈol:i", "ʈol:i", "ʈol:i", "ʈol:i")),
    (123, "day after tomorrow", ("pɪnge", "pɪnge", "pɪnge", "pɪɖne", "pɪɖne")),
    (124, "week", ("aʈ ɖina", "ad ɖina", "aʈ ɖina", "aʈ ɖin", "aɖ ɖina")),
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
                "PDF_Page": "19",
                "Printed_Page": "14",
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
    assert len(output_rows) == 210
    assert len({(row["Item"], row["Site_Code"]) for row in output_rows}) == 210
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-keyed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
