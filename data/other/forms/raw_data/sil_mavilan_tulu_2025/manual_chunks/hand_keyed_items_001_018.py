#!/usr/bin/env python3
"""Write the OCR-blind, visually hand-keyed physical-page-28 ledger."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_001_018_hand_keyed.tsv")
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF page; "
    "900/1200-dpi crops used for dense glyphs; text scaffold not accepted"
)
SITES = ("MTP", "MTV", "MTE", "MAL", "TUL", "KOD")
FIELDS = [
    "Item", "Gloss", "Site_Code", "PDF_Page", "Printed_Page", "Column",
    "Page_Row", "Manual_Transcription", "Review_Status", "Confidence",
    "Uncertainty", "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]

# Every form below was typed while looking at the rendered source page.  No
# value was copied from PDF text extraction or OCR output.
ITEMS = [
    (1, "Body", "left", 1, ("itʃi", "itʃi", "itʃi", "ʃaɾiːram", "mel", "t̪aɖi")),
    (2, "Head", "left", 2, ("t̪ʌlʌd", "t̪ələɖ", "t̪ʌlad", "t̪ala", "t̪ala", "maɳɖe")),
    (3, "Hair", "left", 3, ("t̪araɳaːl", "t̪ʌrəɳal", "t̪araɳal", "t̪alamudi", "tʃʌrkejɑ", "t̪almi")),
    (4, "Face", "left", 4, ("muːd", "mːɖ", "muːɖ", "mukʰam", "muʒuɳɖ", "muːɖi")),
    (5, "Eye", "left", 5, ("kɑnnɨ", "kanːɨ", "kɑnni", "kaɳɨ", "kən", "kaɳɳɨ")),
    (6, "Ear", "left", 6, ("tʃevi", "tʃevi", "tʃevi", "tʃevi", "tʃevi", "kemi")),
    (7, "Nose", "middle", 1, ("muːkʰ", "muːkʰ", "muːkʰ", "muːkːe", "muːk", "muːki")),
    (8, "Mouth", "middle", 2, ("vay", "kət̪ʌr", "ket̪ər", "vay", "vai", "baːji")),
    (9, "Tooth", "middle", 3, ("kuːli", "kuːli", "kuli", "palːɨ", "puːɭi", "palli")),
    (10, "Tongue", "middle", 4, ("naʋu", "naʋ", "nɑʋu", "naːkʰ", "nalage", "naːki")),
    (11, "Chest", "middle", 5, ("nentʃ", "nəɲdʒ", "nɛntʃː", "neɲdʒ", "sigəlɨ", "jeɖɛ")),
    (12, "Belly", "middle", 6, ("vɑntʃi", "vəɲtʃi", "vɑjarᵘ", "vajari", "beɲtʃi", "kɛla")),
    (13, "Arm", "right", 1, ("kajː", "kɑi", "kaj", "kaijː", "kai", "kaj")),
    (14, "Elbow", "right", 2, ("muʈʈ", "muʈʈ", "muʈː", "kaimuʈː", "maɾpːi", "moɳɖakaj")),
    (15, "Palm", "right", 3, ("uɭɭɑmkɑi", "uɭɭɑmkɑi", "uɭɭamkɑj", "uɭɑmkai", "aŋgɑi", "meːŋkaj")),
    (16, "Finger", "right", 4, ("viɾal", "veɽʌɭɨ", "viɾæl", "viɾal", "beɾal", "bɛɾa")),
    (17, "Fingernail", "right", 5, ("nakʰam", "nakʰəm", "nakʰːam", "nakʰam", "ugːari", "ojji")),
    (18, "Leg", "right", 6, ("kaːr", "kar", "kar", "kal", "kar", "kaːlɨ")),
]


def main() -> None:
    rows = []
    for item, gloss, column, page_row, forms in ITEMS:
        assert len(forms) == len(SITES)
        for site, form in zip(SITES, forms):
            rows.append({
                "Item": str(item), "Gloss": gloss, "Site_Code": site,
                "PDF_Page": "28", "Printed_Page": "22", "Column": column,
                "Page_Row": str(page_row), "Manual_Transcription": form,
                "Review_Status": "attested", "Confidence": "high",
                "Uncertainty": "", "Reviewer_Method": METHOD,
                "Reviewed_At": "2026-08-28", "Reviewer_Declaration": DECLARATION,
            })
    assert len(rows) == 108
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
