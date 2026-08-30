#!/usr/bin/env python3
"""Write the OCR-blind, visually hand-keyed items-190--208 ledger."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_190_208_hand_keyed.tsv")
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

# Literal readings entered only after direct visual inspection of 1200-dpi
# rendered-page crops. This final source block contains no blanks or unresolved
# cells. Physical p.38 has only item 208; items 209--210 do not exist.
ITEMS = [
    (190, "He killed", 37, 31, "left", 4, ("kereŋɖu", "kerukːo", "kerekːed", "konnu", "kerpaɳa", "kolli")),
    (191, "It is flying", 37, 31, "left", 5, ("parəguⁱ", "paregeɳu", "paregaɲu", "parakːunːu", "rapəɳa", "paːri")),
    (192, "Walk", 37, 31, "left", 6, ("nədʌt̪in̪d̪e", "nədakːunnu", "nədət̪igɜt̪e", "nad̪akːu", "nadapəⁱ", "nad̪ə")),
    (193, "Run", 37, 31, "left", 7, ("paːjiŋdu", "pajugeɳu", "pejit̪ɜ", "od̪uka", "belt̪unⁱ", "oːd̪i")),
    (194, "Go", 37, 31, "middle", 1, ("poːɳɖ", "po", "poː", "po", "po", "poː")),
    (195, "Come", 37, 31, "middle", 2, ("pɑlla", "ba", "pɑ", "va", "vaɭːa", "baː")),
    (196, "Speak", 37, 31, "middle", 3, ("paɳə", "paɳɳu", "paɲːɜ", "parajː", "paɭⁱ", "pari")),
    (197, "He hears", 37, 31, "middle", 4, ("keɳᵊ", "keɳᵊ", "kɜɲege", "ket̪ːu", "ken̪t̪ːe", "kəːɭi")),
    (198, "He saw", 37, 31, "middle", 5, ("tʃuɳɖ", "tʃugᵊ", "tʃːen̪d̪", "kaɳɖu", "tʃugt̪ːe", "noːt̪t̪i")),
    (199, "I (1st sg)", 37, 31, "middle", 6, ("enu", "enu", "enu", "ɲan", "eːnⁱ", "naːni")),
    (200, "You (2nd sg, inf.)", 37, 31, "middle", 7, ("idʒe", "ijːⁱ", "iy", "ni", "iː", "niː")),
    (201, "You (2nd sg, form.)", 37, 31, "right", 1, ("nigʌɭ", "nikːarⁱ", "nɨkːl", "t̪aŋgaɭ", "irə", "niŋga")),
    (202, "He (3rd sg, m.)", 37, 31, "right", 2, ("aːji", "aːji", "ayi", "avan", "aje", "aũo")),
    (203, "She (3rd sg, f.)", 37, 31, "right", 3, ("ajjini", "ad̪u", "ad̪eu", "avaɭ", "aːɭⁱ", "aʋa")),
    (204, "We", 37, 31, "right", 4, ("ekːalam", "namːa", "ekː", "namːaɭ", "eŋkːaɭa", "naŋga")),
    (205, "We (two)", 37, 31, "right", 5, ("nɑnnːa ɾʌɖːʌɭ", "namːa", "nemmə", "ɲaŋaɭ", "ekkaɭʌ", "naŋga")),
    (206, "You (2nd pl)", 37, 31, "right", 6, ("nikːʌr", "nikːerelːam", "nikːərar", "niŋaɭ", "eŋkaɭʌl", "niŋga")),
    (207, "They (3rd pl)", 37, 31, "right", 7, ("aʋu", "aoʋ", "ɑou", "avar", "agəɭⁱ", "ajiŋga")),
    (208, "Dust", 38, 32, "left", 1, ("pod̪i", "podi", "podɨ", "podi", "podi", "d̪uːɭi")),
]


def main() -> None:
    rows = []
    for item, gloss, pdf_page, printed_page, column, page_row, forms in ITEMS:
        for site, form in zip(SITES, forms):
            rows.append({
                "Item": str(item), "Gloss": gloss, "Site_Code": site,
                "PDF_Page": str(pdf_page), "Printed_Page": str(printed_page),
                "Column": column, "Page_Row": str(page_row),
                "Manual_Transcription": form,
                "Review_Status": "attested", "Confidence": "high",
                "Uncertainty": "", "Reviewer_Method": METHOD,
                "Reviewed_At": "2026-08-28",
                "Reviewer_Declaration": DECLARATION,
            })
    assert len(rows) == 114
    assert all(row["Review_Status"] == "attested" for row in rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
