#!/usr/bin/env python3
"""Write the OCR-blind, visually hand-keyed physical-page-29 ledger."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_019_039_hand_keyed.tsv")
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

# Forms are literal hand-keyed readings made from the rendered page.  None is
# used only for cells where the source itself prints an absence marker.
ITEMS = [
    (19, "Skin", "left", 1, ("t̪ol", "tːoːl", "t̪oːl", "t̪oli", "tʃoːli", "t̪oːli")),
    (20, "Bone", "left", 2, ("ellᵉ", "koʈʈᵊ", "Koʈːə", "elɨ", "elu", "-muːɭe")),
    (21, "Heart", "left", 3, ("hɾidajʌm", "hrid̪ʌjʌm", "hɾid̪ajəm", "hɾid̪ajam", "hɾid̪ajʌm", "neɲtʃi")),
    (22, "Blood", "left", 4, ("tʃoɾeⁱ", "tʃɾe", "tʃoɾɜ", "rekt̪am", "niʈjʌr", "kuːni")),
    (23, "Village", "left", 5, ("uːɾɨ", "uːɾɨ", "uːɾᵊ", "gramam", "gramm", "uːɾɨ")),
    (24, "House", "left", 6, ("illɨ", "illɨ", "illɨ", "veedi illⁱ", "illⁱ", "mane")),
    (25, "Roof", "left", 7, ("mond̪ajʌm", "mond̪ajʌm", "mond̪æjem", "melkːuɾa", "maːɖ", "moːɳɖaja")),
    (26, "Door", "middle", 1, ("vakil", "vat̪il", "vakːil", "vaːt̪il", "baːkːil", "paɖi")),
    (27, "Firewood", "middle", 2, ("kolli", "kolːi", "kolli", "viɾagə", "kaɳakː", "puɭɭi")),
    (28, "Broom", "middle", 3, ("mɑːtʃi", "matʃːi", "mɑːtʃːi", "tʃuːl", "maːjjipː", "tʃiːpe")),
    (29, "Mortar (for grain)", "middle", 4, ("tʃigida", "tʃigida", "tʃigida", "uɾʌl", "uɾʌl", None)),
    (30, "Pestle", "middle", 5, ("dʒigar", "dʒiːger", "dʒigːer", "ulakka", "udʒʌl", None)),
    (31, "Hammer", "middle", 6, ("tʃuttikɑ", "tsitːga", "tʃuttige", "tʃuʈːika", "tʃutigʌ", "muʈʈi")),
    (32, "Knife (small)", "middle", 7, ("katːi", "kat̪i", "kæt̪i", "kat̪ːi", "kat̪ːi", "kaitʃaːku")),
    (33, "Axe", "right", 1, ("kodɑli", "maoʋ", "maɬu", "koɖali", "maɖu", "koɖli")),
    (34, "Rope", "right", 2, ("kɑjʌr", "kajʌr", "kɑjer", "kajar", "baɭⁱ", "keːri")),
    (35, "Thread", "right", 3, ("nuːl", "nuːl", "nul", "nuːl", "nuːl", "nuːli")),
    (36, "Needle", "right", 4, ("suːtʃi", "suːtʃi", "suːdʒi", "suːtʃi", "suːtʃi", "tʃuːɖi")),
    (37, "Cloth", "right", 5, ("muɳdɨ", "muɳɖi", "muɳdi", "vastram", "kuɳtᵉ", "batte")),
    (38, "Ring", "right", 6, ("mot̪iɾʌm", "mot̪iɾam", "mot̪iɾem", "mot̪iɾam", "uŋgila", "moːra")),
    (39, "Sun", "right", 7, ("suːɾjʌm", "suːɾjʌn", "suːɾjaɳ", "surjan", "suriʌn", "suːɾja")),
]


def main() -> None:
    rows = []
    for item, gloss, column, page_row, forms in ITEMS:
        for site, form in zip(SITES, forms):
            blank_marker = "NA" if item == 29 and site == "KOD" else "Nill"
            rows.append({
                "Item": str(item), "Gloss": gloss, "Site_Code": site,
                "PDF_Page": "29", "Printed_Page": "23", "Column": column,
                "Page_Row": str(page_row), "Manual_Transcription": form or "",
                "Review_Status": "attested" if form is not None else "source_blank",
                "Confidence": "high",
                "Uncertainty": "" if form is not None else f"source prints {blank_marker}",
                "Reviewer_Method": METHOD, "Reviewed_At": "2026-08-28",
                "Reviewer_Declaration": DECLARATION,
            })
    assert len(rows) == 126
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 2
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
