#!/usr/bin/env python3
"""Write the OCR-blind, visually hand-keyed items-120--159 ledger."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_120_159_hand_keyed.tsv")
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
# rendered-page crops. None represents the source's printed "Nill" marker.
ITEMS = [
    (120, "Today", 33, 27, "right", 4, ("ini", "ini", "inː", "inːə", "ini", "in̪d̪i")),
    (121, "Tomorrow", 33, 27, "right", 5, ("elːe", "jelːeⁱ", "jelːe", "naɭe", "elːe", "naːɭe")),
    (122, "Week", 33, 27, "right", 6, ("t̪iŋgʌɭ", "aːɕtʃa", "ɑːɻtʃːa", "aːɻtʃa", "ora", "ʋaːra")),
    (123, "Month", 33, 27, "right", 7, ("maːsʌm", "masam", "mæːsam", "masam", "siŋguːɭ", "t̪iŋga")),
    (124, "Year", 34, 28, "left", 1, ("aːɳɖɨ", "kolːʌm", "aːɳɖi", "varṣam", "varza", "kaːla")),
    (125, "Old (things)", 34, 28, "left", 2, ("paɕʌjat̪u", "poʈːaɭi", "poʈeɭɨ", "paɻajat̪", "parat̪ːᵘ", "paɭeja")),
    (126, "New", 34, 28, "left", 3, ("potʃen̪d̪u", "potʃet̪u", "potʃːəndu", "put̪ijat̪", "peʃt̪ː", "puɖija")),
    (127, "Good", 34, 28, "left", 4, ("eɖɖet̪u", "nalːat̪ᵊ", "aɖɜtu", "nalːat̪", "aɖe", "nalla")),
    (128, "Bad", 34, 28, "left", 5, ("eɖɖed̪et̪u", "poʈʈaɭi", "adedet̪u", "moʃam", "bod̪tʃet̪nu", "kətta")),
    (129, "Wet", 34, 28, "left", 6, ("tʃaɳɖɨ", "tʃaɳɖi", "tʃæɳɖi", "n̪anajːa", "tʃaɳɖi", "punɖə")),
    (130, "Dry", 34, 28, "left", 7, ("uɭuŋgɑlɨ", "uɭuŋgal", "uɭœŋgali", "unaŋija", "ʃuŋgʰᵉ", "oɳaki")),
    (131, "Long", 34, 28, "middle", 1, ("mɑɭɭɑgⁱ", "perijʌt̪", "mɛɭːag", "niːn̪d̪at̪", "uɖat̪", "uɖɖa")),
    (132, "Short", 34, 28, "middle", 2, ("kumuŋʌt̪", "kuŋːugat̪i", "kamuŋgat̪", "kuraŋːa", "elːⁱ", "kuɭɭa")),
    (133, "Hot things", 34, 28, "middle", 3, ("pomb", "poɭɭuget̪", "pɔmᵬ", "tʃuːduɭːa", "betʃa", "bisi")),
    (134, "Cold things", 34, 28, "middle", 4, ("tʃeɭi", "t̪aŋikːin", "t̪ːaŋikin", "t̪aŋut̪ːa", "tʃiːt̪ːa", "t̪aniŋtʃa")),
    (135, "Right", 34, 28, "middle", 5, ("bɑlet̪", "balʌt̪", "bɛlat", "valat̪ːə", "balit̪ːe", "balat̪i")),
    (136, "Left", 34, 28, "middle", 6, ("edʌt̪", "edat̪i", "eɖat̪ːᵘ", "iɖat̪ə", "edat̪ː", "əɖat̪i")),
    (137, "Near", 34, 28, "middle", 7, ("tʃikːʌl", "adut̪", "tʃːikːali", "adut̪ː", "git̪ːa", "pakka")),
    (138, "Far", 34, 28, "right", 1, ("t̪uːre", "d̪ure", "d̪ureɨ", "akale", "d̪uːrʌ", "d̪uːra")),
    (139, "Big", 34, 28, "right", 2, ("mallɑt̪u", "perijə", "malːa", "valija", "maɭːoʋ", "balja")),
    (140, "Small", 34, 28, "right", 3, ("kuɲuŋʌt̪", "kuɲːiget̪ᵊ", "kuɲːat̪", "tʃrija", "elːu", "tʃerija")),
    (141, "Heavy", 34, 28, "right", 4, ("kanʌm", "kanapːugit̪", "kænapet̪u", "ɸaramuɭːa", "uːt̪a", "baːra")),
    (142, "Light", 34, 28, "right", 5, ("kanamiɖi", "kanaiːri", "kanəipit̪i", "ɸaramkuraŋːa", "kadime", "t̪uːʋa")),
    (143, "Above", 34, 28, "right", 6, ("mit̪ːɨ", "melɨ", "mit̪ei", "mukaɭil", "mit̪ːⁱ", "koɖi")),
    (144, "Below", 34, 28, "right", 7, ("tʃidət̪", "kiː", "tʃːide", "t̪aɻe", "tʃiret̪ː", "aɖi")),
    (145, "White", 35, 29, "left", 1, ("peɭɭeⁱ", "beɭɭe", "pɜɭɭə", "veɭut̪ːa", "beɭd̪ⁱ", "boɭit̪ːa")),
    (146, "Black", 35, 29, "left", 2, ("kaːri", "kaːri", "kaːrɨ", "karut̪ːa", "kapːu", "karpi")),
    (147, "Red", 35, 29, "left", 3, ("tʃore", "tʃore", "tʃorəⁱ", "tʃuvanːa", "kempːu", "tʃoːɳɖa")),
    (148, "One", 35, 29, "left", 4, ("oɲtʃi", "oɲtʃi", "oɲtʃːɨ", "onːe", "oɲtʃi", "oɳɖi")),
    (149, "Two", 35, 29, "left", 5, ("d̪ad", "d̪adᵘ", "d̪eɖ", "raɳɖ", "raɖᵊⁱ", "d̪aɳɖi")),
    (150, "Three", 35, 29, "left", 6, ("mudʒi", "muːdʒi", "muːdʒɨ", "muːnːə", "mudʒi", "muːɳɖi")),
    (151, "Four", 35, 29, "left", 7, ("nal", "naːl", "nɑl", "nalə", "naɭ", "naːli")),
    (152, "Five", 35, 29, "middle", 1, ("aɲtʃ", "aɲtʃ", "æɲtʃː", "andʒ", "ajin", "aɲtʃi")),
    (153, "Six", 35, 29, "middle", 2, ("aːr", "aːr", "aːrᵘ", "aːrə", "adʒi", "aːrɨ")),
    (154, "Seven", 35, 29, "middle", 3, ("eɕu", "eɕɨ", "eɕɨ", "eɻu", "eːɭᵘ", "əːɭɨ")),
    (155, "Eight", 35, 29, "middle", 4, ("eʈʈ", "eʈʈ", "ɛʈː", "eʈu", "enma", "əʈʈi")),
    (156, "Nine", 35, 29, "middle", 5, ("ompɑt̪u", "ompat̪u", "ompɛt̪", "ombat̪u", "orambʌ", "ojimbad̪i")),
    (157, "Ten", 35, 29, "middle", 6, ("pɑt̪ːɨ", "pat̪ːɨ", "pɑt̪ː", "pat̪ːu", "pat̪ːᵘ", None)),
    (158, "Eleven", 35, 29, "middle", 7, ("pat̪inonnu", "pat̪inonnu", "pat̪inonːᵘ", "pat̪inonːu", "pat̪ondʒu", "pannoɳɖi")),
    (159, "Twelve", 35, 29, "right", 1, ("pɑnd̪raɳd", "pand̪raɳd", "pɑnd̪raɳd", "paɳɖraɳɖu", "pat̪irad", "pannareɳɖi")),
]


def main() -> None:
    rows = []
    for item, gloss, pdf_page, printed_page, column, page_row, forms in ITEMS:
        for site, form in zip(SITES, forms):
            rows.append({
                "Item": str(item), "Gloss": gloss, "Site_Code": site,
                "PDF_Page": str(pdf_page), "Printed_Page": str(printed_page),
                "Column": column, "Page_Row": str(page_row),
                "Manual_Transcription": form or "",
                "Review_Status": "attested" if form is not None else "source_blank",
                "Confidence": "high",
                "Uncertainty": "" if form is not None else "source prints Nill",
                "Reviewer_Method": METHOD, "Reviewed_At": "2026-08-28",
                "Reviewer_Declaration": DECLARATION,
            })
    assert len(rows) == 240
    assert sum(row["Review_Status"] == "attested" for row in rows) == 239
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 1
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
