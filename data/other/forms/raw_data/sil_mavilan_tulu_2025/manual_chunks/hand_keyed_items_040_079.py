#!/usr/bin/env python3
"""Write the OCR-blind, visually hand-keyed physical-pages-30--31 ledger."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_040_079_hand_keyed.tsv")
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

# Forms were keyed only after direct comparison with 1200-dpi crops of the
# rendered source. None represents a literal printed "Nill", never an OCR gap.
ITEMS = [
    (40, "Moon", 30, 24, "left", 1, ("tʃʌn̪ɖɾʌn", "nilaʋ", "tʃʌn̪ɖɾen", "tʃan̪ɖɾan", "tʃan̪ɖɾʌm", "tʃannuːɾa")),
    (41, "Sky", 30, 24, "left", 2, ("akaːʃʌm", "aːkaːʃʌm", "akaːʃam", "aːkaʃam", "akaʃa", None)),
    (42, "Star", 30, 24, "left", 3, ("nʌkʃʌt̪ɾʌm", "koʈʈi", "nʌkʃʌt̪ɾʌm", "n̪akʂat̪ɾam", "nakʃʌt̪ɾʌm", "miːntʃkki")),
    (43, "Rain", 30, 24, "left", 4, ("maːɾi", "maːɾi", "maːɾi", "maɻa", "bəɾʃʌ", "maɭe")),
    (44, "Water", 30, 24, "left", 5, ("niɾᵘ", "niːɾɨ", "niɾᵊ", "veɭːam", "miːr", "niːɾi")),
    (45, "River", 30, 24, "left", 6, ("tʃal", "tʃaːl", "tʃalᵉ", "puɻa", "tʃoɖ", "poɭe")),
    (46, "Cloud", 30, 24, "left", 7, ("məgʰʌm", "megam", "megʰəm", "megʰam", "megʌm", "moːɖa")),
    (47, "Lightning", 30, 24, "middle", 1, ("iɖimuʈʈuget̪", "idiminːʌl", "iɖimuʈʈu", "minːal", "gəɖle", "maɭəbilli")),
    (48, "Rainbow", 30, 24, "middle", 2, ("maɕvillɨ", "maɕʌvillɨ", "maɕvillᵊ", "maɻavilːi", None, "kamanabille")),
    (49, "Wind", 30, 24, "middle", 3, ("kaʈʈ", "kaːt", "kaʈᵘ", "kaːtʰ", "gaːɭi", "gaːɭi")),
    (50, "Stone", 30, 24, "middle", 4, ("kɑllɨ", "kallɨ", "kɑll", "kalːɨ", "kalːⁱ", "kalli")),
    (51, "Path", 30, 24, "middle", 5, ("t̪eɾu", "t̪eru", "t̪əɾu", "vaɻi", "ʂaːdi", "batte")),
    (52, "Sand", 30, 24, "middle", 6, ("pojːa", "pojːe", "pojːɜ", "maɳal", "pojːe", "maɳa")),
    (53, "Fire", 30, 24, "middle", 7, ("tːu", "t̪uʋ", "tːuʋ", "t̪i", "tʃu", "ʈiʈʈi")),
    (54, "Smoke", 30, 24, "right", 1, ("poge", "poge", "pogeɨ", "puga", "poge", "poge")),
    (55, "Ash", 30, 24, "right", 2, ("vennerⁱ", "beɳːirᵊ", "vennerɜ", "tʃaːɾam", "kadʒaʋ", "buːɖi")),
    (56, "Mud", 30, 24, "right", 3, ("tʃaɭi", "tʃaɭi", "tʃelɨ", "maɳːᵊ", "tʃeɭid", "maɳɳi")),
    (57, "Gold", 30, 24, "right", 4, ("ponn", "ponːɨ", "ponː", "svoɾɳʌm", "baŋgar", "ponni")),
    (58, "Tree", 30, 24, "right", 5, ("maɾʌm", "maɾʌm", "maɾʌm", "maram", "maɾʌm", "mara")),
    (59, "Leaf", 30, 24, "right", 6, ("tʃʌppila", "tʃapːila", "tʃapːile", "ila", "ire", "elekaɳɖa")),
    (60, "Root", 30, 24, "right", 7, ("ver", "beru", "vəru", "veɾi", "beɾⁱ", "beːɾi")),
    (61, "Thorn", 31, 25, "left", 1, ("muɭɭɨ", "muɭɭɨ", "muɭː", "muɭːe", "muɭːⁱ", "muɭɭi")),
    (62, "Flower", 31, 25, "left", 2, ("puː", "puː", "pu", "puː", "pu", "puːʋi")),
    (63, "Fruit", 31, 25, "left", 3, ("parən̪d̪", "paren̪d̪ᵊ", "pærənɖ", "paɻam", "kai", "paɳɳi")),
    (64, "Mango", 31, 25, "left", 4, ("maŋʌ", "maŋa", "maŋɑ", "maŋa", "kukːu", "maːŋge")),
    (65, "Banana", 31, 25, "left", 5, ("parən̪ɖkai", "paren̪d̪akɑi", "paren̪dkaɨ", "vaɻa paɻam", "paraɳɖ", "baːɭe")),
    (66, "Wheat", 31, 25, "left", 6, ("got̪ʌmb", "goːt̪ambɨ", "got̪amb", "got̪amb", "godʒi", "goːɖi")),
    (67, "Millet", 31, 25, "left", 7, (None, None, None, None, None, "naʋaɳe")),
    (68, "Rice (uncooked)", 31, 25, "middle", 1, ("var", "aɾi", "vɐr", "aɾi", "aɾi", "akki")),
    (69, "Potato", 31, 25, "middle", 2, ("keɾʌŋg", "keraŋg", "kɜɾaŋgᶤ", "uɾulakiɻaŋ", "batatːa", "haːləgatte")),
    (70, "Eggplant", 31, 25, "middle", 3, (None, None, None, "vaɻut̪anəŋa", None, "bajane")),
    (71, "Groundnut", 31, 25, "middle", 4, ("nelakʌɖʌla", "nelakadʌla", "nɜlakəɖɑla", "nilakadala", "kadla", "nelakɖale")),
    (72, "Chili", 31, 25, "middle", 5, ("kɑppa pareŋgi", "kapːa paraŋgi", "kapa parɜŋgi", "muɭag", "muɲtʃi", "maɭu")),
    (73, "Turmeric", 31, 25, "middle", 6, ("mɑndʒʌɭ", "maɲdʒʌɭ", "mɑndʒɐɭ", "maɲaɭ", "maŋdʒʌɭ", "mandʒa")),
    (74, "Garlic", 31, 25, "middle", 7, ("veɭɭuɭɭi", "beɭut̪uɭːi", "veɭɭuɭɨ", "veɭut̪uɭːi", "beɭut̪ːuɭːi", "boɭɭuɭɭi")),
    (75, "Onion", 31, 25, "right", 1, ("sɑvoɭʌ", "niːruɭɭi", "niːruɭɭɨ", "uɭːi", "niɾiɭi", "iːruɭɭi")),
    (76, "Cauliflower", 31, 25, "right", 2, (None, None, None, "koɭiflaʋor", None, "puːgoːsi")),
    (77, "Tomato", 31, 25, "right", 3, ("t̪akːɭi", "t̪akːaɭi", "t̪akːaːɭi", "t̪akːaɭi", "t̪akːaːɭi", "tometoː")),
    (78, "Cabbage", 31, 25, "right", 4, ("kjʌbədʒ", "kjabadʒe", "kjabɛdʒ", "kjabədʒɨ", "kjabədʒ", "goːsmuʈʈe")),
    (79, "Oil", 31, 25, "right", 5, ("eɳɳa", "veɭitʃeɳɳa", "veɭɨtʃeːɳːa", "eɳːa", "eɳɳe", "əɳɳe")),
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
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 15
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
