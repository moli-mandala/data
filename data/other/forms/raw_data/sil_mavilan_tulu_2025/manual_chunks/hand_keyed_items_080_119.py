#!/usr/bin/env python3
"""Write the OCR-blind, visually hand-keyed items-80--119 ledger."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_080_119_hand_keyed.tsv")
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

# Literal readings entered from 1200-dpi rendered-page crops. This block has
# no source absence markers and no unresolved cells.
ITEMS = [
    (80, "Salt", 31, 25, "right", 6, ("upːɨ", "uppɨ", "uppⁱ", "upːɨ", "upːⁱ", "uppɨ")),
    (81, "Meat", 31, 25, "right", 7, ("kadʒipːɨ", "eratʃi", "kədʒippɨ", "iratʃi", "mas", "jəɾtʃi")),
    (82, "Fat", 32, 26, "left", 1, ("nejːɨ", "nejːɨ", "nejjɨ", "nej", "nejə", "neɳa")),
    (83, "Fish", 32, 26, "left", 2, ("umːi", "miːn", "ummɨ", "miːn", "miːn", "miːni")),
    (84, "Chicken", 32, 26, "left", 3, ("koi", "koi", "koji", "koɻi", "korᵊ", "koːɭi")),
    (85, "Egg", 32, 26, "left", 4, ("muʈʈa", "moʈʈa", "moʈːɑ", "muʈːa", "keʈi", "motte")),
    (86, "Cow", 32, 26, "left", 5, ("kaːli", "pajʋu", "kannukaːlɨ", "paʃu", "peʈːa", "pajju")),
    (87, "Buffalo", 32, 26, "left", 6, ("pot̪", "eruma", "eɾumɐ", "eruma", "goɳa", "poːʈɨ")),
    (88, "Milk", 32, 26, "left", 7, ("palɨ", "paːl", "paːl", "pal", "pere", "paːlɨ")),
    (89, "Horns", 32, 26, "middle", 1, ("kombu", "komb", "komb", "komb", "kombⁱ", "kombi")),
    (90, "Tail", 32, 26, "middle", 2, ("valɨ", "baːl", "bɐːl", "val", "biːɭa", "baːli")),
    (91, "Goat", 32, 26, "middle", 3, ("aːdᵘ", "aːdɨ", "aːdᶤ", "aːɖɨ", "eɖᵊ", "aːɖɨ")),
    (92, "Dog", 32, 26, "middle", 4, ("nɑi", "naji", "naji", "paʈːi", "nai", "naːji")),
    (93, "Snake", 32, 26, "middle", 5, ("paːmb", "pamb", "pamb", "pamb", "par", "paːmbɨ")),
    (94, "Monkey", 32, 26, "middle", 6, ("koɾŋʌn", "koɾaŋ", "koraŋ", "kuɾaŋan", "maŋkej", "koːɖe")),
    (95, "Mosquito", 32, 26, "middle", 7, ("pirəkʰ", "pirukʰe", "pirɐkʰ", "koʈugə", "koʈug", "sollᵉ")),
    (96, "Ant", 32, 26, "right", 1, ("urumbu", "urumbᶤ", "uɾumbuɨ", "urumbɨ", "pidʒiŋᵊ", "irpu")),
    (97, "Spider", 32, 26, "right", 2, ("urupuɭi", "pəruviɭi", "pəɾuviɭi", "tʃiləɳɖi", "baŋʌan pale", "dzoɖe")),
    (98, "Name", 32, 26, "right", 3, ("put̪rᵘ", "peɾə", "pera", "peɾə", "puɖerᵘ", "peɖa")),
    (99, "Man", 32, 26, "right", 4, ("aːi", "aɳɳoʋ", "aɨ", "manuʃjan", "aːɳ", "aːɳaːɭi")),
    (100, "Woman", 32, 26, "right", 5, ("at̪", "peɳɳoʋ", "peɲːe", "st̪riː", "poɳːᵘ", "poɳɳaːɭɨ")),
    (101, "Child", 32, 26, "right", 6, ("kuɲːi", "tʃerumei", "kuɲɲ", "kuɲːᵊ", "baːle", "kuɲi")),
    (102, "Father", 32, 26, "right", 7, ("ajjeⁱ", "ajːan", "ajːen", "atʃan", "amːe", "appo")),
    (103, "Mother", 33, 27, "left", 1, ("apːeⁱ", "apːa", "ɑpːe", "amma", "apːa", "aʋʋo")),
    (104, "Older brother", 33, 27, "left", 2, ("aɳɳei", "aɳɳe", "aɳɳa", "dʒeʈːan", "aɳːe", "aɳɳə")),
    (105, "Younger brother", 33, 27, "left", 3, ("anij", "məgjeⁱ", "anijan", "anijan", "megːe", "t̪ammanə")),
    (106, "Older sister", 33, 27, "left", 4, ("akːʌm", "ɑkːa", "ɑkːɐ", "tʃetʃːi", "akːa", "akkə")),
    (107, "Younger sister", 33, 27, "left", 5, ("anijʌt̪i", "meged̪i", "ɑnijɛt̪ːi", "anudʒat̪ːi", "megit̪ːi", "t̪aŋgə")),
    (108, "Son", 33, 27, "left", 6, ("magei", "magei", "mogaɨ", "makan", "mage", "moːʋo")),
    (109, "Daughter", 33, 27, "left", 7, ("magɑɭ", "magaɭᶤ", "makɐɭ", "makaɭ", "magʌɭ", "moːʋa")),
    (110, "Husband", 33, 27, "middle", 1, ("merei", "merei", "mərə", "ɸart̪aʋ", "kaɳɖʌɳi", "ʋɖija")),
    (111, "Wife", 33, 27, "middle", 2, ("mert̪i", "merət̪i", "merɜd̪ːɨ", "ɸaɾja", "bʰudʌt̪ːi", "poɳɳi")),
    (112, "Boy", 33, 27, "middle", 3, ("dʒikːei", "tʃerumei", "dʒɜkːɨ", "aɳu", "aɳᵊ", "kiɳɳi")),
    (113, "Girl", 33, 27, "middle", 4, ("dʒikːʌɭ", "tʃerumi", "tʃɜrumɩ", "peɳːə", "peɳɳᵊ", "muːɖi")),
    (114, "Day", 33, 27, "middle", 5, ("d̪inʌm", "devasʌm", "dɨnɜm", "divasam", "d̪inə", "piːli")),
    (115, "Night", 33, 27, "middle", 6, ("an̪ɖ̻igᶤ", "an̪d̪i", "an̪ɖ̻ɨ", "ɾat̪ri", "kat̪ːʌlə", "bajit̪i")),
    (116, "Morning", 33, 27, "middle", 7, ("pelʌtʃːe", "pilerᵘ", "pɛlatʃːe", "pakal", "pagelⁱ", "polaːka")),
    (117, "Noon", 33, 27, "right", 1, ("mad̪jenu", "utʃa", "mæd̪jan", "utʃːa", "mad̪ːenᵊ", "madzːaɳu")),
    (118, "Evening", 33, 27, "right", 2, ("pɑjjegu", "san̪d̪ja", "vaikiʈː", "vaikunːeram", "bʰajᵊ", "bajit̪aːppaka")),
    (119, "Yesterday", 33, 27, "right", 3, ("kode", "koɖe", "ked:ɜ", "inːale", "koɖe", "ninnaːn̪d̪i")),
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
                "Reviewed_At": "2026-08-28", "Reviewer_Declaration": DECLARATION,
            })
    assert len(rows) == 240
    assert all(row["Review_Status"] == "attested" for row in rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
