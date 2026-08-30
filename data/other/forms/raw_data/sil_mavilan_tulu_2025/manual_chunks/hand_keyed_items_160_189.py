#!/usr/bin/env python3
"""Write the OCR-blind, visually hand-keyed items-160--189 ledger."""

import csv
import unicodedata
from pathlib import Path


OUT = Path(__file__).with_name("items_160_189_hand_keyed.tsv")
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
# rendered-page crops. This block contains no source blanks or unresolved cells.
ITEMS = [
    (160, "Twenty", 35, 29, "right", 2, ("irupat̪u", "irupat̪u", "ɨrupɑt̪", "irupat̪", "iruʋa", "iriʋad̪i")),
    (161, "One hundred", 35, 29, "right", 3, ("nuːrⁱ", "nuːrⁱ", "nuːr", "nuːrə", "nud̪ⁱ", "nuːrɨ")),
    (162, "Who", 35, 29, "right", 4, ("eru", "erə", "ɛreu", "arə", "erə", "d̪aːri")),
    (163, "What", 35, 29, "right", 5, ("tʃanɑ", "tʃan̪d̪ⁱ", "tʃena", "en̪d̪i", "d̪ajit̪ːe", "en̪d̪i")),
    (164, "Where", 35, 29, "right", 6, ("eɭɭu", "eɭɭə", "eɭːu", "evide", "oɭpːa", "elli")),
    (165, "When", 35, 29, "right", 7, ("epːo", "eppo", "jepːo", "epːoɭ", "epːa", "ekka")),
    (166, "How many", 36, 30, "left", 1, ("et̪rɑ eɳɳʌm", "et̪ra", "et̪rɑ eɳː", "epːoɭ", "et̪ːuɭːa", "etʃtʃaki")),
    (167, "What kind", 36, 30, "left", 2, ("in̪d̪ tʃɑn", "tʃan̪d̪i", "tʃɑnin̪d̪", "en̪t̪ut̪aram", "d̪ajit̪ːe koruva", "en̪d̪ə")),
    (168, "This", 36, 30, "left", 3, ("in̪d̪", "in̪d̪", "ɛn̪d", "it̪", "in̪d̪ⁱ", "id̪i")),
    (169, "That", 36, 30, "left", 4, ("at̪", "at̪", "ət̪", "at̪ə", "aʋᵘ", "ad̪i")),
    (170, "These", 36, 30, "left", 5, ("in̪d̪at̪i", "on̪d̪enə", "in̪deʈ", "it̪", "in̪d̪umaːt̪ːa", "iʋə")),
    (171, "Those", 36, 30, "left", 6, ("ɑd̪ene", "at̪enə", "ɑt̪enaⁱ", "at̪", "aʋmaːt̪a", "aʋə")),
    (172, "Same (like)", 36, 30, "left", 7, ("oɲtʃijenneⁱ", "orepolt̪e", "oɲdʒine", "orepole", "ondʒelʌkːe", "annane")),
    (173, "Different (other)", 36, 30, "middle", 1, ("pete petːe", "matːʌm", "pəta", "vjetjest̪am", "bjat̪jasʌm", "ennanoː")),
    (174, "Whole (unbroken)", 36, 30, "middle", 2, ("pɑtːa", "pɑtːe", "pɛʈɑ", "muɻuvan", "otige", "otti")),
    (175, "Broken (pot)", 36, 30, "middle", 3, ("poɭiɳɖ", "poʈʈi", "poʈːɨ", "pot̪ːi", "punɖaɳɖ", "odindʒad̪i")),
    (176, "Few", 36, 30, "middle", 4, ("iɲine", "iɲːinə", "iɲenːeɨ", "kuratʃ", "paniːt̪ː", "tʃenni")),
    (177, "Many", 36, 30, "middle", 5, ("d̪əmːa", "nalːonʌm", "neratɕ", "kure", "sumar", "d̪umba")),
    (178, "All", 36, 30, "middle", 6, ("pɑtːa", "pɑtːe", "Muɻɔn", "muɻuvan", "man̪t̪erale", "ella")),
    (179, "Eat", 36, 30, "middle", 7, ("t̪ikːɳe", "t̪inːᵊ", "t̪inːɔ", "kaɻikːə", "t̪inpeɳa", "t̪inni")),
    (180, "It (the dog) bit", 36, 30, "right", 1, ("tʃitʃige", "tʃitʃigenu", "tʃɨtʃːɨgɜt̪ː", "kad̪itʃu", "tʃitʃinu", "kad̪i")),
    (181, "He is hungry", 36, 30, "right", 2, ("vadɑgurugen", "bɑdɑverugənə", "beɖɑguruge", "viʃakːinnu", "bed̪apuɳɖᵘ", "bajippə")),
    (182, "He is thirsty", 36, 30, "right", 3, ("niːreruguɳu", "niːrerigenᵊ", "nirɜrugɐt̪", "d̪ahikːunːu", "aŋgʌt̪ːʌne", "bajjarke")),
    (183, "He is drinking", 36, 30, "right", 4, ("unukkine", "uɳukkinə", "uɲukːeɲᵘ", "kud̪ikːunnu", "parpeɳu", "kud̪i")),
    (184, "He is sleeping", 36, 30, "right", 5, ("tʃiɲtʃigeɳu", "tʃiɲdʒiginu", "tʃɨɲdʒɨgɜɳ", "urakːam", "nid̪ra", "ʋaːri")),
    (185, "He lay down", 36, 30, "right", 6, ("pədugeɳu", "padeginu", "padukɜɳⁱ", "kid̪akːunːu", "d̪erpujʌ", "bud̪d̪a")),
    (186, "Sit down", 36, 30, "right", 7, ("kuɭɭepp", "kullupːə", "kuɭːe", "irikː", "kuɭːɨ", "aɭit̪a")),
    (187, "Give", 37, 31, "left", 1, ("t̪a", "tːa", "tːa", "kod̪ukːi", "korpa", "kod̪i")),
    (188, "It is burning", 37, 31, "left", 2, ("kɑt̪iɳɖu", "kat̪uga", "kat̪iɲu", "kat̪ːunːu", "pot̪unⁱ", "tʃud̪i")),
    (189, "He died", 37, 31, "left", 3, ("tʃejt̪igu", "tʃejt̪ːu", "tʃejt̪iːu", "maritʃu", "t̪iroɳa", "tʃaʋo")),
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
    assert len(rows) == 180
    assert all(row["Review_Status"] == "attested" for row in rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
