#!/usr/bin/env python3
"""Emit the OCR-blind, visually hand-keyed Appendix C items 176--210 chunk.

Every response cell, including the six printed horizontal-dash blanks, was
entered while inspecting the rendered scan at 400 dpi and independently
re-read at 900 dpi. OCR and PDF text did not seed, supply, or verify a lexical
reading.
"""
from __future__ import annotations

import csv
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_176_210_hand_keyed.tsv"
SITES = "KEL DHA DIG AMO MUN AST MAN BHU AML SEG KAN SHA TOR".split()
GLOSSES = {
    176: "different", 177: "whole", 178: "broken", 179: "few",
    180: "many", 181: "all", 182: "eat", 183: "bite",
    184: "to-be_hungry", 185: "drink", 186: "be_thirsty",
    187: "sleep", 188: "lie_down", 189: "sit_down", 190: "give",
    191: "burn", 192: "die", 193: "kill", 194: "fly", 195: "walk",
    196: "run", 197: "go", 198: "come", 199: "speak", 200: "hear",
    201: "see", 202: "I", 203: "you_(informal)", 204: "you_(formal)",
    205: "he", 206: "she", 207: "we_(inclusive)",
    208: "we_(exclusive)", 209: "you_(plural)", 210: "they",
}
FORMS = {
    176: ["1 dʒuɖõ", "1 dʒuɖo", "1 dʒuɖõ", "3 alʌgʌlʌg", "1 dzuɖu, 2 alog", "2 alog", "1 dʒuɖo", "2 olʌg", "3 alʌgʌlʌg, 4 phorok", "3 alʌgʌlʌg", "3 olʌgʌlʌg", "3 alʌgʌlʌg", "4 pharek"],
    177: ["1 purõ", "1 puro", "1 puro", "4 buɖkõ", "2 akhuwo", "2 akhuwo", "2 akhlo", "2 akhlo", None, "3 gulʌ", None, None, "2 akhwalo"],
    178: ["1 phuʈʌlõ", "1 ɸuʈʌlo", "1 ɸuʈʌlo", "1 phuʈʈo", "1 puʈʈio", "1 puʈno", "1 phuʈʈo", "1 phuʈlo", "1 phuʈlo", "1 phuʈʈʌ", "1 phuʈʈo", "1 phuʈlu", "1 phuʈel"],
    179: ["1 wai", "1 wai", "1 wai", "3 dzʌraksõ", "2 t̪oɖo", "2 t̪oɖo", "2 t̪huɖo", "2 t̪uɖo", "2 t̪huɖa", "2, 4 t̪huɖʌk", "2, 4 t̪huɖʌk", "2 t̪hulo", "4 t̪horaʂ"],
    180: ["1 dzahakõ", "1 dʒahakõ", "1 dʒahako", "2 haoto", "1 dzaʔakho", "1 dzaʔakho", "2 hoʈo", "2 huɭʈo", "1 dʒahti", "3 gɦʌɳʂʌʈʌ", "2 holʈa", "2 hʌɭʈa", "4 dʒobeɖ"],
    181: ["1 baɖõ", "1 baɖẽ", "1 baɖ:e", "1, 3 hʌlɪχ", "1 baɖe", "2 akhe", "2 akho", "2 akho", "2 akha", "2 akha", "2 akha", "2 akha", "2 akha, 3 boʈha"],
    182: ["1 khaɖʌlo", "1 khaɖõ", "1 khaɖo", "1 khaɖʌlo", "1 khaɖo", "1 khaɖo", "1 khaɖʌlo", "1 khaɖʌlo", "1 khaɖʌlo", "1 khaɖulo", "1 khaɖʌlo", "1 khaɖʌlu", "1 khaije"],
    183: ["1 tsaolo", "1 tʃaolo", "1 tʃawijo", "1 tsaolo", "1 sauwio", "1 saulio", "1 tsaolo", "1 tsaolo", "1 tsaijo", "1 tsailu", "1 tsaula", "1 tsaolu", "1 tsaiel"],
    184: ["1 pukh-lagʌli", "1 pukh-lagli", "1 pukh-lagi", "1 bhukh-lagɦi", "1 pukh-lagi", "1 pukh-lagi", "1 bukhlo", "1 bhukh-lagʌle", "1 bhukulu", "1 bhuklu", "1 bhuk-lagli", "1 bhukh-lagʌle", "1 bukh-lagʌɳo"],
    185: ["1 piɖʌlõ", "1 piɖõ", "1 piɖo", "1 piɖʌlo", "1 piɖo", "1 piɖo", "1 piɖʌlo", "1 piɖʌlo", "1 piɖʌlo", "1 piɖʌlu", "1 piɖʌlo", "1 piɖʌlu", "1 piɭ"],
    186: ["1 pãhpi-lagʌli", "1 pahpi-lagli", "1 phahpi-lagi", "2 t̪ʌhʌro-lagɦi", "2 toroho-lagi", "2 toroho-lagi", "3 ʈihlu", "3 ʈih-lagʌɳe", "3 ʈihulu", "3 ʈiʂʌlu", "3 ʈiʂ-lagli", "3 ʈih-lagʌɳe", "3 ʈih-lagʌɳo"],
    187: ["1 huʈʌlo", "1 huʈʌlo", "1 huʈʌlo", "1 huʈʌlo", "1 huwio", "1 huwio", "1 hulo", "1 hulo", "1 hulo", "1 sulu", "1 sulo", "1 hulu", "1 hujel"],
    188: ["1 uɳɖʌli-poɖʌlo", "1 uɳɖʌli-poɖʌlo", "1 uɳɖʌli-poɖʌlo", "2 huβirʌ", "3 loʈu", "3 noʈio", "3 l:uʈʈo", "3 luʈlo", "3 luʈhʈo", "3 bhima-luʈiu", "3 luʈʈo", "3 luʈlu", "3 loʈel"],
    189: ["1 boʈʈo", "1 boʈhʈo", "1 boʈho", "1 boʈhʈo", "1 boʈhuhu", "1 boʈhio", "1 bohʌlo", "1 boʈhlu", "1 boʈhio", "2 bhim-baʃiu", "1 bohoɳo", "1 bohʌlo", "1 bohel"],
    190: ["1 dedo", "1 dedo", "1 deɖo", "1 dʌɖʌlo", "2 apʈho", "2 apʈho", "2 aplo", "2 aplo", "2 aplo", "2 apli", "2 aplo", "2 aplu", "2 apil"],
    191: ["1 ʂilgi-geɦlo", "1 ʂilgi-geɦlo", "2 boljo", "1 hilgi-gʌilo", "2 holehe", "2 balio", "2 bol:o", "2 balo", "3 bauɳe", "4 ɖɦʌpiju", "2 bol-gʌilo", "2 baulu", "2 bolil"],
    192: ["1 modzo", "1 modʒo", "1 moi-gijo", "1 moi jo", "1 maju", "1 maju", "1 moilu", "1 mal:o", "1 mʌr-gijo", "1 mʌr-giju", "1 mor-gʌijo", "1 mol:u", "1 morjel"],
    193: ["1 maijõ", "1 maijõ", "1 maiʈakjo", "1 maiʈakẽ", "1 mait-ʈakju", "1 mait-ʈekju", "1 mal:o", "1 mol:o", "1 mari", "1 mariu", "2 marnakhjo", "1 mar-del:u", "1 maril"],
    194: ["1 uɖʌlõ", "1 uɖʌlõ", "1 uɖulo", "1 uɖi-gʌijo", "1 uɖio", "1 uɖio", "1 uɖlo", "1 uɖlo", "1 uɖ-gijo", "1 uɖ-gojo", "1 uɖ-gʌijo", "1 uɖlu", "1 uɖel"],
    195: ["1 tsali-geɦlo", "1 tsali-geɦlo", "1 t̪s̪alulo", "1 tsailijo", "1 saliu", "2 gojo", "3 dzaʈʌlo", "2 goilu", "3 dzaʈʌlo", "3 dzaʈriu", "1 tsal:u", "1 tsal:u", "1 tsalel"],
    196: ["1 dɦauɖio", "1 dɦʌuɖu-togijo", "1 dɦʌuɖulo", "1 ɖoɖi-gʌijo", "3 gugdiu", "3 gugdiu", "1 ɖouɖu", "1 ɖoɖo", "1 ɖoɭɳe", "1 dauɖi-gʌɖiu", "1 ɖoweɖu", "1 ɖoɖu", "1 ɖowɖil"],
    197: ["1 gehlo", "1 ghio", "1 gijo", "1 gʌijo", "1 goju", "1 gojo", "2 dzaʈrilo", "1 goilu", "2 dzaɳe", "2 dzaɳe", "1 goju", "2 dzaʈrilu", "1 giljo, 2 dzael"],
    198: ["1 aodʒe", "1 alo", "1 aβulo", "1 ano", "1 aiju", "1 aiju", "1 aulu", "1 aolo", "1 aulu", "1 aiju", "1 aolu", "1 aul:u", "1 aijel"],
    199: ["1 gogʌlo", "1 gogʌlo", "1 gogulo", "2 bolio", "1 gogijo", "1 gogijo", "2 bul:o", "2 bul:u", "2 buliu", "2 buliu", "2 bul:u", "2 bul:u", "2 bolil"],
    200: ["1 uɳʌlo", "1 uɳʌlo", "1 uɳaulo", "2 hombʌilo", "1 uɳaiju", "1 uɳaiju", "2 hombʌl:o", "2 hombʌlo", "2 hombiliu", "2 sʌmiliu", "2 homel:o", "2 hombol:u", "2 hombʌlil"],
    201: ["1 elõ", "1 βelo", "2 herulo", "2 helo", "4 palio", "4 palio", "5 ɖekhʈo", "5 ɖekhlo", "5 ɖekhlo", "5 ɖekhlu", "5 ɖekhʈo", "5 ɖekhlo", "5 ɖekhel"],
    202: ["1 ai", "1 aɪ", "1 aɪ", "3 mʌi", "1 ai", "1 ai", "2 mi", "2 mi", "2 mi", "2 me", "2 mi", "2 mi", "2 mi"],
    203: ["1 t̪ũ", "1 t̪u", "1 t̪ũ", "1 t̪u", "1 t̪u", "1 t̪u", "1 t̪u", "1 t̪ũ", "1 t̪u", "1 t̪u", "1 t̪u", "1 t̪u", "1 t̪u"],
    204: ["1 t̪uma", "1 t̪uma", "1 t̪umã", "1 t̪umu", "1 t̪umu", "1 t̪umo", "1 t̪umu", None, "1 t̪umo", "1 t̪umo", "1 t̪umu", None, "1 t̪umo"],
    205: ["1 ʈo", "1 ʈo", "1 ʈo", "1 ʈo", "1 ʈo", "1 ʈo", "2 tʃu", "2 tʃu", "2 t̪ʂu", "2 t̪ʂu", "2 t̪ʂo", "2 tʃu", "1 ʈo"],
    206: ["1 ʈi", "1 ʈi", "1 ʈi", "1 ʈi", "1 ʈe", "1 ʈe", "2 tʃi", "2 tʃi", "2 tʃi", "2 tʃi", "2 tʃi", "2 tʃi", "1 ʈi"],
    207: ["1 apu", "1 apuhõ", "1 apu", "1 apu", "1 apu", "1 apu", "1 apu", "1 apu", "1 apu", "1 apʌɳo", "1 apu", "1 apu", None],
    208: ["1 amã", "1 amã", "1 amã", "1 amu", "1 amu", "1 amu", "1 amu", "1 amu", "1 amu", "1 amo", "1 amu", "1 amu", "1 amu"],
    209: ["1 t̪umã", "1 t̪umo", "1 t̪umuhõ", "1 t̪umo", "1 t̪umu", "1 t̪umo", "1 t̪umu", "1 t̪umo", "1 t̪umo", "1 t̪umo", "1 t̪umu", "1 t̪umu", "1 tumu"],
    210: ["1 ʈe", "1 ʈe", "1 ʈe", "1 ʈẽ", "1 ʈe", "1 ʈe", "2 tʃja", "2 tʃa", "2 tʃe", "2 tʃe", "2 tʃe", "2 tʃju", "3 ʈa"],
}
FIELDS = [
    "Item", "Gloss", "Site_Code", "PDF_Page", "Printed_Page", "Column",
    "Manual_Transcription", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = "manual-source-image; rendered-400dpi+900dpi-rereview; OCR-not-accepted"


def page_for(item: int) -> int:
    if item <= 179:
        return 126
    if item <= 209:
        return 127 + (item - 180) // 5
    return 133


def main() -> None:
    rows = []
    for item in range(176, 211):
        assert len(FORMS[item]) == 13
        page = page_for(item)
        for index, (site, form) in enumerate(zip(SITES, FORMS[item])):
            blank = form is None
            row = {
                "Item": str(item), "Gloss": GLOSSES[item], "Site_Code": site,
                "PDF_Page": str(page), "Printed_Page": str(page - 8),
                "Column": "left" if index < 6 else "right",
                "Manual_Transcription": form or "",
                "Review_Status": "blank" if blank else "attested",
                "Confidence": "high",
                "Uncertainty": "Confirmed printed horizontal dash rule; no lexical response." if blank else "",
                "Reviewer_Method": METHOD, "Reviewed_At": "2026-08-28",
                "Reviewer_Declaration": DECLARATION,
            }
            rows.append({key: unicodedata.normalize("NFC", value) for key, value in row.items()})
    if len(rows) != 455:
        raise AssertionError("review-chunk topology drift")
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} hand-keyed review rows to {OUTPUT}")


if __name__ == "__main__":
    main()
