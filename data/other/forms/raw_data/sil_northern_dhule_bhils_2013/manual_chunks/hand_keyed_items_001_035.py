#!/usr/bin/env python3
"""Emit the OCR-blind, visually hand-keyed Appendix C items 1--35 chunk.

The literals below were entered while inspecting the 300-dpi rendered scan.
They were not generated, seeded, or copied from OCR.  Similarity group numbers
are retained exactly as evidence; they are not lexical form content.
"""
from __future__ import annotations

import csv
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_001_035_hand_keyed.tsv"
SITES = "KEL DHA DIG AMO MUN AST MAN BHU AML SEG KAN SHA TOR".split()
GLOSSES = {
    1: "body", 2: "head", 3: "hair", 4: "face", 5: "eye",
    6: "ear", 7: "nose", 8: "mouth", 9: "teeth", 10: "tongue",
    11: "breast", 12: "belly", 13: "arm", 14: "elbow", 15: "palm",
    16: "finger", 17: "nail", 18: "leg", 19: "skin", 20: "bone",
    21: "heart", 22: "blood", 23: "urine", 24: "feces", 25: "village",
    26: "house", 27: "roof", 28: "door", 29: "firewood", 30: "broom",
    31: "mortar", 32: "pestle", 33: "hammer", 34: "knife", 35: "axe",
}
FORMS = {
    1: ["1 ɖil", "1 ɖil", "1 ɖil", "1 ɖil", "1 ɖil", "1 ɖil", "1 ɖil", "1 ɖiɭ", "1 ɖiɭ", "2 ɖhʌɖ", "1 ɖil", "1 ɖiɭ", "1 ɖiɭ"],
    2: ["1 muŋkõ", "1 muŋko", "1 munɖʌko", "1 muŋko", "1, 2 munɖkʌ", "1, 2 munɖkʌ", "2 munɖ", "2 moɳɖ", "1, 2 munɖkʌ", "1, 2 munɖkʌ", "1, 2 munɖkʌ", "1, 2 munɖkʌ", "3 maʈha"],
    3: ["1 tʃotiẽ", "1 tʃotia", "1 tʃote", "3 baɭ", "2 sindʒe", "2 sindʒe", "1 dʒuta", "1 dʒuta", "1 dʒhʌʈa", "4 kesː", "1 dʒhʌʈa", "1 dʒuta", "4 keh"],
    4: ["1 mwɪ", "1 mwi", "1 mwɪ", "1 mwi", "2 sob", "2 sob", "1 mui", "1 moi", "1 mwi", "3 mukh", "1 mwi", "1 mwi", "1 mwi"],
    5: ["1 ɖo", "1 ɖo", "1 ɖo", "2, 3 ɖolo", "2 ɖou", "2 ɖou", "3 ɖula", "3 ɖuɭa", "3 ɖuwa", "2, 3 ɖuɭu", "2, 3 ɖuɭu", "3 ɖuwa", "2, 3 ɖolo"],
    6: ["1 kaŋ", "1 kaŋ", "1 kan", "1 kaŋ", "1 kaŋ", "1 kaŋ", "1 kaŋ", "1 kaŋ", "2 kanto", "1 kaŋ", "1 kaŋ", "1 kaŋ", "1 kaŋ"],
    7: ["1 nak", "1 nak", "1 nak", "1 nakh", "1 nakh", "1 nakh", "1 nakh", "1 nakh", "1 nakh", "1 nakh", "1 nakh", "1 nakh", "1 nakh"],
    8: ["1 mwɪ", "1 mwi", "1 mwi", "1 mwi", "2 sob", "2 sob", "1 mui", "1 moi", "1 mui", "1 mui", "1 mui", "1 mui", "1 mui"],
    9: ["1 ɖaʈh", "1 ɖaʈh", "1 ɖaʈ", "1 ɖaʈ", "1 ɖaʈh", "1 ɖaʈh", "1 ɖaʈh", "1 ɖaʈ", "1 ɖaʈ", "1 ɖaʈh", "1 ɖaʈ", "1 ɖaʈ", "1 ɖaʈ"],
    10: ["1 dʒibh̃", "1 dʒib", "1 dʒib", "1 dʒib", "1 dʒib", "1 dʒib", "1 dʒib", "1 dʒip", "1 dʒip", "1 dʒip", "1 dʒib", "1 dʒib", "1 dʒib"],
    11: [None, "1 budʒʌlo", "1 budʒʌlo", None, "1 budʒi", "1 budʒi", "2 ɖai", None, "2 ɖh̃ai", "2 ɖh̃ai", "2 ɖh̃ai", "2 ɖai", None],
    12: ["1 ɖeɖ", "1 ɖheɖ", "1 ɖeɖ", "3 pet̪", "2 pot̪u", "2 pot̪u", "3 pet̪", "3 pet̪", "3 pet̪", "3 pet̪", "3 pet̪", "3 pet̪", "4 potʌɭiu"],
    13: ["1 at̪h", "1 at̪h", "1 at̪h", "1 at̪", "1 at̪h", "1 at̪h", "1 at̪", "1 at̪h", "1 hat̪", "1 hat̪", "1 hat̪", "1 at̪", "1 at̪h"],
    14: ["1 kʌpo", "2 guɖʌgo", "2 guɖʌgo, 3 kuni", "3 khuno", "3 khumi", "3 khumi", "3 khumi", "3 khom", "3 kohoni", "3 kuhuŋi", "3 kuhuŋi", "3 khum", "3 khuŋi"],
    15: ["1 t̪hʌt̪o", "1 t̪hat̪o", "1 t̪haʌt", "2 t̪ʌi-at̪", "3 t̪holt̪u", "2, 3 t̪holʌt̪", "3 t̪haɭi", "3 t̪holt̪i", "4 hetewi", "4 heteɭi", "4 heteɭi", "3 t̪hout̪i", "5 t̪eɭsõ"],
    16: ["1 akʌɖio", "1 akʌɖi", "1 akʌɖi", "3 aŋɭhi", "2 aŋgu", "2 aŋgu", "2 aŋgul", "2 aŋgul", "2 aŋgul", "2 aŋgul", "2 aŋguɭ", "3 aŋt̪hi", "2 aŋguɭ"],
    17: ["1 nokh", "1 nʌkh", "1 nokh", "1 nʌkhõ", "1 nokh", "1 nokh", "1 nʌkh", "1 nokh", "2 nokhʌɖo", "1 nʌkh", "2 nʌkhʌɖio", "1 nokh", "1 nokh"],
    18: ["1 pag", "1 pag", "1 pag", "1 pag", "2 guɖu", "2 guɖu", "3 pai", "3 pai", "3 pai", "3 pai", "3 pai", "3 pai", "3 pai"],
    19: ["1 tʃambaɖo", "1 tʃambʌɖo", "1 tʃambʌɖi", "1 tsambaɖo", "1 samoɖo", "1 sambʌɖʌ", "1 tsambaɖi", "1 tsambaɖo", "1 tsambaɖo", "1 tsambaɖʌ", "1 tʃambaɖʌ", "1 tsambaɖo", "1 tsambaɖo"],
    20: ["1 atko", "1 atko", "1 athakõ", "1 aɖkõ", "2 aɖ", "1 atko", "2 aɖe", "2 aɖ", "1 haɖko", "1 haɖkʌ", "1 haɖkʌ", "1 atkʌ", "2 aɖ"],
    21: ["1 dʒiu", "2 ɪja", "1 dʒib", "3 phopsõ", None, None, None, None, None, "1 dʒiu", None, None, None],
    22: ["1 rʌgʌt̪", "1 rogʌt̪", "1 rogʌt̪", "1 rʌgʌt̪", "1 rokto", "1 rogʌt̪", "3 ɭui", "3 ɭui", "3 ɭui", "3 ɭui", "3 ɭui", "3 ɭui", "3 ɭoi"],
    23: ["1 mut̪", "2 mut̪hai", "1 mut̪", "1 mut̪", "1, 2 mut̪e", "1, 2 mut̪h", "1 mut̪", "1, 2 mut̪h", "1, 2 mut̪h", "1, 2 mut̪h", "1 mut̪", "1 mut̪", "1 mut̪"],
    24: ["1 ogiõ", "1 ogio", "1 ogiõ", "1 agla", "1 ogio", "1 ogit", "1 ogiʌ", "1 aglo", "2, 3 hagdʒo", "3 hadʒʌ", "1, 2 haglo", "1 aglo", "1 agio"],
    25: ["1 gʌõ", "1 gau", "1 gaõ", "1 gau", "1 gau", "1 gau", "1 gaõ", "1 gau", "1 gaõ", "1 gaõ", "1 gau", "1 gau", "1 gaõ"],
    26: ["1 poŋgo", "1 poŋgo", "1 poŋgo", "5 go", "2 koʔo", "3 ko", "4 gʌr", "4 ghor", "4 ghʌr", "4 ghʌr", "4 ghʌr", "4 ghʌr", "6 gjur"],
    27: ["1 pʌhʌ, 2 hiɖu", "2 hiɖu", "1 pʌhʌ, 2 hiʌɭu", "2 hiɖo", "1 paha", None, None, None, "3 pʌɖawo", "3 pʌɖʌlʌ", None, None, "4 set̪"],
    28: ["1 bãɳo", "1 bãɳo", "1 baʌɭo", "1 baiɭo", "1 baʔaŋo", "2 ba", "3 dʒupu", "3 dʒupu", "3 dʒhoplʌ", "3 dʒhoplu", "3 dʒhopʌlu", "3 dʒupu", "1 baiɳo"],
    29: ["2 silpe, 1 lakaɖo", "2 silpe", "2 tsilpe", "3 bait̪ʌŋ", "1 lakʌɖu", "1 lakʌɖʌ", "1 lakaɖa", "1 lakʌɖa", "1 lakaɖo", "1 lakʌɖa", "1 lakʌɖa", "1 lakaɖa", "1 lakoɖ"],
    30: ["1 baɖːi", "1 baɖi", "1 baʌɖi", "1 baiɖi", "1 baʔaɖi", "1 bʌʌɖi", "1 bari", "1 bari", "3 bhahari", "3 bhahari", "3 bhahari", "1 bari", "1 baiɖi"],
    31: ["1 ukhõ", "1 ukhõ", "1 ukhõ", "1 ukhiõ", "2 khaŋʌɖo?", "2 khaɖno", "2 khaŋɖʌɳi", "2 khaɖɭiŋiu", "2 khaŋɖʌɳiu", "2 khaŋɖʌɳiu", "2 khaŋɖʌɳio", "1 ukhõ", "3 sapar"],
    32: ["1 mʌhʌlo", "1 mʌhlo", "1 muhʌlo", "1 muhʌjõ", "2 ɖoŋ", "3 ukʌl", "4 guŋɖu", "4 guŋɖu", "5 ɭuɖu", "6 ɭoguɖu", "1 musʌlʌ", "1 muhʌwʌ", "4 goŋɖo"],
    33: ["1 at̪oɖi", "1 at̪uɖi", "1 at̪huɖi", "1 at̪hoɖi", "1 at̪huɖi", "1 at̪huɖi", "1 at̪huɖu", "1 at̪hoɖi", "1 hat̪hoɖi", "1 hat̪oɖi", "1 hat̪oɖi", "1 hat̪oɖi", "1 at̪hoɖa"],
    34: ["1 s̪uri, 2 ɖahaɭo", "2 ɖahaɭo", "1 s̪uri, 2 ɖahaɭo", "1 s̪uri", "1 s̪uru", "1 s̪uru", "1 s̪uri", "1 s̪ur", "3 t̪s̪oku", "3 t̪s̪oku", "3 t̪s̪oku", "3 t̪s̪oku", "3 tʃaku"],
    35: ["1 kuwaɖo", "1 kuwaɖõ", "1 kuwaɖo", "1 kuraɖ", "1 kuwaɖe", "1 kuwaɖo", "1 kuraɖ", "1 kuraɖ", "1 kuraɖ", "1 kuraɖ", "1 kuraɖ", "1 kuraɖ", "1 kuraɖ"],
}
FIELDS = [
    "Item", "Gloss", "Site_Code", "PDF_Page", "Printed_Page", "Column",
    "Manual_Transcription", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = "manual-source-image; rendered-300dpi; OCR-not-accepted"


def main() -> None:
    rows = []
    for item in range(1, 36):
        assert len(FORMS[item]) == 13
        page = 91 + (item - 1) // 5
        for index, (site, form) in enumerate(zip(SITES, FORMS[item])):
            status = "blank" if form is None else "attested"
            uncertainty = ""
            confidence = "high"
            if item == 10 and site == "KEL":
                status = "ambiguous"
                confidence = "low"
                uncertainty = "Independent 900-dpi re-review confirms a final h-like character with superscript tilde, but its exact intended scope/value remains unclear; retained as visually read."
            if item == 31 and site == "MUN":
                status = "ambiguous"
                confidence = "medium"
                uncertainty = "Independent 900-dpi re-review confirms the source-printed question mark; punctuation and the visually clear response are retained."
            method = METHOD
            if (item, site) in {(10, "KEL"), (31, "MUN")}:
                method = "manual-source-image; rendered-300dpi+900dpi-rereview; OCR-not-accepted"
            row = {
                "Item": str(item), "Gloss": GLOSSES[item], "Site_Code": site,
                "PDF_Page": str(page), "Printed_Page": str(page - 8),
                "Column": "left" if index < 6 else "right",
                "Manual_Transcription": form or "", "Review_Status": status,
                "Confidence": confidence, "Uncertainty": uncertainty,
                "Reviewer_Method": method, "Reviewed_At": "2026-08-28",
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
