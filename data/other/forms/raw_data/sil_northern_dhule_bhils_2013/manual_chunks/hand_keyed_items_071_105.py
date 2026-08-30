#!/usr/bin/env python3
"""Emit the OCR-blind, visually hand-keyed Appendix C items 71--105 chunk.

Every literal below was entered while inspecting the 400-dpi rendered scan.
Selected difficult glyphs were independently rechecked at 900 dpi. OCR did
not seed, supply, or verify a lexical reading.
"""
from __future__ import annotations

import csv
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_071_105_hand_keyed.tsv"
SITES = "KEL DHA DIG AMO MUN AST MAN BHU AML SEG KAN SHA TOR".split()
GLOSSES = {
    71: "rice_(husked)", 72: "potato", 73: "eggplant", 74: "groundnut",
    75: "chili", 76: "turmeric", 77: "garlic", 78: "onion",
    79: "cauliflower", 80: "tomato", 81: "cabbage", 82: "oil",
    83: "salt", 84: "meat", 85: "fat", 86: "fish", 87: "chicken",
    88: "egg", 89: "cow", 90: "buffalo", 91: "milk", 92: "horns",
    93: "tail", 94: "goat", 95: "dog", 96: "snake", 97: "monkey",
    98: "mosquito", 99: "ant", 100: "spider", 101: "name",
    102: "man", 103: "woman", 104: "child", 105: "father",
}
FORMS = {
    71: ["1 t̪s̪okha", "1 tʃokha", "1 tʃokha", "1 t̪s̪okha", "2 morio", "2 moria", "2 muria", "2 muria", "1 t̪s̪ukha", "3 uɖʌri", "3 uɖʌri", "2 murʌ", "1 tʃoha"],
    72: ["1 bʌt̪at̪o", "1 bʌt̪at̪õ", "1 bʌt̪at̪o", "1 bʌt̪at̪ʌ", "1 bʌt̪at̪o", "1 bʌt̪at̪o", "1 bʌt̪ala", "1 bʌt̪at̪a", "2 ala", "2 alːa", "1 bʌt̪at̪a", "1 bʌt̪at̪a", "2 alu"],
    73: ["1 beŋgõ", "1 beŋge", "1 weŋgo", "1 beŋgʌ", "2 riŋʌɳo", "1 weŋge", "2 reŋgʌɳa", "2 riŋgʌɳa", "2 riŋʌɳa", "2 riŋʌɳʌ", "2 riŋʌɳa", "2 riŋʌɳa", "2 riŋgʌɳʌ"],
    74: ["1, 5 hiŋgio", "1, 5 heŋgia", "5 heŋgu", "5 hiŋgi", "2 muŋe", "2 mũge", "1, 2 mũŋiu", "1, 2 mũŋiu", "3 bhuŋdʒa", "3 bhumdzja", "3 bhuŋdʒa", "1 heŋgo", "4 hui-muŋgjan-(laŋa"],
    75: ["1 mirtsu", "1 mirtʃu", "1 mirtʃu", "1 mirtʃe", "1 mirʃu", "1 mirtse", "1 mirtsa", "1 mirtsa", "2 miri", "2 miri", "2 miri", "1 mirtsa", "1 mirtʃõ"],
    76: ["1 eɖo", "1 iɖo", "1 iʌɖo", "1 ʌiɖo", "1 elɖo", "1 elɖʌ", "2 oliɖ", "2 oliɖ", "2 houɖi", "2 hʌliɖ", "2 hʌliɖ", "2 oliɖ", "2 eliɖ"],
    77: ["1 goɳɖʌlõ", "1 goɳɖʌlõ", "1 goɳɖʌlõ", "2 lohõ", "2 lohoɳo", "2 nohoɳo", "2 lʌhon", "2 lohoŋ", "2 losoɳ", "2 loʂum", "2 lohoɳ", "2 lohoɳ", "2 leheɳ"],
    78: ["1 kaɳɖo", "1 kaɳɖo", "1 kaɳɖo", "1 kaɳɖo", "1 kaɳɖu", "1 kaɳɖo", "2 ɖuŋgʌli", "2 ɖuŋgʌli", "2 ɖoŋʌwi", "1 kaɳɖa, 2 ɖoŋgʌli", "2 ɖoŋgʌli", "1 kaɳɖu", "1 kaɳɖa"],
    79: ["1 ɸul-kobi", "1 ɸul-kobi", "1 ɸul-kobi", "1 ɸul-kobi", "1 gobi", "1 gobi", "1 kobi", "1 kobi", "1 gupi", "1 gupi", "1 ɸul-gobi", "1 ɸul-kobi", "1 gobi"],
    80: ["1 tʌmata", "1 tʌmate", "1 tʌmatõ", "1 tʌmatʌ", "2 duʔule", "2 duʔune", "1 tʌmata", "1 tomato", "1 tamʌtirio", "1 tʌmatar", "1 tʌmaterio", "1 tʌmato", "3 ira-riŋʌɳo"],
    81: ["1 paŋ-kobi", "1 paŋ-kobi", "1 paŋ-kobi", "1 paŋ-kobi", "1 gobi", "1 gobi", "1 kobi", "1 kobi", "1 gupi", "1 gupi", "1 paŋ-gubi", "1 gʌɖa-kobi", "1 pʌʈ-gobi"],
    82: ["1 t̪el", "1 t̪el", "1 t̪el", "1 t̪el", "1 t̪el", "1 t̪en", "1 t̪el", "1 t̪el", "1 t̪el", "1 t̪el", "1 t̪el", "1 t̪el", "1 t̪el"],
    83: ["1 kharõ", "1 kharo", "1 kharõ", "1 kharo", "1 kharo", "1 kharo", "1 khaɖo", "1 kharo", "3 noɳ", "3 nʌɳ", "1 kharo", "1 kharo", "2 mith"],
    84: ["1 mʌha", "1 mʌha", "1 mah", "2 bhʌdʒi", "1 mʌha", "1 maha", "1 mah", "1 mah", "1 mas", "1 maʂ", "1 mah", "1 mah", "1 mah"],
    85: ["1 t̪s̪orbi, 2 t̪aɖzo", "2 t̪aɖzõ", "1 tʃorbi", "2 t̪aɖzõ", "3 doɖo", "3 doɖo", "3 ɖʌɖʌ", "3 ɖoɖo", "1 tsʌrbo, 3 ɖoɖo", "1 tʃʌrbi, 3 ɖʌɖʌ", "1 tsʌrbi, 3 ɖʌɖʌ", "1 tsʌrbi", "1 tʃerbi"],
    86: ["1 maʂo", "1 maʂõ", "1 maʂe", "1 maʂõ", "1 maʂu", "1 maʂu", "1 maʂo", "1 mat̪s̪ho", "1 mat̪s̪ha", "1 mat̪s̪hʌ", "1 mat̪s̪ha", "1 maʂu", "1 maʂo"],
    87: ["1 kukʌɖi", "1 kukʌɖi", "1 kukʌɖi", "1 kukʌɖi", "1 kukʌɖi", "1 kukʌɖi", "1 kukʌɖa", "1 kukʌɖa", "1 kukʌɖo", "1 kukʌɖʌ", "1 kukʌɖi", "1 kukʌɖi", "1 kukʌɖo"],
    88: ["1 hakovõ", "1 hakʌvõ", "1 hakʌvõ", "1 hakʌwõ", "2 iɳɖo", "2 iɳɖo", "2 iɳɖo", "2 iɳɖo", "2 aiɳɖo", "2 aɳɖo", "2 aɳɖo", "2 iɳɖo", "1 hakõ"],
    89: ["1 gauɖi", "1 gauɖi", "1 gauɖi", "1 gauɖi", "1 gauɖi", "1 gauɖi", "1 gauɖi", "2 gai", "2 gai", "2 gai", "2 gai", "1 gauɖi", "2 gai"],
    90: ["1 mohoɖi", "1 mohoɖi", "1 mohʌɖi", "3 ɖobʌr", "2 paɖi", "2 paɖi", "2 paɖi", "2 paɖi", "3 ɖuba", "3 ɖubi", "3 ɖubo", "2 paɖi", "3 ɖobo"],
    91: ["1 ɖuɖ", "1 ɖuɖ", "1 ɖuɖ", "1 ɖuɖ", "1 duɖ", "1 duɖ", "1 ɖuɖɦ", "1 ɖuɖ", "1 ɖuɖ", "1 ɖuɖɦ", "1 ɖuɖ", "1 ɖuɖ", "1 ɖuɖ"],
    92: ["1 hiŋtõ", "1 hiŋʌtõ", "1 hiŋʌto", "1 hiŋgʌɖõ", "2 hiŋg", "2 hiŋ", "2 hiŋg", "2 hiŋg", "1 ʃiŋgʌɖo", "1 ʃiŋgʌɖʌ", "1 hiŋgʌɖo", "2 hiŋg", "2 hiŋg"],
    93: ["1 seŋti", "1 seŋtõ", "1 seŋti", "1 seŋto", "1 seŋto", "1 seŋti", "1 semti", "1 semto", "1 tʃhemto", "1 tʃhemti", "1 tʃhemto", "1 semto", "1 semto"],
    94: ["1 bokʌri", "1 bokʌɖi", "1 bokʌɖi", "1 bokʌɖi", "1 bokʌɖi", "1 bokʌɖi", "1 bukʌɖi", "1 bukʌɖo", "1 bukʌɖo", "1 bukʌɖo", "1 bukʌɖi", "1 bukuɖo", "1 bokʌɖ"],
    95: ["2 huɳo, 1 kuʈʌrõ", "2 huɳo", "2 huɳo, 1 kuʈro", "3 tʃitõ", "2 huɳi", "2 huɳi", "1 kuʈri", "1 kuʈʌro", "1 kuʈʌro", "1 kuʈʌrʌ", "1 kuʈʌro", "1 kuʈʌro", "3 tʃiʈõ"],
    96: ["1 hapʌɖõ", "1 hapʌɖo", "1 hapʌɖo", "1 hapʌɖõ", "4 hap", "4 hap", "4 hap, 2 goɖhu", "4 hap, 2 goɖhu", "2 goɖsu", "2 ghʌɖsʌ", "2 goɖʌhu", "4 hap, 2 goɖʌhu", "3 geɖe"],
    97: ["1 bʌɳɖorõ", "1 bʌɳɖro", "1 bʌɳɖro", "1 wʌɳɖʌrõ", "2 makoɖ", "2 makoɖ", "2 makoɖ", "2 makoɖ", "2 makʌɖiu", "2 makʌɖiu", "2 makʌɖiu", "2 makʌɖio", "2 makoɖ"],
    98: ["1 mogʌhẽ", "1 moghẽ", "1 moghe", "2 ɖaha", "1 moghe, 2 ɖahẽ", "1 moghe, 2 ɖahẽ", "2 ɖaha, 3 t̪s̪at̪s̪ʌɖio", "2 ɖah, 3 t̪s̪at̪s̪hʌɖio", "2 ɖaʃa, 3 t̪s̪at̪s̪ʌɖio", "2 ɖaʃu, 3 t̪s̪at̪s̪iɖiʌ", "3 tʃatʃʌɖio", "2 ɖas, 3 t̪s̪at̪s̪ʌɖiu", "3 tʃatʃʌɖio"],
    99: ["1 kiɖo", "1 kiɖõ", "1 kiɖõ", "1 kiɖi", "1 kiɖawi", "1 kiɖo", "1 kiɖawi", "1 kiɖawi", "1 kiɖawi", "1 kiɖawi", "1 kiɖawo", "1 kiɖawo", "1 kiɖawõ"],
    100: ["1 huʈʌɖo", "1 huʈʌɖo", "1 huʈʌɖo", "1 huʈʌɖo", "2 botkil", "2 botkil", "3 hʌʈkuli", "3 hoʈkuli", "3 huʈkuwio", "3 suʈkuli", "3 huʈkuwio", "3 hʌʈkuwio", "1 huʈʌɖo"],
    101: ["1 ɳau", "1 ɳau", "1 ɳaβ", "1 ɳau", "1 nau", "1 nau", "1 nau", "1 nau", "1 ɳau", "1 nau", "1 ɳau", "1 nau", "1 nau"],
    102: ["1 mahʌ", "1 mahõ", "1 mahu", "1 mahõ", "1 mahu", "2 mati", "2 mati", "2 mati", "1 maɳhõ", "3 oɖmi", "1 maɳu", "2 mati", "3 ʌɖmi"],
    103: ["1 bai-mahu", "1 bai-maho", "1 bai-mahu", "5 t̪he", "2 bojõ", "2 bojõ", "3, 4 baihro", "3, 4 bairo", "3, 4 bairo", "3 baiʌr", "3 baiʌr", "3, 4 bairo", "4 baiku"],
    104: ["1 poiro", "1 poiro", "1 pojʌro", "1 poirõ", "1 porijo", "1 pori", "1 puriʌ", "2 ʂuro", "1 purio, 2 t̪s̪huro", "1 puriʌ", "1 purio", "1 purio", "2 ʂoro"],
    105: ["1 bahako", "1 bɦako", "1 bahʌko", "2, 3 babo", "1 bahku, 3 abu", "1 bahku", "2, 3 babu", "2, 3 babu", "2, 3 babu", "4 bas", "2 baba", "2, 3 babu", "5 aboxkhʌ"],
}
FIELDS = [
    "Item", "Gloss", "Site_Code", "PDF_Page", "Printed_Page", "Column",
    "Manual_Transcription", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = "manual-source-image; rendered-400dpi; OCR-not-accepted"


def page_for(item: int) -> int:
    return 101 + (item - 50) // 5


def main() -> None:
    rows = []
    for item in range(71, 106):
        assert len(FORMS[item]) == 13
        page = page_for(item)
        for index, (site, form) in enumerate(zip(SITES, FORMS[item])):
            status = "attested"
            confidence = "high"
            uncertainty = ""
            method = METHOD
            if item == 74 and site == "TOR":
                status = "ambiguous"
                confidence = "low"
                uncertainty = (
                    "The 900-dpi scan confirms the long printed candidate retained here, "
                    "but the final parenthesized segment and its internal consonant cannot "
                    "be resolved confidently; visual candidates: (laŋa / (laɳa."
                )
                method = "manual-source-image; rendered-400dpi+900dpi-rereview; OCR-not-accepted"
            if (item, site) in {
                (87, "TOR"), (91, "MAN"), (91, "SEG"),
                (101, "DIG"), (105, "TOR"),
            }:
                method = "manual-source-image; rendered-400dpi+900dpi-rereview; OCR-not-accepted"
            row = {
                "Item": str(item), "Gloss": GLOSSES[item], "Site_Code": site,
                "PDF_Page": str(page), "Printed_Page": str(page - 8),
                "Column": "left" if index < 6 else "right",
                "Manual_Transcription": form, "Review_Status": status,
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
