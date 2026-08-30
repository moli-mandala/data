#!/usr/bin/env python3
"""Emit the OCR-blind, visually hand-keyed Appendix C items 36--70 chunk.

Every literal below was entered while inspecting the 400-dpi rendered scan.
Selected difficult glyphs were rechecked at 900 dpi. OCR did not seed, supply,
or verify a lexical reading.
"""
from __future__ import annotations

import csv
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_036_070_hand_keyed.tsv"
SITES = "KEL DHA DIG AMO MUN AST MAN BHU AML SEG KAN SHA TOR".split()
GLOSSES = {
    36: "rope", 37: "thread", 38: "needle", 39: "cloth", 40: "ring",
    41: "sun", 42: "moon", 43: "sky", 44: "star", 45: "rain",
    46: "water", 47: "river", 48: "cloud", 49: "lightning",
    50: "rainbow", 51: "wind", 52: "stone", 53: "path", 54: "sand",
    55: "fire", 56: "smoke", 57: "ash", 58: "mud", 59: "dust",
    60: "gold", 61: "tree", 62: "leaf", 63: "root", 64: "thorn",
    65: "flower", 66: "fruit", 67: "mango", 68: "banana",
    69: "wheat_(husked)", 70: "millet_(husked)",
}
FORMS = {
    36: ["1 as̪ʌɖa", "1 as̪ʌɖo", "1 as̪ʌɖo", "3 ɖor", "2 humb", "2 humb", "1 usoɖu", "6 humʈa", "4 ras̪, 6 sumʈo", "1 usoɖu, 4 ras̪", "3 ɖuri, 5 ɖawo", "5 ɖawõ, 6 humʈa", "5 ɖawõ"],
    37: ["1 huʈ", "1 huʈ", "1 huʈ", "1 huʈ", "1 huʈ", "1 huʈ", "2 ɖuru", "1 huʈ, 2 ɖuru", "2 ɖuru", "2 ɖuru", "1 s̪uʈ, 2 ɖuru", "2 ɖuru", "1 huʈ"],
    38: ["1 hwi", "1 hwi", "1 hwi", "1 hwi", "1 hwi", "1 hwi", "1 hwi", "1 hwi", "1 s̪wi", "1 s̪wi", "1 s̪wi", "1 hwi", "1 hwi"],
    39: ["1 poʈʌrõ", "1 poʈoɖõ", "1 poʈʌɖõ", "1 poʈʌɖõ", "3 saʔkõ", "3 sako, 1 poʈʌɖʌ", "4 lugaɖʌ", "4 lugaɖo", "5 tʃhinɖʌro", "5 tʃhinɖra", "5 tʃhinɖʌro", "4 lugaɖa", "6 ʈaɖko"],
    40: ["1 munɖi", "1 munɖhi", "1 munɖi", "1 munɖi", "1 munɖi", "1 munɖi", "1 munɖi", "1 munɖi", "1 munɖi", "1 munɖi", "1 munɖi", "1 munɖi", "2 munɖʌɖo"],
    41: ["1 ɖihi", "1 ɖihi", "1 ɖih", "1 ɖihi", "1 dihi", "1 dihi", "1 dih", "1 dih", "2 ɖahaɖu", "2 ɖahaɖu", "2 ɖahaɖu", "1 ɖih", "1 ɖih"],
    42: ["1 tʃaŋɖ", "1 tʃaŋɖ", "1 tʃaŋɖ", "1 t̪s̪aŋɖ", "2 sãɖ", "2 sãɖ", "1 t̪s̪aŋ", "3 nowõ", "1 tʃaŋɖ", "1 t̪s̪aŋɖ", "1 tʃaŋɖ", "1 t̪s̪aŋɖ", "1 tʃaŋɖ"],
    43: ["1 dʒuŋ", "1 dzuŋ", "1 dzuŋ", "1 dzuŋ", "1 dzuŋ", "1 dzuŋ", "1 dʒuŋ", "2 horoŋ", "2 s̪ʌrʌk", "2 s̪orʌŋ", "2 horʌŋ", "3 wadʱwo", "1 dzuŋ"],
    44: ["1 tʃaŋɖuli", "1 tʃaŋɖuli", "2 tʃaŋɖ, 3 tara", "1 tʃaŋɖuli", "3 ʈaru", "3 ʈaru", "3 ʈara", "3 taru", "3 taru", "3 ʈara", "3 ʈaru", "3 tara", "3 ʈara"],
    45: ["1 paɾ", "1 paɾ", "1 paɾ", "1 paɾ", "1 paʔɾ, 2 wohʌrʌʈ", "1 paɾ, 2 wohʌrʌʈ", "2 wohʌrʌʈ", "1 paɳi, 2 worʌhaʈ", "1 paɳi", "1 paɳi", "1 paɳi", "1 paɳi", "1 paɳi"],
    46: ["1 paɾ", "1 paɾ", "1 paɾ", "1 paɾ", "1 paɾ", "1 paɾ", "1 paɳi", "1 paɳi", "1 paɳi", "1 paɳi", "1 paɳi", "1 paɳi", "1 paɳi"],
    47: ["1 khaɖi", "1 khaɖi", "1 khaɖi", "3 ɳʌi", "1 khaɖi", "1 khaɖ", "2 noŋɖi", "2 nõŋɖi", "2 noŋɖi", "2 nʌŋɖ", "2 noŋɖ", "4 khudʌru", "2 njeŋɖ"],
    48: ["1 baɖʌlõ", "1 baɖʌlo", "1 baɖʌlo", "1 waɖijõ", "1 waɖʌlo", "1 waɖʌlo", "1 waɖʌlo", "1 waɖʌlo", "1 waɖʌwo", "1 waɖʌlʌ", "1 waɖʌwo", "1 waɖhwo", "1 waɖeɭ"],
    49: ["1 bidʒʌle", "1 bidzʌle", "2 bidz", "2 bidʒ", "2 widz", "2 widz", "2 bidz", "2 bidz", "1 bidʒʌwe", "1 bidzʌli", "1 bidzʌli", "2 bidʒ", "1 bidzʌle"],
    50: ["1 baŋʌ", "1 baŋɖ", "1 baŋ", "1 baŋɖ", "1 ban", "1 baŋ", "1 baŋ", "1 baŋɖ", "1 baŋ", "1 baŋ", "1 baŋɖ", "1 baŋ", "1 baŋɖ"],
    51: ["1 waro", "1 wargõ", "1 warɣõ", "1 wargõ", "1 waru", "1 waro", "2 wahaɭu", "2 wahaɭu", "2 wahawu", "2 wahʌɭʌ", "2 wahaɭu", "2 wahʌwʱu", "1 waro"],
    52: ["1 ɖogoro", "1 ɖogoɖo", "1 ɖogʌɖo", "1 ɖʌgʌɖõ", "2 ɖoŋ", "2 ɖoŋ", "1 ɖʌgʌɖu", "1 ɖoguɖu", "1 ɖogʌɖu", "1 ɖoguɖu", "1 ɖogʌɖu", "1 ɖoguɖu", "1 ɖegoɖ"],
    53: ["1 waʈ", "1 waʈ", "1 waʈ", "1 pag-waʈ", "1 waʈ, pag-waʈ", "1 waʈ, pag-waʈ", "1 pai-waʈ", "1 waʈ, 1 pai-waʈ", "1 pai-waʈ", "1 waʈ, pai-waʈ", "1 pai-waʈ", "1 pai-waʈ", "1 waʈ"],
    54: ["1 reʈõ", "1 reʈõ", "1 reʈõ", "1 reŋʈõ", "1 reʈo", "1 reʈo", "2 weɭʈi", "1 reʈo", "1, 2 reʈo", "1 reɭʈʌ", "1, 2 reʈo", "1 reoʈi", "1, 2 weɭʈo"],
    55: ["1 ag", "1 ag", "1 ag", "1 ag", "1 ag", "1 ag", "2 atʱi", "2 aktʱi", "2 aktʱo", "2 aktʱi", "2 agtʱʌ", "2 agtʱo", "2 agtʱi"],
    56: ["1, 2 ʈuwaro", "1, 2 ʈuwaro", "1, 2 ʈuwarõ", None, "2 ʈumaŋo", "2 ʈumaŋo", "1, 2 ɖhuwano", "1, 2 ɖhuwano", "1 ɖhuwaɖu", "1 ɖhuwaɖu", "1 ɖhuwaɖu", "1, 2 ɖhuwano", "1, 2 ɖuwaɖo"],
    57: ["1 kha", "1 kha", "1 khaʌ", "2 rʌkhaɖo", "1 khaʔa", "1 kha", "2 rukhʌɖu", "2 rukhuɖu", "2 rukhuɖu", "2 rukhuɖu", "2 rukhuɖu", "2 rukhuɖu", "2 rokhoɖo"],
    58: ["1 dorõ", "1 dorõ", "1 dorõ", "2 garo", "2 garu", "2 garu, 1 doru", "2 garu", "2 garu", "2 garu", "3 kiʈs̪ʌr", "2 garu", "2 garu", "2 garo"],
    59: ["1 uɖʌlõ", "1 uɖʌlo", "1 uɖʌlo", "4 ɖhuwaɖo", "2 ʈulo", "2 ʈulo", "3 rodzudu", "3 roɖzʌɖu", "3 rodzudu", "3 rodzʌɖu", "3 roɖzudu", "3 rodzedu", "3 redzʌlo"],
    60: ["1 hono", "1 hoŋo", "1 hoŋo", "1 hono", "1 huno", "1 hono", "1 hunʌ", "1 huŋo", "1 sonːo", "1 sonʌ", "1 sona", "1 hunʌ", "1 hoŋo"],
    61: ["1 tʃaɖʌwõ", "1 tʃaɖwõ", "1 tʃaɖõ", "3 dʒʱaɖ", "2 saɖ", "2 saɖ", "3 dzaɖ", "3 dʒaɖ", "3 dzʱaɖ", "3 dzhaɖ", "3 dzhaɖ", "3 dzaɖ", "3 dzaɖ"],
    62: ["1 paŋ", "2 paŋʈhe", "2 paŋʈho", "1 paŋ", "1 paŋ", "1 paŋ", "1 paŋ", "1, 2 paŋe", "2 paŋʈo", "2 paŋʈʌ", "2 paŋʈo", "1, 2 paŋe", "1, 2 palo"],
    63: ["1 muɭõ", "1 muɭo", "1 muɭo", "2 mwi-aŋɖʌ", "1 mul", "1 mul", "1 muɭːʌ", "1 muɭ", "1 moː", "1 muɭe", "1 muɭ", "2 mu-aŋɖu", "1 muɭ"],
    64: ["1 kaʈo", "1 kaʈo", "1 kaʈo", "1 kaʈo", "1 kaʈu", "1 kaʈu", "1 kaʈa", "1 kaʈu", "1 kaʈu", "1 kaʈa", "1 kaʈa", "1 kaʈu", "1 kaʈo"],
    65: ["1 phul", "1 ɸul", "1 ɸulo", "1 ɸuɭ", "1 ɸul", "1 ɸun", "1 ɸul", "1 phuɭ", "1 phuɭ", "1 ɸul", "1 ɸul", "1 ɸuɭ", "1 ɸuɭ"],
    66: ["1 phʌlwõ", "1 ɸʌlwo", "1 ɸʌlwõ", "2 ɸʌɭ", "1, 2 ɸolʌ", "2 ɸol", "2 phʌl", "2 phʌl", "2 ɸʌl", "2 ɸʌɭ", "2 ɸʌɭ", "2 ɸoʌ", "2 ɸol"],
    67: ["1 ambo", "1 ambo", "1 ambo", "1 ambo", "1 ambo", "1 ambo", "1 ambʌ", "1 ambʌ", "1 ambʌ", "1 ambʌ", "1 ambʌ", "1 amba", "1 ambo"],
    68: ["1 keɭo", "1 keɭõ", "1 keɭõ", "1 kejõ", "1 keɭo", "1 keɭo", "1 keɭːʌ", "1 keɭːa", "1 kewa", "1 keɭʌ", "1 keɭa", "1 kewo", "1 keɭ"],
    69: ["1 gʌũ", "1 gʌũ", "1 gʌũ", "2, 3 gowʌ", "2, 3 gõwʌ", "1, 2 gʌõ", "1 gʌũ", "1 gʌũ", "1 ghʌhũ", "1 ghauhõ", "1 ghʌo", "3 gõwe", "1, 2 gʌõ"],
    70: ["1 dʒuwa", "1 dʒuwa", "1 dʒuwa", "1 dʒuwa", "1 dzuwar", "1 dzuwar", "1 dʒuwar", "1 dʒuwar", "1 dʒuwar", "1 dʒuwar", "1 dʒuwar", "1 dʒuwar", "1 dzuwar"],
}
FIELDS = [
    "Item", "Gloss", "Site_Code", "PDF_Page", "Printed_Page", "Column",
    "Manual_Transcription", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = "manual-source-image; rendered-400dpi; OCR-not-accepted"


def page_for(item: int) -> int:
    if item <= 45:
        return 91 + (item - 1) // 5
    if item <= 49:
        return 100
    return 101 + (item - 50) // 5


def main() -> None:
    rows = []
    for item in range(36, 71):
        assert len(FORMS[item]) == 13
        page = page_for(item)
        for index, (site, form) in enumerate(zip(SITES, FORMS[item])):
            status = "blank" if form is None else "attested"
            uncertainty = ""
            if item == 56 and site == "AMO":
                uncertainty = "Source prints a horizontal dash rule instead of a lexical response; confirmed source blank."
            method = METHOD
            if (item, site) in {(47, "TOR"), (51, "DIG")} or item == 62:
                method = "manual-source-image; rendered-400dpi+900dpi-rereview; OCR-not-accepted"
            row = {
                "Item": str(item), "Gloss": GLOSSES[item], "Site_Code": site,
                "PDF_Page": str(page), "Printed_Page": str(page - 8),
                "Column": "left" if index < 6 else "right",
                "Manual_Transcription": form or "", "Review_Status": status,
                "Confidence": "high", "Uncertainty": uncertainty,
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
