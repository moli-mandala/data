#!/usr/bin/env python3
"""Emit the OCR-blind, visually hand-keyed Appendix C items 106--140 chunk.

Every literal below was entered while inspecting the 400-dpi rendered scan.
Dense kinship/time rows and selected difficult glyphs were rechecked at 900
dpi. OCR did not seed, supply, or verify a lexical reading.
"""
from __future__ import annotations

import csv
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_106_140_hand_keyed.tsv"
SITES = "KEL DHA DIG AMO MUN AST MAN BHU AML SEG KAN SHA TOR".split()
GLOSSES = {
    106: "mother", 107: "older_brother", 108: "younger_brother",
    109: "older_sister", 110: "younger_sister", 111: "son",
    112: "daughter", 113: "husband", 114: "wife", 115: "boy",
    116: "girl", 117: "day", 118: "night", 119: "morning",
    120: "noon", 121: "evening/afternoon", 122: "yesterday",
    123: "today", 124: "tomorrow", 125: "week", 126: "month",
    127: "year", 128: "old", 129: "new", 130: "good", 131: "bad",
    132: "wet", 133: "dry", 134: "long", 135: "short", 136: "hot",
    137: "cold", 138: "right", 139: "left", 140: "near",
}
FORMS = {
    106: ["1 jahaki", "1 jɦaki", "1 jahʌki", "5 ma", "2 aihi", "3 jʌʔhi", "4 ai", "2 alje", "4 ai", "6 aiʃ", "4 ai", "4 ai", "7 ax"],
    107: ["2 ɖaɖa", "2 ɖaɖo", "1 modo-pau, 2 ɖaɖa", "4 motho-bhaha", "1 woɖu-paihi", "1 woɖu-pauhu", "3 motto-baih, 2 ɖaɖa", "3 motlu-bai", "3 motho-bhai", "3 motto-bhaiʃ", "3 motu-bhaiʂ", "3 motlu-bai", "5 ɖawalo-ɖaɖo"],
    108: ["1 haɳo-pau", "1 haɳo-pau", "1 haɳo-pau", "2 aiʈo-bhaha", "1 hanu-paihi", "1 hanu-pauhu", "4 aiʈlo-baih", "4 aiʈʌlo-bai", "3 ɳaɳlu-bhai", "3 ɳaɳɖlu-bhaiʃ", "3 ɳaɳlu-bhaiʂ", "4 aiʈʌlu-bai", "5 aʈʌlio-ɖaɖo"],
    109: ["1, 3 moɖi-bõhi", "1, 3 moɖi-bõhɪ", "1, 3 moɖi-bõhi", "4 mothi-bʌiho", "3 woɖi-boʔhɪ", "3 woɖi-boʔhɪ", "2 bai", "1 motli-boɳi", "1 mothi-boɳi", "1 moti-bhõniʂ", "1 moti-bani", "1 motli-boɳi", "5 ɖawali-bai"],
    110: ["1 hani-bõhi", "1 hani-bõhɪ", "1 hani-bõhi", "4 aiʈi-bʌiho", "1 hani-boʔhɪ", "1 hani-boʔhɪ", "2 aiʈli-boɳihɪ", "2 aiʈʌli-boɳi", "3 ɳaɳli-boɳi", "3 ɳaɳɖli-bhoniʂ", "3 ɳaɳli-bani", "2 aiʈʌli-boɳi", "4 aʈʌli-bai"],
    111: ["1 poiro", "1 poiro", "1 pojʌro", "1 poiro", "1 poriu", "1 porlu", "2 ʂuru", "2 ʂuru", "1 puriu, 2 t̪s̪huro", "1 puriu", "1 purio", "2 ʂuru", "2 ʂoro"],
    112: ["1 poiri", "1 poiri", "1 pojʌri", "1 poiri", "1, 2 pori", "1, 2 pori", "2 ʂuri", "2 ʂuri", "1 purai, 2 t̪s̪huri", "1 purai", "1 purai", "2 ʂuri", "2 ʂori"],
    113: ["1 mati", "1 mati", "1 mati", "1 mati", "1 mati", "1 mati", "1 mati", "1 mati", "2 oɖmi, 3 pahaɳo", "2 oɖmi", "3 pahaɳo", "1 mati", "2 ʂmi"],
    114: ["1 t̪he", "1 t̪he", "1 t̪he", "1 t̪he", "2 bojõ", "2 bojõ", "3 laɖi", "3 laɖi, 4 bairo", "3 laɖi, 4 bairo", "3 laɖi", "3 laɖi", "3 laɖi", "4 baiko"],
    115: ["1 poiro", "1 poiro", "1 poiro", "1 poiro", "1 poriu", "1 poriu", "2 ʂuru", "2 ʂuru", "1 puriu", "1 puriʌ", "1 purio", "1 purio", "2 ʂoro"],
    116: ["1 poiri", "1 poiri", "1 poiri", "1 poiri", "1, 2 pori", "1, 2 pori", "2 ʂuri", "2 ʂuri", "1 purai", "1 purai", "1 purai", "2 ʂuri", "2 ʂori"],
    117: ["1 ɖihi", "1 ɖihi", "1 ɖih", "1 ɖihi", "1 ɖih", "1 ɖihi", "1 ɖih", "1 ɖih", "2 ɖahaɖu", "2 ɖihaɖu", "2 ɖahaɖu", "1 ɖih", "1 ɖih"],
    118: ["1 raʈ", "1 raʈ", "1 raʈ", "1 raʈ", "1 raʈ", "1 raʈ", "1 raʈ", "1 raʈ", "1 raʈ", "1 raʈi", "1 raʈ", "1 raʈ", "1 raʈ"],
    119: ["1 bejiβel", "1 bejiβel", "1 bejiwel", "4 hakaiwʌ", "2 wegidz", "2 wegi", "3 haɳɖare", "3 hoɳɖaro", "3 hoɳɖare", "3 ʂʌɳɖare", "3 hoɳɖare", "3 hoɳɖare", "5 uʈjain"],
    120: ["1 bopʌr", "2 madʒhan", "2 madʒun", "7 madʒni-wʌ", "3 ɖihe", "3 ɖihe", "4 mathe-ɖihi", "3 ɖiho, 4 mate-ɖihi", "5 mathe-ɖahaɖu", "6 t̪s̪hoʈipʌr-ɖahaɖu", "5 mate-ɖhahaɖu", "4 mathe-ɖihi", "8 bor-ɖih"],
    121: ["1, 6 βatiwel", "1, 6 βatiwel", "1 wahʌdʒiβel", "6 wathli-wʌ", "2 hãndze-po", "3 hasti-wele", "4 hanspʌr", "4 hanspʌir", "5 ʂaɳʈo", "5 ʂaɳʈo", "5 ʂaɳʈu", "4 hanspʌr", "7 welʈo"],
    122: ["2 hʌkaɭ", "2 hʌkaɭ", "2 hʌkaɭ", "1 kal", "1 kal", "1 kal", "1 kal", "1 kal", "1 kal", "1 kal", "1 kal", "1 kal", "1 kal"],
    123: ["1 adz", "1 adʒ", "1 adʒ", "1 adʒ", "1 adz", "1 adz", "1 adʒ", "1 adz", "1 adʒ", "1 adz", "1 adz", "1 adz", "1 adz"],
    124: ["1 hakal", "1 hakal", "1 hakal", "1 hakai-wʌ", "2 handa", "2 handa", "3 waɳe", "3 waɳe", "3 wahaɳe", "3 wahaɳe", "3 wahaɳe", "3 waɳe", "4 hekaɭe"],
    125: ["1 at̪huɖo", "1 at̪huɖõ", "1 at̪hʌuɖo", "1 at̪haoɖa", "2 aʈ", "1 atoɖʌ", "3 hapʈu", "3 apʈu", "4 saʈ-ɖahaɖe, 2 haʈ", "3 hapʈu", "1 at̪hauɖa", "1 at̪hoʔɖu", "3 apʈo"],
    126: ["1 maiɳo", "1 mʌiɳo", "1 moiɳo", "1 mʌiɳo", "1 mojʌnu", "1 moinu", "1 maonu", "1 mʌiɳo", "1 mohoɳo", "1 mohʌɳo", "1 mʌiɳu", "1 mʌiɳo", "1 mʌiɳo"],
    127: ["1 warʌhõ", "1 warhõ", "1 warʂõ", "1 warhõ", "1 wariho", "1 warih", "1 wari", "1 warih", "1 warih", "1 waris", "1 wariʂ", "1 warih", "1 worih"],
    128: ["1 dʒuno", "1 dʒuɳo", "1 dʒuno", "1 dʒuɳo", "1 dzunu", "1 dzunʌ", "1 dʒunlo", "1 dʒuɳlo", "1 dʒuɳlu", "1 dʒuɳlʌ", "1 dʒuɳlo", "1 dʒuɳlo", "1 dʒuɳalo"],
    129: ["1 nowo", "1 nʌβo", "1 nʌβo", "1 ɳowo", "1 nowu", "1 naowʌ", "1 ɳaoɭʌ", "1 ɳaolo", "1 ɳaulu", "1 naulʌ", "1 ɳaolo", "1 ɳaolo", "1 ɳawalo"],
    130: ["1 harõ", "1 haro", "1 haro", "2 hadzʌ", "2 hadzu", "2 hadzo", "2 haɖzlʌ", "2 hadzʌlo", "1 warlo", "1 warlʌ", "1 warlo", "2 haɖzlo", "2 hadzo"],
    131: ["1 harõ-nʌha", "1 haro-naha", "1 nah-haro", "2 haɖzo-nʌha", "2 na-hadzu", "2 ne-hadzu", "2 ne-hadzʌ", "3 khʌrap", "3 khʌrap, 1 ni-waru", "3 khʌrap", "1 ni-waru", "3 khʌrap", "3 khʌrap"],
    132: ["1 t̪it̪ʌlo", "1 t̪it̪ʌlõ", "1 t̪it̪ʌlo", "1 t̪it̪lʌ", "2 pigli", "2 pigui", "2 bigli", "2 biglo", "2 bhiglo, 1 t̪iɖʌlo", "2 bhiglo", "2 bhiglo", "2 bhiglo", "3 bidʒel"],
    133: ["1 hukalo", "1 hukalõ", "1 hukʌlo", "1 hukailʌ", "1 ugalu", "1 ugauu", "1 huklʌ", "1 huklo", "1 huklo", "1 suklʌ", "1 sukːʌ", "1 hukʌlʌ", "1 hukeli"],
    134: ["1 lambo", "1 lambo", "1 lambo", "1 lambo", "1 nambi", "1 nambi", "1 ɳambʌlʌ", "1 ɳambʌɭi", "1 nambo", "1 ɳambo", "1 nambu", "1 lamlo", "1 lambo"],
    135: ["1 tuko", "1 tukõ", "1 tukõ", "3 aiʈo", "1 toki", "1 toki", "2 tsuʈlʌ", "2 tshuʈli", "1 tukʌ", "1 tukʌ", "1 tukʌ", "3 aiʈʌlo", "4 aʈioʂ"],
    136: ["1 t̪aʈo", "2 uɳo", "2 uɳo, 1 t̪aʈõ", "1 t̪aʈõ", "2 uɳo", "2 uɳʌ", "1 t̪aʈʌlo", "1 t̪aʈʌlo", "1 t̪aʈʌlo", "1 t̪aʈʌlo", "1 t̪aʈʌlo", "1 t̪aʈʌlo", "1 t̪aʈalo"],
    137: ["1 helõ", "1 helo", "1 helo", "2 thʌɳɖo", "1 helo", "1 helo", "1 helːo", "1 helːo", "1 hewo", "1 ʂelo", "1 helo", "1 hewo", "1 helo"],
    138: ["1 huɖo", "1 huɖo", "1 huɖo", "1 huɖo", "2 hadzʌlu", "1 hoɖu", "3 dzeuɖu", "3 dzeuɖo", "3 dʒeuɖu", "3 dʒeuɖu", "3 dʒeuɖu", "3 dzeuɖo", "3 dʒeoɖo"],
    139: ["1 ulʈo", "1 ulʈo", "1 ulʈo", "1 ulʈo", "2 baŋgaɖi", "2 baŋgaɖi", "3 ɖakhʌri", "3 ɖakhri", "3 ɖakhriu", "3 ɖakhriu", "3 ɖakriu", "3 ɖakhriu", "3 ɖakhrio"],
    140: ["1 pahi", "1 paihɪ", "1 pahɪ", "5 dzoɖʌ", "2 aʔhʌɳe", "2 aʔhaɳe", "2 ahɳe", "2 ahɳo", "4 haʈe", "4 saʈe", "4 haʈe", "2 aɭaɳo, 4 haʈe", "2 ahɳe"],
}
FIELDS = [
    "Item", "Gloss", "Site_Code", "PDF_Page", "Printed_Page", "Column",
    "Manual_Transcription", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = "manual-source-image; rendered-400dpi; OCR-not-accepted"
HIGH_RES = {
    *( (item, site) for item in range(106, 110) for site in SITES ),
    *( (item, site) for item in range(110, 115) for site in SITES ),
    *( (item, site) for item in (120, 125, 140) for site in SITES ),
}


def page_for(item: int) -> int:
    if item <= 109:
        return 112
    return 113 + (item - 110) // 5


def main() -> None:
    rows = []
    for item in range(106, 141):
        assert len(FORMS[item]) == 13
        page = page_for(item)
        for index, (site, form) in enumerate(zip(SITES, FORMS[item])):
            method = METHOD
            if (item, site) in HIGH_RES:
                method = "manual-source-image; rendered-400dpi+900dpi-rereview; OCR-not-accepted"
            row = {
                "Item": str(item), "Gloss": GLOSSES[item], "Site_Code": site,
                "PDF_Page": str(page), "Printed_Page": str(page - 8),
                "Column": "left" if index < 6 else "right",
                "Manual_Transcription": form, "Review_Status": "attested",
                "Confidence": "high", "Uncertainty": "",
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
