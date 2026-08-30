#!/usr/bin/env python3
"""Emit the OCR-blind, visually hand-keyed Appendix C items 141--175 chunk.

Every literal below was entered while inspecting the 400-dpi rendered scan.
Dense numeral/interrogative/demonstrative rows and confusable retroflex or
aspiration glyphs were independently rechecked at 900 dpi. OCR and PDF text
did not seed, supply, or verify any lexical reading.
"""
from __future__ import annotations

import csv
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_141_175_hand_keyed.tsv"
SITES = "KEL DHA DIG AMO MUN AST MAN BHU AML SEG KAN SHA TOR".split()
GLOSSES = {
    141: "far", 142: "big", 143: "small", 144: "heavy", 145: "light",
    146: "above", 147: "below", 148: "white", 149: "black", 150: "red",
    151: "one", 152: "two", 153: "three", 154: "four", 155: "five",
    156: "six", 157: "seven", 158: "eight", 159: "nine", 160: "ten",
    161: "eleven", 162: "twelve", 163: "twenty", 164: "one_hundred",
    165: "who?", 166: "what?", 167: "where?", 168: "when?",
    169: "how_many?", 170: "what_kind?", 171: "this", 172: "that",
    173: "these", 174: "those", 175: "same",
}
FORMS = {
    141: ["1 ʂete", "1 ʂete", "1 ʂete", "2 ɖur", "1 ʂete", "1 ʂete", "1 ʂetu", "1 ʂeto", "1 t̪s̪heto", "1 tʃheto", "1 tʃhetu", "1 ʂetu", "1 ʂeto"],
    142: ["1, 2 moɖo", "1, 2 moɖo", "1, 2 moɖo", "2 motho", "1 woɖu", "1 woɖu", "1, 2 motu", "2 motlo", "2 motlu", "2 motto", "2 mota", "2 motlu", "1, 2 moto"],
    143: ["1 haɳo", "1 haɳo", "1 haɳo", "2 aiʈõ", "1 hanu", "1 hanu", "2 aiʈu", "2, 4 aiʈʌlo", "3 ɳaɳlu", "3 ɳaɳlo", "3 ɳaɳlo", "2, 4 aiʈʌlu", "4 aʈʌlu"],
    144: ["1 woɖzo", "1 wodʒʌo", "1 wʌdʒõ", "1 wʌdʒo", "2 paʔajo", "2 paʔajo", "3 baro", "3 baro", "3 bɦaru", "3 bɦarʌ", "3 bɦaro", "3 bɦaro", "3 baro"],
    145: ["1, 2 olkõ", "1, 2 olkõ", "1, 2 olkõ", "1 alkõ", "2 olwo", "2 olwo", "2 ol:wo", "3 phʌõro", "3 phʌuro", "3 phoʔrʌ", "3 phʌoro", "1 hʌlko", "3 phʌorõ"],
    146: ["1 ut̪s̪o", "1 utʃe", "1 utʃõ", "1 utsʌ", "1 uʂe", "1 uʂe", "3 upʌr", "3 upʌr", "3 oɖʌr", "3 upʌr", "3 upʌr", "3 upʌr", "4 khaʈla-pʌr"],
    147: ["2 t̪ole", "2 t̪ole", "2 thule", "5 etha", "1 niʂe", "1 niʂe", "3 buɳɖe", "3 boɳɖ", "2 towe", "4 ɳeɖʌ", "2 t̪ole", "1 neʈsʌ, 3 buɳɖe", "3 boɳɖ"],
    148: ["1 uɖzʌlõ", "1 udʒʌlõ", "1 udzʌlõ, 2 paɳɖo", "4, 5 dɦojo", "2 paɳɖu", "2 paɳɖo", "3 bul:o", "3 bul:o", "4, 5 ɖɦowu", "4 ɖɦaulio", "4 ɖɦʌolio", "3 bul:o", "5 ɖolo"],
    149: ["1 kalõ", "1 kalo", "1 kalõ", "1 kaijo", "1 kalu", "1 kalu", "1 kal:o", "1 kal:o", "1 kawu", "1 kal:ʌ", "1 kalo", "1 kawo", "1 kalo"],
    150: ["1 raʈõ", "1 raʈõ", "1 raʈõ", "2 lal", "1 raʈo", "1 raʈu", "1 raʈʌlo", "1 raʈʌlo", "1 raʈʌlu", "1 raʈʌlʌ", "1 raʈʌlo", "1 raʈʌlo", "1 raʈʌlõ"],
    151: ["1 ek", "1 ek", "1 ek", "1 ek", "1 ek", "1 ek", "1 ek", "1 ek", "1 ek", "1 ek", "1 ek", "1 ek", "1 ək"],
    152: ["1 beɳ", "1 ben", "1 ben", "1 ben", "1 ben", "1 ben", "2 ɖwi", "2 ɖwi", "2 ɖwi", "2 ɖwi", "3 ɖoɳ", "2 ɖwi", "2 ɖwi"],
    153: ["1 ʈiɳ", "1 ʈiɳe", "1 ʈiɳ", "1 ʈiɳ", "1 ʈiɳ", "1 ʈiɳ", "1 ʈiɳ", "1 ʈiɳ", "1 ʈiɳ", "1 ʈiɳ", "1 ʈiɳ", "1 ʈiɳ", "1 ʈiɳ"],
    154: ["1 tʃar", "1 tʃar", "1 tʃar", "1 tʃar", "1 ʃar", "1 tʃar", "1 tʃar", "1 tʃar", "1 tʃar", "1 tʃar", "1 tʃar", "1 tʃar", "1 tʃar"],
    155: ["1 paʈʂ", "1 paʈʂ", "1 paʈʂ", "1 paʈʂ", "1 pas", "1 pas", "1 pats", "1 paʈʂ", "1 paʈʂ", "1 paʈʂ", "1 patʃ", "1 patʃ", "1 paʈʂ"],
    156: ["1 sʌo", "1 sʌo", "1 sʌo", "1 so", "1 sʌo", "1 sʌo", "1 soʌ", "1 so", "2 t̪sho:", "2 t̪ʂho", "2 t̪ʂho", "3 saha", "1 ʂo"],
    157: ["1 haʈ", "1 saʈ", "1 haʈ", "1 saʈ", "1 haʈ", "1 haʈ", "1 haʈ", "1 haʈ", "1 saʈ", "1 ʂaʈ", "1 saʈ", "1 saʈ", "1 haʈ"],
    158: ["1 aʈ", "1 aʈh", "1 aʈh", "1 aʈh", "1 aʈ", "1 aʈ", "1 aʈ", "1 aʈh", "1 aʈh", "1 aʈh", "1 aʈh", "1 aʈh", "1 aʈh"],
    159: ["1 nau", "1 nau", "1 nʌu", "1 nʌo", "1 nʌo", "1 nʌo", "1 nʌo", "1 nʌo", "1 nau", "1 nʌu", "1 nʌo", "1 nʌo", "1 nʌo"],
    160: ["1 ɖoho", "1 ɖoho", "1 ɖoh", "1 ɖʌs", "1 ɖoho", "1 ɖoho", "1 ɖoh", "1 ɖoh", "1 ɖʌs", "1 ɖʌs", "1 ɖʌs", "1 ɖʌs", "1 ɖoh"],
    161: ["1 igjʌr", "2 akra", "2 ʌkra", "2 ʌkra", "1 igjʌr", "1 igjʌr", "1 igjʌr", "1 igjʌr", "1 igjare", "1 igjʌro", "1 igjare", "2 ʌkʌra", "1 igjara"],
    162: ["1 bara", "1 bara", "1 bara", "1 bara", "1 bare", "1 bar", "1 bar", "1 bar", "1 bare", "1 bare", "1 bara", "1 bara", "1 bara"],
    163: ["1 βihi", "1 βihi", "1 βih", "1 βis", "1 βihi", "1 βis", "1 βihi", "1 βih", "1 βis", "1 bis", "1 bis", "1 βis", "1 βih"],
    164: ["1 hʌo", "1 hʌo", "1 hʌo", "1 hʌo", "1 ek-hʌo", "1 ek-hʌo", "1 ek-ho", "1 ek-ho", "1 ek-so", "1 ek-so:", "1 ek-ho", "1 ek-ho", "1 ho"],
    165: ["1 keɽo", "1 keɽo", "1 keɽo", "3 kuwã", "1 koɽu", "1 koɽu", "2 kuɳ", "2 kuɳ", "2 koɳ", "2 kuɳ", "2 kuɳ", "2 kuɳ", "2 kuɳ"],
    166: ["1 kʌi", "1 kai", "1 kai", "1 kai", "2 ki", "2 ki", "1 kai", "1 kai", "1 kai", "1 kai", "1 kai", "1 kai", "1 kai"],
    167: ["1 kɪhɪ", "1 kɪhɪ", "1 kʌhi", "1 kɪhi", "2 ka", "2 ka", "2 kã", "2 kã", "2 kã", "2 kã", "2 kã", "2 kã", "2 kã"],
    168: ["2 kiɖhi", "2 kiɖhi", "2 kɪɖhi", "5 kʌɖɦa", "2 keɖihi", "2 keɖihi", "3 keʈijʌr", "3 keʈʌr", "3 koʈjʌr", "3 kʌʈjʌr", "3 koʈjʌr", "3 keʈʌr", "2 keɖi"],
    169: ["1 koʈʌhẽ", "1, 2, 4 koʈa", "2 koʈẽ", "4, 5 kʌlɪχ", "3 keteho", "3 ketəho", "1 koʈʌra", "1 koʈʌru", "1 koʈʌra", "1 kʌʈʌra", "1 koʈʌra", "1 koʈʌra", "5 kolakh"],
    170: ["1 kel:-dʒaʈi", "1 kel:i-dʒaʈi", "1 kel:i-dʒaʈi", "4 kʌlik-dzaʈi", "1 kete-dzaʈ", "1 keʈi-dzaʈ", "2 kalʈaɳ", "2 kalʈaɳ", "3 kanla-baʈin", "3 kahale-baʈiɳ", "3 kaɳli-dzaʈin", "3 kalli-dzaʈin", "4 kolak-dzaʈiɳ"],
    171: ["1 o", "1 o", "1 o", "2 jo", "1 õ", "1 o", "2 jo", "2 jo", "3 dʒu", "2 jo", "2 jo", "2 ju", "2 jo"],
    172: ["1 ʈo", "1 ʈo", "1 ʈo", "1 ʈo", "1 ʈo", "1 ʈo", "2 tʃo", "2 tʃo", "3 polu", "3 polo", "3 polo", "2 tʃu", "1 ʈo"],
    173: ["1 ɪ", "1 e", "1 e", "5 ja", "1 e", "1 e", "4, 5 ju", "4 je", "3 dʒe", "4 je", "4 je", "4, 5 ja", "5 ja"],
    174: ["1 ʈe", "1 ʈe", "1 ʈe", "2 tjo", "1 ʈe", "1 ʈe", "2 tʃju", "3 polo", "3 pola", "3 pola", "3 pola", "2 tʃjo", "4 ʈa"],
    175: ["1 harko", "2 hotʃ", "1 ʂarko", "1 ʂarkhẽ", "1 harko", "1 harko", "1 harkʌ", "1 harko", "1 sarkʌ", "1 sarkhʌ", "1 sarkha", "1 sarkha", "1 harkaʂ"],
}
FIELDS = [
    "Item", "Gloss", "Site_Code", "PDF_Page", "Printed_Page", "Column",
    "Manual_Transcription", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = "manual-source-image; rendered-400dpi; OCR-not-accepted"
HIGH_RES = {
    *((item, site) for item in range(141, 150) for site in SITES),
    *((item, site) for item in range(151, 175) for site in SITES),
}


def page_for(item: int) -> int:
    if item <= 144:
        return 119
    return 120 + (item - 145) // 5


def main() -> None:
    rows = []
    for item in range(141, 176):
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
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} hand-keyed review rows to {OUTPUT}")


if __name__ == "__main__":
    main()
