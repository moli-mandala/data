#!/usr/bin/env python3
"""Write visually checked Adi Appendix B cells for items 159--188."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_159_188_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF page; "
    "text scaffold not accepted without cell visual match"
)
SITES = {
    "MN": "Minyong", "BR": "Bori", "RM": "Ramo", "ML": "Milang",
    "PL": "Pailibo", "AS": "Ashing (Bogum Bokang)", "PD": "Padam",
    "SM": "Shimong", "BK": "Bokar",
}

# Each cell is (manual transcription, source cognate label). Pipe-separated
# responses preserve distinct printed response lines and their label sequence.
DATA = {
    159: ("knife (to cut meat)", {"MN": ("jokhuuk", "2"), "BR": ("joʃik", "2"), "RM": ("joʔʃikh", "2"), "ML": ("jogu", "1"), "PL": ("jokʃuuk | jokʃuuk", "2 | 3"), "AS": ("tʃuukd̪o", "3"), "PD": ("joktʃuk | joktʃuk", "2 | 3"), "SM": ("jokhik", "2"), "BK": ("jokʃuuk | jokʃuuk", "2 | 3")}),
    160: ("hammer", {"MN": (None, "0"), "BR": ("mant̪ruŋ", "1"), "RM": ("mərt̪um", "2"), "ML": ("mart̪ul", "2"), "PL": ("mart̪ul", "2"), "AS": ("mart̪ul", "2"), "PD": ("mart̪ul", "2"), "SM": ("mart̪ul", "2"), "BK": ("t̪oh", "3")}),
    161: ("axe", {"MN": ("əguŋ", "1"), "BR": ("həguŋ", "1"), "RM": ("ja", "2"), "ML": ("rapu", "3"), "PL": ("əgu", "1"), "AS": ("əguŋ", "1"), "PD": ("əguŋ", "1"), "SM": ("əguŋ", "1"), "BK": ("ja", "2")}),
    162: ("bow", {"MN": ("ijjə", "1"), "BR": ("itʃe", "3"), "RM": ("iji", "1"), "ML": ("rabha", "2"), "PL": ("uji", "1"), "AS": ("ijpə", "1"), "PD": ("ujji", "1"), "SM": ("ijji", "1"), "BK": ("iji", "1")}),
    163: ("arrow", {"MN": ("əʔuk", "2"), "BR": ("əpuk", "2"), "RM": ("upukh", "4"), "ML": ("appha", "1"), "PL": ("upukh", "4"), "AS": ("pud̪uu", "3"), "PD": ("əpuk", "2"), "SM": ("əpuk", "2"), "BK": ("opuk", "2")}),
    164: ("spear", {"MN": ("ɲud̪uuŋ | ɲud̪uuŋ", "1 | 2"), "BR": ("mud̪uuk", "2"), "RM": ("nuubuŋ", "3"), "ML": ("rad̪aŋ", "4"), "PL": ("nuubu", "3"), "AS": ("uunuŋ", "1"), "PD": ("guunuŋ", "1"), "SM": ("ɲud̪uuŋ | ɲud̪uuŋ", "1 | 2"), "BK": ("nuuŋbuŋ", "3")}),
    165: ("fire", {"MN": ("əmə", "1"), "BR": ("əmə", "1"), "RM": ("em", "1"), "ML": ("ami", "1"), "PL": ("əmə", "1"), "AS": ("əmə", "1"), "PD": ("əmə", "1"), "SM": ("əmə", "1"), "BK": ("əmə", "1")}),
    166: ("ashes", {"MN": ("mət̪ʔi", "3"), "BR": ("məpi", "1"), "RM": ("mid̪bu", "2"), "ML": ("mipi", "1"), "PL": ("mitʃo", "3"), "AS": ("məpio", "1"), "PD": ("məpi", "1"), "SM": ("mət̪pi | mət̪pi", "1 | 3"), "BK": ("məpio", "1")}),
    167: ("smoke", {"MN": ("məjin", "1"), "BR": ("məjin", "1"), "RM": ("məjin", "1"), "ML": ("muukkhuu", "2"), "PL": ("məjin", "1"), "AS": ("muukkhuu", "2"), "PD": ("muukkhuu", "2"), "SM": ("muukkhuu", "2"), "BK": ("muukkhuu", "2")}),
    168: ("candle", {code: (None, "0") for code in SITES}),
    169: ("boat", {"MN": ("əlluŋ", "2"), "BR": ("ʃupuu", "3"), "RM": ("nau", "1"), "ML": ("ət̪kuŋ", "4"), "PL": ("ʃupuu", "3"), "AS": ("ʃupuu", "3"), "PD": ("ət̪kuŋ", "4"), "SM": ("ət̪kuŋ", "4"), "BK": ("ʃupuu", "3")}),
    170: ("road", {"MN": ("d̪aːt̪ə", "2"), "BR": ("bəd̪aŋ", "3"), "RM": ("lamt̪ə", "4"), "ML": ("d̪puu", "2"), "PL": ("ali", "1"), "AS": ("daːpuu", "2"), "PD": ("d̪puu", "2"), "SM": ("d̪puu", "2"), "BK": ("lambə", "4")}),
    171: ("path", {"MN": ("bəd̪aŋ", "1"), "BR": ("bəd̪aŋ", "1"), "RM": ("lambə", "2"), "ML": ("bud̪a", "1"), "PL": ("bəd̪a", "1"), "AS": ("bəd̪aŋ", "1"), "PD": ("bəd̪aŋ", "1"), "SM": ("bəd̪aŋ", "1"), "BK": ("lamt̪ə", "2")}),
    172: ("to go", {"MN": ("gunam", "1"), "BR": ("ənnam", "1"), "RM": ("innam", "1"), "ML": ("hiːt̪uŋ", "2"), "PL": ("innam", "1"), "AS": ("innam", "1"), "PD": ("ənnam", "1"), "SM": ("gunam", "1"), "BK": ("innam", "1")}),
    173: ("to come", {"MN": ("anam", "1"), "BR": ("anam", "1"), "RM": ("õjenam", "2"), "ML": ("haːt̪uŋ", "3"), "PL": ("anam", "1"), "AS": ("anam", "1"), "PD": ("anam", "1"), "SM": ("anam", "1"), "BK": ("onam", "1")}),
    174: ("to stand", {"MN": ("d̪aŋnam", "1"), "BR": ("d̪aːnam", "1"), "RM": ("ropt̪onam", "3"), "ML": ("d̪japt̪uŋ", "2"), "PL": ("d̪knam", "1"), "AS": ("d̪aŋnam", "1"), "PD": ("d̪knam", "1"), "SM": ("d̪aŋnam", "1"), "BK": ("robnam", "3")}),
    175: ("to sit", {"MN": ("d̪unam", "1"), "BR": ("d̪uːnam", "1"), "RM": ("d̪unam", "1"), "ML": ("dʒuŋt̪uŋ", "2"), "PL": ("d̪unam", "1"), "AS": ("d̪unam", "1"), "PD": ("d̪unam", "1"), "SM": ("d̪unam", "1"), "BK": ("d̪unam", "1")}),
    176: ("to lie down", {"MN": ("d̪unohunam", "1"), "BR": ("d̪unonam", "1"), "RM": ("happenam", "2"), "ML": ("dʒuŋkat̪uŋ", "3"), "PL": ("apenam", "2"), "AS": ("apenam", "2"), "PD": ("apenam", "2"), "SM": ("d̪upenam", "2"), "BK": ("apenam", "2")}),
    177: ("to walk", {"MN": ("gunam", "1"), "BR": ("gunam", "1"), "RM": ("ind̪əbəj", "3"), "ML": ("hiːma", "2"), "PL": ("innam", "1"), "AS": ("gunam", "1"), "PD": ("gunam", "1"), "SM": ("gunam", "1"), "BK": ("innam", "1")}),
    178: ("to fly", {"MN": ("d̪ənam", "1"), "BR": ("d̪ənam", "1"), "RM": ("biard̪əbəj", "3"), "ML": ("berma", "2"), "PL": ("d̪ənam", "1"), "AS": ("d̪ənam", "1"), "PD": ("d̪ənam", "1"), "SM": ("d̪ənam", "1"), "BK": ("bjarnam", "4")}),
    179: ("to enter", {"MN": ("aːnam", "2"), "BR": ("aːnam", "2"), "RM": ("ojẽ", "1"), "ML": ("arahama", "3"), "PL": ("rabuanam", "4"), "AS": ("aːnam", "2"), "PD": ("aːnam", "2"), "SM": ("aːnam", "2"), "BK": ("naŋonam", "5")}),
    180: ("to kick", {"MN": ("t̪unam", "1"), "BR": ("t̪unam", "1"), "RM": ("t̪unam", "1"), "ML": ("tʃima", "2"), "PL": ("t̪unam", "1"), "AS": ("t̪unam", "1"), "PD": ("t̪unam", "1"), "SM": ("t̪unam", "1"), "BK": ("t̪unam", "1")}),
    181: ("to swim", {"MN": ("bjanam", "1"), "BR": ("dʒanam", "2"), "RM": ("bjonam", "1"), "ML": ("bjama", "1"), "PL": ("dʒanam", "2"), "AS": ("bjanam", "1"), "PD": ("bjanam", "1"), "SM": ("bjanam", "1"), "BK": ("bjonam", "1")}),
    182: ("to see", {"MN": ("kanam", "1"), "BR": ("kanam", "1"), "RM": ("kõnam", "1"), "ML": ("kama", "1"), "PL": ("kanam", "1"), "AS": ("kanam", "1"), "PD": ("kanam", "1"), "SM": ("kanam", "1"), "BK": ("khõnam", "1")}),
    183: ("to hear", {"MN": ("t̪annam", "1"), "BR": ("t̪ənnam", "1"), "RM": ("t̪anam", "1"), "ML": ("tʃuma", "2"), "PL": ("t̪ət̪nam", "1"), "AS": ("t̪annam", "1"), "PD": ("t̪annam", "1"), "SM": ("t̪annam", "1"), "BK": ("t̪anam", "1")}),
    184: ("to wait", {"MN": ("t̪ojanam", "2"), "BR": ("t̪ojanam", "2"), "RM": ("khõjanam", "1"), "ML": ("dʒuŋkala", "3"), "PL": ("t̪ojanam", "2"), "AS": ("t̪ojanam", "2"), "PD": ("kajanam", "1"), "SM": ("kajanam", "1"), "BK": ("kjaŋnam", "1")}),
    185: ("to cry", {"MN": ("kamnam", "2"), "BR": ("konnam", "2"), "RM": ("kapnam", "2"), "ML": ("huuma", "1"), "PL": ("kapnam", "2"), "AS": ("kannam", "2"), "PD": ("kamnam", "2"), "SM": ("kamnam", "2"), "BK": ("kãpnam", "2")}),
    186: ("to cook", {"MN": ("monam", "2"), "BR": ("keːnam", "2"), "RM": ("monam", "2"), "ML": ("ɲuma", "1"), "PL": ("kənam", "2"), "AS": ("monam", "2"), "PD": ("monam", "2"), "SM": ("monam", "2"), "BK": ("kõnam", "2")}),
    187: ("to boil (water)", {"MN": ("kirnam", "1"), "BR": ("tʃirnam", "1"), "RM": ("əgut̪o", "2"), "ML": ("t̪utʃa", "3"), "PL": ("tʃurnam", "1"), "AS": ("kirnam", "1"), "PD": ("kirnam", "1"), "SM": ("kirnam", "1"), "BK": ("kõnam", "4")}),
    188: ("to eat", {"MN": ("d̪onam", "1"), "BR": ("d̪onam", "1"), "RM": ("d̪onam", "1"), "ML": ("t̪uma", "2"), "PL": ("d̪onam", "1"), "AS": ("d̪onam", "1"), "PD": ("d̪onam", "1"), "SM": ("d̪onam", "1"), "BK": ("d̪õnam", "1")}),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Site_Name", "PDF_Page", "Printed_Page",
    "Column", "Manual_Transcription", "Source_Cognate_Labels",
    "Review_Status", "Confidence", "Uncertainty", "Reviewer_Method",
    "Reviewed_At", "Reviewer_Declaration",
]


def coordinates(item, code):
    if 159 <= item <= 161:
        return "28", "24", "left"
    if item == 162:
        return ("28", "24", "left") if code in {"MN", "BR", "RM", "ML", "PL"} else ("28", "24", "middle")
    if 163 <= item <= 166:
        return "28", "24", "middle"
    if item == 167:
        return ("28", "24", "middle") if code in {"MN", "BR"} else ("28", "24", "right")
    if 168 <= item <= 171:
        return "28", "24", "right"
    if item == 172:
        return ("28", "24", "right") if code in {"MN", "BR"} else ("29", "25", "left")
    if 173 <= item <= 176:
        return "29", "25", "left"
    if item == 177:
        return ("29", "25", "left") if code in {"MN", "BR"} else ("29", "25", "middle")
    if 178 <= item <= 181:
        return "29", "25", "middle"
    if item == 182:
        return ("29", "25", "middle") if code in {"MN", "BR"} else ("29", "25", "right")
    if 183 <= item <= 186:
        return "29", "25", "right"
    if item == 187:
        return ("29", "25", "right") if code in {"MN", "BR"} else ("30", "26", "left")
    return "30", "26", "left"


def build_rows():
    rows = []
    for item, (gloss, cells) in DATA.items():
        assert set(cells) == set(SITES)
        for code, name in SITES.items():
            form, labels = cells[code]
            pdf_page, printed_page, column = coordinates(item, code)
            blank = form is None
            row = {
                "Item": str(item), "Gloss": gloss, "Site_Code": code,
                "Site_Name": name, "PDF_Page": pdf_page,
                "Printed_Page": printed_page, "Column": column,
                "Manual_Transcription": form or "",
                "Source_Cognate_Labels": labels,
                "Review_Status": "source_blank" if blank else "attested",
                "Confidence": "high",
                "Uncertainty": "Source prints cognate label 0 and ‘no entry’." if blank else "",
                "Reviewer_Method": METHOD, "Reviewed_At": "2026-08-28",
                "Reviewer_Declaration": DECLARATION,
            }
            assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
            rows.append(row)
    return rows


def main():
    rows = build_rows()
    assert len(rows) == 30 * 9 == 270
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
