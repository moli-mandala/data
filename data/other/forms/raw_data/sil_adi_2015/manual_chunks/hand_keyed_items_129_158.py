#!/usr/bin/env python3
"""Write visually checked Adi Appendix B cells for items 129--158."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_129_158_hand_keyed.tsv"
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
    129: ("woman", {"MN": ("miməko", "1"), "BR": ("ɲimə", "1"), "RM": ("ɲẽmə", "1"), "ML": ("mami", "1"), "PL": ("ɲimə", "1"), "AS": ("miməko", "1"), "PD": ("miməko", "1"), "SM": ("miməko", "1"), "BK": ("ɲõmẽ", "1")}),
    130: ("father", {"MN": ("jai", "1"), "BR": ("at̪e", "3"), "RM": ("abo", "2"), "ML": ("abe", "2"), "PL": ("abo", "2"), "AS": ("abu", "2"), "PD": ("abu", "2"), "SM": ("abu", "2"), "BK": ("abo", "2")}),
    131: ("mother", {"MN": ("ummo", "4"), "BR": ("aji", "1"), "RM": ("ane", "2"), "ML": ("aji | adʒi", "1 | 3"), "PL": ("ane | ɲene", "2 | 2"), "AS": ("anə", "2"), "PD": ("ane", "2"), "SM": ("ane", "2"), "BK": ("anə | nanə", "2 | 2")}),
    132: ("husband", {"MN": ("midʒiŋ", "2"), "BR": ("milo", "1"), "RM": ("melo", "1"), "ML": ("madʒaŋ", "2"), "PL": ("ɲilo", "1"), "AS": ("milo", "1"), "PD": ("milo", "1"), "SM": ("milo", "1"), "BK": ("melo", "1")}),
    133: ("wife", {"MN": ("mimə", "1"), "BR": ("mijəŋ", "5"), "RM": ("mejaŋ", "5"), "ML": ("mase", "2"), "PL": ("ɲie", "4"), "AS": ("mimə", "1"), "PD": ("meŋ", "3"), "SM": ("meŋ", "3"), "BK": ("mejaŋ", "5")}),
    134: ("son", {"MN": ("ao", "1"), "BR": ("miloao", "4"), "RM": ("horo", "3"), "ML": ("oru", "3"), "PL": ("horo", "3"), "AS": ("ad̪o", "1"), "PD": ("ue", "2"), "SM": ("ao", "1"), "BK": ("horo", "3")}),
    135: ("daughter", {"MN": ("omə | omə", "1 | 3"), "BR": ("ɲiməao", "2"), "RM": ("home", "1"), "ML": ("ormi", "3"), "PL": ("omə | omə", "1 | 3"), "AS": ("omə | omə", "1 | 3"), "PD": ("omə | omə", "1 | 3"), "SM": ("omə | omə", "1 | 3"), "BK": ("omə | omə", "1 | 3")}),
    136: ("elder brother (gen)", {"MN": ("baabi", "2"), "BR": ("bubuuŋ", "3"), "RM": ("ebuuŋ", "3"), "ML": ("baba", "2"), "PL": ("bubuu", "3"), "AS": ("at̪ə", "1"), "PD": ("bubuuŋ", "3"), "SM": ("abuuŋ", "3"), "BK": ("abuuŋ", "3")}),
    137: ("elder sister (gen)", {"MN": ("mame", "3"), "BR": ("ame | ame", "1 | 3"), "RM": ("ame | ame", "1 | 3"), "ML": ("au", "2"), "PL": ("meme", "3"), "AS": ("aːbi", "1"), "PD": ("ami | ami | m mi", "1 | 3 | 3"), "SM": ("meme", "3"), "BK": ("ame | ame", "1 | 3")}),
    138: ("younger brother (gen)", {"MN": ("buro", "2"), "BR": ("anu", "1"), "RM": ("buro", "2"), "ML": ("ani", "1"), "PL": ("buro", "2"), "AS": ("ɲaɲa", "3"), "PD": ("anijaŋburo", "1"), "SM": ("ani", "1"), "BK": ("niro", "2")}),
    139: ("younger sister (gen)", {"MN": ("burmə", "1"), "BR": ("burmə", "1"), "RM": ("burme", "1"), "ML": ("barmi", "1"), "PL": ("burmə", "1"), "AS": ("bannə", "1"), "PD": ("burmə", "1"), "SM": ("burmə", "1"), "BK": ("burmə", "1")}),
    140: ("friend (male)", {"MN": ("aŋoŋ", "2"), "BR": ("adʒoŋ", "1"), "RM": ("adʒen", "1"), "ML": ("aŋo", "2"), "PL": ("adʒen", "1"), "AS": ("adʒoŋ", "1"), "PD": ("aŋoŋ", "2"), "SM": ("aŋoŋ", "2"), "BK": ("adʒen", "1")}),
    141: ("name", {"MN": ("amuun", "1"), "BR": ("amin", "1"), "RM": ("emin", "1"), "ML": ("raman", "2"), "PL": ("amin", "1"), "AS": ("amuun", "1"), "PD": ("amuun", "1"), "SM": ("amuun", "1"), "BK": ("amin", "1")}),
    142: ("village", {"MN": ("d̪oluŋ", "1"), "BR": ("d̪oluŋ", "1"), "RM": ("d̪õluŋ", "1"), "ML": ("himbu", "2"), "PL": ("d̪oulu", "1"), "AS": ("d̪oluŋ", "1"), "PD": ("d̪oluŋ", "1"), "SM": ("d̪oluŋ", "1"), "BK": ("d̪õluŋ", "1")}),
    143: ("house", {"MN": ("əkum", "2"), "BR": ("əraŋ", "1"), "RM": ("ugu", "3"), "ML": ("anuk", "4"), "PL": ("əra", "1"), "AS": ("əraŋ", "1"), "PD": ("əkum", "2"), "SM": ("əkum", "2"), "BK": ("ugu", "3")}),
    144: ("door", {"MN": ("jabgo", "2"), "BR": ("joggo", "2"), "RM": ("japgo", "2"), "ML": ("ad̪um", "3"), "PL": ("japgo", "2"), "AS": ("jaggo", "2"), "PD": ("əjap", "1"), "SM": ("əjap", "1"), "BK": ("japgo", "2")}),
    145: ("window", {"MN": (None, "0"), "BR": ("kirki", "1"), "RM": ("majeŋ", "2"), "ML": ("kirki", "1"), "PL": (None, "0"), "AS": ("kotʃuŋ", "3"), "PD": (None, "0"), "SM": (None, "0"), "BK": ("gudʒuŋaruŋ", "4")}),
    146: ("roof", {"MN": ("muumio", "1"), "BR": ("muloŋ", "1"), "RM": ("namkoŋ", "2"), "ML": ("kjarkio", "4"), "PL": ("muulo", "1"), "AS": ("kumuuŋ", "3"), "PD": ("muloŋ", "1"), "SM": ("muloŋ", "1"), "BK": ("mũloŋ", "1")}),
    147: ("wall of house", {"MN": ("t̪aluŋ", "1"), "BR": ("t̪od̪uk", "2"), "RM": (None, "0"), "ML": ("pard̪ə", "3"), "PL": ("ʃuksi", "4"), "AS": ("tʃuppaŋ", "4"), "PD": ("ʃuppi", "4"), "SM": (None, "0"), "BK": ("tʃipi", "4")}),
    148: ("pillow", {"MN": ("d̪umt̪ən", "1"), "BR": ("d̪umpər", "1"), "RM": ("d̪umt̪om", "1"), "ML": ("d̪umkən", "1"), "PL": ("d̪umt̪an", "1"), "AS": ("d̪umpər", "1"), "PD": ("d̪umt̪ən", "1"), "SM": ("d̪umt̪ən", "1"), "BK": ("d̪umt̪om", "1")}),
    149: ("blanket", {"MN": ("əga", "1"), "BR": ("ədʒe", "1"), "RM": ("d̪uʃaŋ", "3"), "ML": ("jambu", "2"), "PL": ("ʃube", "4"), "AS": ("jombo", "2"), "PD": ("jambo", "2"), "SM": ("əga", "1"), "BK": ("pamʃu", "5")}),
    150: ("ring (on finger)", {"MN": ("lakkap", "1"), "BR": (None, "0"), "RM": ("ʃud̪u", "4"), "ML": ("laktʃi", "2"), "PL": ("lakʃət̪age", "3"), "AS": ("surd̪ut̪", "5"), "PD": ("lakkap", "1"), "SM": (None, "0"), "BK": ("tʃiŋd̪u", "6")}),
    151: ("clothing", {"MN": ("gənaməga", "4"), "BR": ("ədʒe", "1"), "RM": ("edʒekonam", "1"), "ML": ("agu", "5"), "PL": ("ədʒəəjok", "1"), "AS": ("əga", "2"), "PD": ("əgə | əgə", "2 | 5"), "SM": ("əbəgaluk", "3"), "BK": ("edʒekonam", "1")}),
    152: ("cloth", {"MN": ("əga", "2"), "BR": ("ədʒe", "3"), "RM": ("edʒe", "3"), "ML": ("agu", "1"), "PL": ("ədʒe", "3"), "AS": ("əga", "2"), "PD": ("əgə | əgə", "1 | 2"), "SM": ("əga", "2"), "BK": ("edʒe", "3")}),
    153: ("medicine", {"MN": ("kuʃereŋ", "3"), "BR": ("d̪obaj", "2"), "RM": ("d̪obaj", "2"), "ML": ("kusere", "3"), "PL": ("men", "1"), "AS": ("kuʃereŋ", "3"), "PD": ("kuʃereŋ", "3"), "SM": ("kuhereŋ", "3"), "BK": ("d̪obaj", "2")}),
    154: ("paper", {"MN": ("kakot̪", "1"), "BR": ("kagoʃ", "1"), "RM": ("kagədʒ", "1"), "ML": ("kakot̪", "1"), "PL": ("kagoʃ", "1"), "AS": ("kakot̪", "1"), "PD": ("kakot̪", "1"), "SM": ("kakot̪", "1"), "BK": ("kagoʃ", "1")}),
    155: ("needle", {"MN": ("koŋuuŋ", "1"), "BR": ("pəʃu", "2"), "RM": ("pisi", "2"), "ML": ("pesi", "2"), "PL": ("pisi", "2"), "AS": ("pəʃu", "2"), "PD": ("pəʃu", "2"), "SM": ("pəhi", "2"), "BK": ("puusu", "2")}),
    156: ("thread", {"MN": ("ənno", "2"), "BR": ("t̪atʃak", "6"), "RM": ("un", "1"), "ML": ("aŋiu", "3"), "PL": ("nət̪u", "4"), "AS": ("ənno", "2"), "PD": ("ənno", "2"), "SM": ("nojiŋ", "5"), "BK": ("t̪apjak", "6")}),
    157: ("broom", {"MN": ("hamʔək", "2"), "BR": ("əppək", "1"), "RM": ("ʃampək", "2"), "ML": ("ʃampek", "2"), "PL": ("ʃampek", "2"), "AS": ("əppək", "1"), "PD": ("əppək", "1"), "SM": ("əppək", "1"), "BK": ("ʃampək", "2")}),
    158: ("spoon (for eating)", {"MN": ("lukuŋ", "4"), "BR": ("əgut̪", "1"), "RM": ("əjup", "2"), "ML": ("lukuŋ", "4"), "PL": ("ajup", "2"), "AS": ("d̪aru", "3"), "PD": ("kot̪up", "5"), "SM": ("lukuŋ", "4"), "BK": ("ajup", "2")}),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Site_Name", "PDF_Page", "Printed_Page",
    "Column", "Manual_Transcription", "Source_Cognate_Labels",
    "Review_Status", "Confidence", "Uncertainty", "Reviewer_Method",
    "Reviewed_At", "Reviewer_Declaration",
]


def coordinates(item, code):
    if item == 129:
        return ("25", "21", "right") if code in {"MN", "BR"} else ("26", "22", "left")
    if 130 <= item <= 133:
        return "26", "22", "left"
    if 134 <= item <= 137:
        return "26", "22", "middle"
    if 138 <= item <= 142:
        return "26", "22", "right"
    if 143 <= item <= 147:
        return "27", "23", "left"
    if 148 <= item <= 152:
        return ("27", "23", "right") if item == 152 and code == "BK" else ("27", "23", "middle")
    if 153 <= item <= 157:
        return ("28", "24", "left") if item == 157 and code == "BK" else ("27", "23", "right")
    return "28", "24", "left"


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
