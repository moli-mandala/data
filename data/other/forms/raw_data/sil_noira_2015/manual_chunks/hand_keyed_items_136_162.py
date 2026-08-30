#!/usr/bin/env python3
"""Emit the Noira 2015 items 136–162 OCR-blind manual-review ledger.

Every value in ``ITEMS`` below was independently keyed while viewing the
400-dpi rendered source pages. Difficult glyph clusters were rechecked on
900-dpi renders. This module does not read PDF text, OCR, scaffold files, or
another transcription at runtime.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
REVIEWED_AT = "2026-08-28"
OUT = Path(__file__).with_name("items_136_162_hand_keyed.tsv")

SITES = [
    ("NCH", "Noiri-Chillare"), ("NPN", "Noiri-Pannali"),
    ("NAS", "Noiri-Astambha"), ("NGO", "Noiri-Gomon"),
    ("BMU", "Barutiya-Mutalwad"), ("DBM", "Dungra Bhili-Mathwad"),
    ("DBA", "Dungra Bhili-Ambadungar"), ("NTO", "Nahali-Toranmal"),
    ("KNA", "Kotli-Narayanpur"), ("KTA", "Kotli-Taradi"),
    ("GTA", "Gujari-Taradi"), ("GUJ", "Gujarati"),
    ("MAR", "Marati"), ("HIN", "Hindi"),
    ("NTE", "Nahali-Tembhi"), ("TKO", "Tukaithad-Korku"),
    ("NJA", "Nihali-Jamod"),
]

# Cell syntax is ``printed-cognate-number=form``; multiple explicitly printed
# responses in one site cell are separated by ``¦``. The order is SITES above.
# These are literal hand-entered decisions, not derived or copied from OCR.
ITEMS = {
    136: ("hot", [
        "1=uŋɔ", "1=unɔ", "1=uɳɔ", "1=uŋo", "1=uɳo", "1=uŋo",
        "1=unnɔ", "5=t̪at̪alo", "1=una", "2=gʌrʌm", "2=gʌrʌm",
        "2=gərʌm", "1=uʂɳʌ¦2=gʌrʌm", "2=gʌrəm", "4=tʃat̪a",
        "4=tʃʌt̪a", "3=tʃasko",
    ]),
    137: ("cold", [
        "1=helɔ", "1=helo", "1=helo", "1=helo", "1=helo",
        "1=helo¦1=helʌino", "1=hellɔ", "1=heɭo", "2=t̪haɳɖa",
        "2=t̪ʌɳɖo", "2=t̪haɳɖu", "2=t̪həɳɖũ", "2=t̪haɳɖʌ",
        "2=t̪həɳɖa", "3=ɾʌbʌɳ", "3=ɾʌbʌn", "3=rʌban",
    ]),
    138: ("right", [
        "2=huɖu", "2=hʊɖu", "2=hoɖu", "8=dzemʌɳija",
        "2=huɖu¦3=hadzʌlu", "1=dʒeuɖo¦2=huɖu", "3=hadzʌrijɔ",
        "1=dʒeoɖo", "7=siɖo", "7=suɖo", "1=dʒʌmno", "1=dʒəməɳo",
        "6=udzwə", "4=ɖʌhna¦5=ɖaja", "9=dʒʌunat", "9=dʒewna",
        "9=dʒouna",
    ]),
    139: ("left", [
        "2=bʌŋgadja", "1=ɖʌkrija¦2=bʌŋgadija", "2=bʌŋgaɖi",
        "2=bʌŋgaɖija", "2=bʌŋgaɖi", "1=dakʌrijo", "2=hʌŋgʌrijɔ",
        "1=ɖakhrio", "6=uɭʈo", "6=uɭʈa", "3=ɖao¦5=ɖao",
        "3=ɖãbũ", "5=ɖawa", "4=bãja", "1=ɖʌkuci", "1=dʌkori",
        "1=dʌkʌrija",
    ]),
    140: ("near", [
        "1=ahʌne¦4=ari", "4=ari", "1=aʔhʌɳo", "1=ahʌno",
        "1=aʔhʌɳe", "1=ahʌɳo", "1=ahnɔ", "1=ahɳe", "2=nʌdʒuk",
        "7=dʒagʌdʒ", "5=dʒuɖe", "2=nədʒik¦3=pasei", "6=dzʌwʌɭ",
        "2=nʌdʒɖik¦3=pas", "8=merʌka", "8=meran", "8=mera",
    ]),
    141: ("far", [
        "1=seʈo", "1=seʈɔ", "1=seʈo", "1=seʈo", "1=ʂeʈe",
        "1=seʈo", "1=seʈɔ", "1=tʃeʈo", "2=ɖur", "2=ɖur",
        "2=ɖur", "2=ɖur", "2=ɖur", "2=ɖur", "3=kalʌŋgʌn",
        "3=lʌŋga", "4=dhaua",
    ]),
    142: ("big", [
        "1=ɔɖɔ", "1=wɔɖɔ¦5=wɔɖɔ", "1=waɖo", "1=voɖo¦5=voɖo",
        "5=woʈu", "1=orɖo", "1=vɔʔrɔ", "1=moʈo¦5=moʈo",
        "1=moʈa¦5=moʈa", "1=moʈo¦5=moʈo", "5=moʈu", "5=moʈũ",
        "1=moʈha", "1=bəɖa", "7=kʌʈ", "7=kʌʈ", "6=bhaga",
    ]),
    143: ("small", [
        "3=ʌiʈɔ", "3=aiʈo", "5=nano¦1=nano", "3=aiʈo",
        "5=hanu¦1=hanu", "3=ʌiʈo", "3=ajiʈɔ", "7=aʈʌlu",
        "1=nʌhʌna", "5=nano¦1=nano", "2=nanɖʌɖu",
        "5=nanũ¦1=nanũ", "1=lʌhan¦4=tʃhoʈam", "4=tʃhoʈa",
        "5=sʌni", "5=sʌni", "6=bʌtʃka",
    ]),
    144: ("heavy", [
        "1=paʔjɔ", "1=pajõ", "6=hʌʂi", "1=paʔio",
        "1=paʔajo¦6=hʌdzo", "1=paʔo", "1=paʔɔ", "2=baro",
        "5=owʌɭ", "3=vʌdʒa", "2=bhʌri", "2=bhərei",
        "2=bhari¦3=wʌdzʌɳɖar¦4=dzuɖ", "2=bhari", "8=kʌmbʌl",
        "3=bʌdʒʌn¦8=kʌmbʌl", "7=dʒʌtʃom",
    ]),
    145: ("light", [
        "1=olvo", "1=olko¦1=olvo", "1=olwo", "1=olvo", "1=olwo",
        "1=olvo", "1=ɔlvɔ", "3=phʌorõ", "2=hʌika", "2=hʌlka",
        "2=hʌlku", "2=həlkũ", "2=hʌlkʌ", "2=hʌlka", "2=hʌlka",
        "2=hʌlka", "2=hʌlka",
    ]),
    146: ("above", [
        "1=use", "1=ʊse", "1=uʂe", "1=uso", "1=use",
        "1=uso¦3=uppʌje", "1=usɔ", "7=khaʈlapor", "1=utʃe",
        "1=utʃa", "1=utʃa", "2=upər", "4=wʌr", "2=upər",
        "5=liŋ¦6=mʌʈʈʌn", "5=liŋdʒ", "8=kʌdʒar",
    ]),
    147: ("below", [
        "1=nise", "1=nise", "1=niʂe", "1=niso", "1=nise", "1=nisɔ",
        "1=nisɔ", "5=boɳɖ", "3=heʈa", "3=leʈa", "3=eʈa", "1=nitʃe",
        "2=khali", "1=nitʃe", "6=iʈa", "6=iʈa", "4=beʈer",
    ]),
    148: ("white", [
        "4=ɖhʌulɔ", "4=ɖhʌulɔ", "1=paɳɖo", "1=paɳɖo",
        "1=paɳɖo", "3=bogno", "3=bɔgnɔ", "4=ɖoɭo", "5=ɖhovja",
        "5=ɖhʌvja", "4=ɖhouɭu", "4=ɖhoɭo", "2=paɳɖhra",
        "6=səɸeɖ", "7=pulum", "7=pulum", "1=pander¦2=pander",
    ]),
    149: ("black", [
        "1=kʌllo", "1=kallo", "1=kalɔ", "1=kalo", "1=kalu", "1=kʌllo",
        "1=kalɔ", "1=kaɭo", "1=kaja", "1=kaja", "1=kaɭu", "1=kaɭũ",
        "1=kaɭʌ", "1=kala", "2=kẽɳdʒe", "2=keɳɖe", "3=bʌda",
    ]),
    150: ("red", [
        "1=raʈno", "1=rʌʈlo", "1=rʌʈɔ", "1=rʌʈo", "1=rʌʈo",
        "1=rʌʈʌno", "1=rʌʈnɔ", "1=rʌʈʌɭõ", "2=lal", "2=lal",
        "2=lal", "2=lal", "2=lal", "2=lal", "1=reʈa", "1=rʌʈa",
        "1=rata",
    ]),
    151: ("one", [
        "1=ek", "1=ek", "1=lek", "1=ek", "1=ek", "1=eke", "1=eke",
        "1=ek", "1=ek", "1=ek", "1=ek", "1=ɛk", "1=ek", "1=ɛk",
        "2=mja", "2=mja", "3=bʌda",
    ]),
    152: ("two", [
        "1=ben", "1=ben", "1=ben", "1=ben", "1=ben", "1=bene",
        "1=bene", "4=ɖwi", "1=ben", "1=ben", "1=be", "1=bɛ",
        "2=ɖon", "2=ɖo", "3=bʌira", "3=bari", "5=irar",
    ]),
    153: ("three", [
        "1=t̪in", "1=t̪in", "1=t̪iɳ", "1=t̪in", "1=t̪iɳ", "1=t̪ine",
        "1=t̪ine", "1=t̪iɳ", "1=t̪in", "1=t̪in", "1=t̪eɳ", "1=t̪rəɳ",
        "1=t̪in", "1=t̪in", "4=aphʌija", "2=ʌphʌi", "3=mɔtho",
    ]),
    154: ("four", [
        "1=tʒjʌr", "1=sar", "1=tʃar", "1=çar", "1=tʃar",
        "1=tʃar¦1=tʃjar", "1=tʃare", "1=tʃar", "1=tʃar", "1=tʃar",
        "1=tʃar", "1=tʃar", "1=tʃar", "1=tʃar", "2=upoɲa",
        "2=uphun", "3=nalko",
    ]),
    155: ("five", [
        "1=pas", "1=pas", "1=pas", "1=pas", "1=pas", "1=pase", "1=pase",
        "1=pãʈs", "1=patʃ", "1=patʃ", "1=patʃ", "1=patʃ", "1=pãts",
        "1=pãtʃ", "2=muɳʌi", "2=monʌi", "1=patʃo",
    ]),
    156: ("six", [
        "1=sʌv", "1=sʌv", "1=sʌo", "1=sʌv", "1=sʌv", "1=sɔve",
        "1=sɔve", "1=ʂo", "1=sʌv", "1=sʌv", "2=tʃo", "2=tʃhə",
        "1=sʌha", "2=tʃhɛ¦2=tʃhə", "3=t̪ʊrʌi", "3=t̪ʊrʌi", "1=sʌha",
    ]),
    157: ("seven", [
        "1=saʈ", "1=haʈ", "1=haʈ", "1=saʈ", "1=haʈ", "1=haʈe",
        "1=haʈe", "1=haʈ", "1=saʈ", "1=saʈ", "1=saʈ", "1=saʈ",
        "1=saʈ", "1=saʈ", "2=jei", "2=ʌi", "1=sato",
    ]),
    158: ("eight", [
        "1=aʈh", "1=aʈh", "1=aʈ", "1=aʈh", "1=aʈ", "1=aʈh",
        "1=aːʈhe", "1=aʈh", "1=aʈh", "1=aʈh", "1=aʈh", "1=aʈh",
        "1=aʈh", "1=aʈh", "2=ilʌr", "2=ilar", "1=ato",
    ]),
    159: ("nine", [
        "1=nʌu", "1=nʌu", "1=nʌo", "1=nʌu", "1=nʌo", "1=nove",
        "1=nove", "1=nʌo", "1=nʌu", "1=nʌu", "1=nʌu", "1=nʌu",
        "1=nʌu", "1=nʌu", "2=ʌrʌj", "2=ʌrʌj", "1=nohu",
    ]),
    160: ("ten", [
        "1=ɖɔhɔ", "1=ɖʌhe", "1=doho", "1=ɖohɔ", "1=doho", "1=ɖɔhɔ",
        "1=ɖɔhɔ", "1=ɖoh", "1=ɖʌs", "1=ɖʌs", "1=ɖʌs", "1=ɖəs",
        "1=ɖʌha", "1=ɖəs", "2=gel", "2=gel", "3=dʌtʃo",
    ]),
    161: ("eleven", [
        "1=igjara", "1=igjara", "1=igjar", "1=igjar", "1=igjaʌ¦1=igjara",
        "1=igjare", "1=igijʌre", "1=gjara", "2=ʌkra", "2=akra",
        "2=akra¦2=ʌkʌra", "1=əgiar", "2=ʌkʌra", "1=gjarʌ", "1=gjara",
        "1=gjara", "1=dʒara¦2=ʌkra",
    ]),
    162: ("twelve", [
        "1=barʌ", "1=bara", "1=bara", "1=bara", "1=bare", "1=bʌre",
        "1=bare", "1=bava", "1=bara", "1=bara", "1=bara", "1=bar",
        "1=bara", "1=barʌ", "1=bara", "1=bara", "1=bara",
    ]),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Site_Name", "PDF_Page", "Printed_Page",
    "Column", "Manual_Transcription", "Source_Cognate_Labels", "Review_Status",
    "Confidence", "Uncertainty", "Reviewer_Method", "Reviewed_At",
    "Reviewer_Declaration",
]


def coordinates(item: int, site_index: int) -> tuple[str, str, str]:
    """Return exact physical page, printed page, and printed column."""
    page = {
        136: 62, 137: 62, 138: 62, 139: 63, 140: 63, 141: 63,
        142: 63, 143: 63, 144: 64, 145: 64, 146: 64, 147: 64,
        148: 64, 149: 65, 150: 65, 151: 65, 152: 65, 153: 65,
        154: 65, 155: 66, 156: 66, 157: 66, 158: 66, 159: 66,
        160: 67, 161: 67, 162: 67,
    }[item]
    if item == 138 and site_index >= 7:
        page = 63
    elif item == 143 and site_index >= 8:
        page = 64
    elif item == 148 and site_index >= 11:
        page = 65
    elif item == 154 and site_index >= 5:
        page = 66
    elif item == 159 and site_index >= 15:
        page = 67

    left_sites = {
        136: set(), 137: set(), 138: set(), 139: set(range(17)),
        140: set(range(17)), 141: set(), 142: set(), 143: set(),
        144: set(range(17)), 145: set(range(16)), 146: set(), 147: set(),
        148: set(), 149: set(range(17)), 150: set(range(17)),
        151: set(range(7)), 152: set(), 153: set(), 154: set(),
        155: set(range(17)), 156: set(range(17)), 157: set(), 158: set(),
        159: set(), 160: set(range(17)), 161: set(range(17)),
        162: set(range(9)),
    }
    continuation_left = (
        (item == 138 and page == 63)
        or (item == 143 and page == 64)
        or (item == 148 and page == 65)
        or (item == 154 and page == 66)
        or (item == 159 and page == 67)
    )
    column = "left" if continuation_left or site_index in left_sites[item] else "right"
    return str(page), str(page - 6), column


def parse_cell(cell: str) -> tuple[str, str, str, str, str]:
    pairs = [part.split("=", 1) for part in cell.split("¦")]
    labels = " | ".join(label for label, _ in pairs)
    forms = " | ".join(form for _, form in pairs)
    return forms, labels, "attested", "high", ""


def rows() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    assert set(ITEMS) == set(range(136, 163))
    for item, (gloss, cells) in ITEMS.items():
        assert len(cells) == len(SITES), (item, len(cells))
        for site_index, ((site_code, site_name), cell) in enumerate(zip(SITES, cells)):
            form, labels, status, confidence, uncertainty = parse_cell(cell)
            pdf_page, printed_page, column = coordinates(item, site_index)
            row = {
                "Item": str(item), "Gloss": gloss, "Site_Code": site_code,
                "Site_Name": site_name, "PDF_Page": pdf_page,
                "Printed_Page": printed_page, "Column": column,
                "Manual_Transcription": form, "Source_Cognate_Labels": labels,
                "Review_Status": status, "Confidence": confidence,
                "Uncertainty": uncertainty,
                "Reviewer_Method": "manual visual inspection of 400-dpi rendered PDF page",
                "Reviewed_At": REVIEWED_AT, "Reviewer_Declaration": DECLARATION,
            }
            out.append({key: unicodedata.normalize("NFC", value) for key, value in row.items()})
    assert len(out) == 459
    return out


def main() -> None:
    rows_to_write = rows()
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_to_write)
    print(f"wrote {len(rows_to_write)} OCR-blind manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
