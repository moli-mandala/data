#!/usr/bin/env python3
"""Emit the Noira 2015 items 1–27 OCR-blind manual-review ledger.

Every value in ``ITEMS`` below was independently keyed while viewing the
400-dpi rendered source pages.  This module does not read PDF text, OCR,
scaffold files, or another transcription at runtime.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
REVIEWED_AT = "2026-08-28"
OUT = Path(__file__).with_name("items_001_027_hand_keyed.tsv")

SITES = [
    ("NCH", "Noiri-Chillare"),
    ("NPN", "Noiri-Pannali"),
    ("NAS", "Noiri-Astambha"),
    ("NGO", "Noiri-Gomon"),
    ("BMU", "Barutiya-Mutalwad"),
    ("DBM", "Dungra Bhili-Mathwad"),
    ("DBA", "Dungra Bhili-Ambadungar"),
    ("NTO", "Nahali-Toranmal"),
    ("KNA", "Kotli-Narayanpur"),
    ("KTA", "Kotli-Taradi"),
    ("GTA", "Gujari-Taradi"),
    ("GUJ", "Gujarati"),
    ("MAR", "Marati"),
    ("HIN", "Hindi"),
    ("NTE", "Nahali-Tembhi"),
    ("TKO", "Tukaithad-Korku"),
    ("NJA", "Nihali-Jamod"),
]

# Cell syntax is ``printed-cognate-number=form``; multiple explicitly printed
# responses in one site cell are separated by ``¦``.  ``BLANK`` means the
# source explicitly prints "0 no entry".  The order is exactly SITES above.
ITEMS = {
    1: ("body", [
        "2=ɖin", "2=ɖiɭ", "2=ɖiɭ", "2=ɖin", "2=ɖiɭ", "2=ɖine",
        "2=ɖine", "2=ɖiɭ", "2=ɖiɭ", "3=aŋg", "1=çʌrir", "1=ʃərir",
        "1=sərir¦3=ʌŋg", "1=ʃərir", "4=dʒju", "4=dʒju", "5=pʌkor",
    ]),
    2: ("head", [
        "2=mungo¦3=mungo", "1=munɖ", "3=mundko", "3=munka", "3=mundkʌ",
        "1=muɖ", "1=mʊɳɖe¦2=mʊɳɖe", "4=maʈha", "3=muŋka", "3=munka",
        "4=maʈha", "4=maʈhũ", "6=ɖoke", "5=sɪr", "8=kapar", "8=kʌpar",
        "9=peĩ",
    ]),
    3: ("hair", [
        "1=kɛhe¦2=sindʒe", "1=kɛhẽ", "2=siŋgje¦2=sindʒe", "1=kɛhẽ",
        "2=sindʒe", "1=kẽhe", "1=kẽhe¦3=nɪmɑle", "1=kẽh", "5=bɑl",
        "5=bal", "5=bal", "5=βɑɭ", "6=kes", "5=bɑl", "7=hup", "7=hup",
        "8=kukso",
    ]),
    4: ("face", [
        "1=sub", "1=sub", "1=sob", "1=sub", "1=sob", "1=sube", "1=sube",
        "4=muj", "4=muj", "4=muj", "4=muɖʊ", "2=tʃɛro¦4=moɖhũ¦4=mõh",
        "3=tʃɛhəra", "3=tʃʌhəra¦4=mʊkh¦4=mũh", "8=mʊar",
        "8=mar¦8=mʊar", "8=muahar",
    ]),
    5: ("eye", [
        "1=doʔa¦3=doʔa", "1=doʔa¦3=doʔa", "1=doʔa¦3=doʔa",
        "1=doʔa¦3=doʔa", "1=doʔa¦3=ɖou¦3=doʔa", "1=duʔa¦3=duʔa",
        "1=ɖʊaʔ", "3=ɖoɭo", "3=ɖoja", "3=ɖoja", "3=ɖoɭa", "2=əŋkh",
        "3=ɖoɭa", "2=ãkh", "7=meʈ", "7=meɖ", "8=dʒikitʃ",
    ]),
    6: ("ear", [
        "1=kan", "1=kan", "1=kan", "1=kan", "1=kaɳ", "1=kan", "1=kane",
        "1=kaɭ", "1=kan", "1=kan", "1=kan", "1=kan", "1=kan", "1=kan",
        "2=lʊʈʊr", "2=lʊʈʊr", "3=tʃigam",
    ]),
    7: ("nose", [
        "1=ɳak", "1=nak", "1=nakh", "1=ɳak", "1=ɳak", "1=nak", "1=ɳake",
        "1=ɳakh", "1=nak", "1=ɳak", "1=ɳak", "1=nak", "1=nak", "1=nak",
        "2=mu", "2=mu", "3=tʃon",
    ]),
    8: ("mouth", [
        "1=sub", "1=sub", "1=sob", "1=sub", "1=sob", "1=sube", "1=sube",
        "2=mui", "2=muj", "2=muj", "4=wɔʈ", "2=mõh",
        "2=mukh¦3=ʈõɳɖ", "2=mũh", "5=tʃabu", "5=tʃabu¦7=koʈo", "6=knogo",
    ]),
    9: ("tooth", [
        "1=ɖaʈ", "1=ɖaʈ", "1=daʈh", "1=ɖaʈ", "1=ɖaʈ", "1=ɖaʈ",
        "1=ɖaʈe", "1=ɖaʈ", "1=ɖaʈ", "1=ɖaʈ", "1=ɖaʈ", "1=ɖanʈ",
        "1=ɖaʈə", "1=ɖãʈ", "3=ʈiʈin¦3=ʈirin", "3=ʈirin", "2=meŋge",
    ]),
    10: ("tongue", [
        "1=dʒib", "1=dʒib", "1=dʒib", "1=dʒib", "1=dʒib", "1=dʒibe",
        "1=dʒibe", "1=dʒib", "1=dʒib", "1=dʒib", "1=dʒib", "1=dʒibh",
        "1=dʒibh", "1=dʒibh", "2=lan", "2=lan", "2=lai",
    ]),
    11: ("breast", [
        "1=tʃʌʈi", "1=saʈi", "2=buɖʒi", "BLANK", "1=saʈi¦2=buɖʒi",
        "1=saʈi", "2=buɖʒi", "1=saʈi¦4=ɖai", "BLANK", "BLANK", "1=tʃaʈi",
        "1=tʃaʈi", "3=sʈʌn", "1=tʃaʈi¦3=sʈʌn", "BLANK", "BLANK", "BLANK",
    ]),
    12: ("belly", [
        "1=putta¦1=puttu", "1=puttu", "1=poʈu", "1=putta¦1=puttu", "1=poʈu",
        "1=poʈu", "1=puʈʊ", "1=poʈʌɭiu", "1=peʈ", "1=peʈi", "1=peʈ",
        "1=pɛʈh", "1=pot", "1=pɛʈ", "4=ladʒ", "4=ladʒ", "5=popo",
    ]),
    13: ("arm", [
        "1=aːʈ", "1=aʈ", "1=ath", "1=aʈ", "1=ath", "1=aʈhe", "1=aːʈhe",
        "1=aʈh", "1=aʈ", "1=haʈ", "1=haʈ", "1=haʈh", "1=haʈh¦2=bãh",
        "1=haʈh¦2=bãh", "3=ʈi", "3=ʈi", "4=bʌkko",
    ]),
    14: ("elbow", [
        "2=khum", "2=khum", "1=khumi¦2=khumi", "1=koɳi", "1=khumi¦2=khumi",
        "1=koɳi", "1=koɳi", "1=khuɳi", "4=guʈʈa", "5=huɳɖo", "4=guʈi",
        "1=kõɳi", "3=kopʌr", "1=kohəni", "1=koini", "1=koini", "1=koini",
    ]),
    15: ("palm", [
        "1=thɔɭʌʈ", "1=ʈhɔɭʈi", "1=ʈhoɭʌʈ", "1=thɔɭʌʈ",
        "1=ʈhɔɭɔʈ¦1=ʈhoɭʈu", "1=ʈɔɭeʈe", "1=ʈɔɭɔʈe", "8=ʈeɭsõ",
        "4=ʈʌjaʈ", "5=ʈẽj", "2=haʈ", "2=hʌʈheli", "2=ʈʌɭʌhaʈ",
        "2=hʌʈheli", "6=ʈiʈala", "6=ʈiʈala", "7=bʌkku¦7=midʒar",
    ]),
    16: ("finger", [
        "1=ʌŋʈi", "1=aŋʈi", "1=aiŋgi¦1=aiŋgu", "2=akiʈi", "1=aŋgu¦1=ʌŋʈu",
        "1=ʌŋguʈi", "1=aŋguʈija", "1=aŋguɭ", "1=ʌŋʈi", "1=ʌŋʈi",
        "1=aŋgʌɭi", "1=aŋgɭi", "1=aŋgoɖi", "1=ãguli", "3=boʈo", "3=boʈo",
        "4=kʌɳɖa",
    ]),
    17: ("fingernail", [
        "1=nʌkh", "1=nɔk", "1=nʌkh", "1=nɔkh", "1=nokh", "1=nɔkhe",
        "1=nɔkhe", "1=nekh", "1=nʌk", "1=nak", "1=nʌk", "1=nəkh",
        "1=nʌkh", "1=nʌkh¦1=nʌkhun", "1=neko", "1=neko", "1=nakho",
    ]),
    18: ("leg", [
        "1=guɖu", "1=guɖu", "1=guɖu", "1=guɽ", "1=guɖu", "1=guɖu",
        "1=guɽ", "2=pai¦3=pai", "2=pag¦3=pag", "6=tʌŋgɖo",
        "2=pʌg¦3=pʌg", "2=pʌg¦3=pʌg", "2=pai¦3=pai",
        "2=ʈãg¦3=pəir¦4=ʈãg", "7=naŋa", "7=naŋga", "8=kuri",
    ]),
    19: ("skin", [
        "1=sambɽi", "1=sʌmʌro", "1=ambaɖu", "1=sambaɖɔ", "1=ʂamoɖo",
        "1=tʃʌmɖɔ", "1=samʊɽɔ", "1=ʈsambaɖo", "1=tʃʌmʌɖi", "1=tʃamɽa",
        "1=tʃamʌɽa", "1=tʃaməɖi", "1=tʃambaɖi", "1=tʃəməɖa",
        "1=tʃambre", "1=tʃʌmbre", "2=ʈol",
    ]),
    20: ("bone", [
        "1=aʈhkɔ", "1=aʈiko", "1=aʈko", "4=hʌɖka", "3=aɖ",
        "1=aɽko¦4=haɖɖe¦4=hʌɖɖe", "2=aːɽe", "3=aɖɛ", "4=hʌɖəkka",
        "4=hʌɽika", "4=hʌɖʌka", "4=haɖəkũ", "3=haɖ¦4=haɖuk", "4=həɖɖi",
        "6=harge", "6=bʌrge", "7=pʌkʈo",
    ]),
    21: ("heart", [
        "4=kaldʒo¦5=pupsja", "BLANK", "BLANK", "5=puprija", "BLANK", "6=mon",
        "BLANK", "BLANK", "BLANK", "BLANK", "2=rʊɖʌj", "1=ɖil¦2=rəɖai",
        "3=rʌkʈʊ", "2=hriɖəi", "1=ɖil", "1=ɖil", "BLANK",
    ]),
    22: ("blood", [
        "5=rʌkʈɔ", "5=rokʈɔ", "5=rogʌʈ", "5=rʌkʈ", "5=rokto", "1=nuje",
        "1=noje", "2=ɭoi", "5=rʌgʌʈ", "5=rʌkʌʈ", "5=rʌkʈ", "2=lohi¦2=loi",
        "5=rʌkʈʌ", "3=khun¦5=rʌkʈrə", "6=mʌjʌm", "3=khun¦6=mʌjʌm",
        "7=tʃorʈo",
    ]),
    23: ("urine", [
        "BLANK", "BLANK", "1=muʈh", "BLANK", "BLANK", "BLANK", "1=muʈrɪɔ",
        "1=muʈ", "BLANK", "BLANK", "BLANK", "1=moʈər¦1=muʈrə¦2=pɛʃəb",
        "1=muʈrə¦3=lʌgβi", "1=muʈrə¦2=pɛʃəb", "BLANK", "BLANK", "BLANK",
    ]),
    24: ("feces", [
        "BLANK", "BLANK", "5=logit", "BLANK", "BLANK", "BLANK", "1=ogjɔ",
        "1=agio", "BLANK", "BLANK", "BLANK", "2=tʌtti¦3=gu", "3=gu¦4=mʌl",
        "2=tʌtti¦3=guh¦4=mʌl", "BLANK", "BLANK", "BLANK",
    ]),
    25: ("village", [
        "1=gau", "1=gav", "1=gau", "1=gav", "1=gau", "1=game", "1=game",
        "1=gaũ", "1=gav", "1=gau", "1=gau", "1=gam", "1=gaũ", "1=gaũ",
        "1=gav", "1=gav", "2=bia",
    ]),
    26: ("house", [
        "3=koʔo", "3=koʔo", "3=ko¦3=koʔo", "3=ko¦3=koʔo", "3=koʔo",
        "3=koʔo", "3=kɔʔɔ", "1=gjar", "1=ghʌr", "1=gɦʌr", "1=gher",
        "1=gɦʌr", "1=gɦʌr¦2=məkan", "1=gɦʌr¦2=məkan", "6=ura", "6=ura",
        "5=avar",
    ]),
    27: ("roof", [
        "5=benu pʌha¦6=ɖaba", "5=benu pʌha¦6=ɖaba", "5=pʌha",
        "5=ben pʌhe", "5=pʌha¦6=ɖaba", "7=malu", "1=nɔlje", "10=ʂeʈ",
        "8=paʈʃ", "6=ɖaba", "6=ɖabu", "3=tʃapərũ", "3=tʃʌpʌr",
        "4=tʃhəʈ", "9=paɖiri", "9=paɖiɽi", "9=pʌɖiri",
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
    if item <= 4:
        page = 33
    elif item == 5:
        page = 33 if site_index <= 4 else 34
    elif item <= 9:
        page = 34
    elif item == 10:
        page = 34 if site_index <= 13 else 35
    elif item <= 14:
        page = 35
    elif item == 15:
        page = 36 if site_index == 16 else 35
    elif item <= 19:
        page = 36
    elif item == 20:
        page = 37 if site_index >= 15 else 36
    elif item <= 24:
        page = 37
    elif item == 25:
        page = 38 if site_index >= 15 else 37
    else:
        page = 38

    left_sites = {
        1: range(17), 2: range(17), 3: range(5), 4: range(0),
        5: range(5, 17), 6: range(17), 7: range(17), 8: range(1),
        9: range(0), 10: range(14, 17), 11: range(17), 12: range(17),
        13: range(6), 14: range(0), 15: range(0), 16: range(17),
        17: range(17), 18: range(8), 19: range(0), 20: range(0),
        21: range(17), 22: range(17), 23: range(7), 24: range(0),
        25: range(0), 26: range(17), 27: range(17),
    }
    # Continuations that moved to the next physical page begin in the left column.
    if (item, page) in {(5, 34), (10, 35), (15, 36), (20, 37), (25, 38)}:
        column = "left"
    else:
        column = "left" if site_index in left_sites[item] else "right"
    return str(page), str(page - 6), column


def parse_cell(cell: str) -> tuple[str, str, str, str, str]:
    if cell == "BLANK":
        return "", "", "source_blank", "high", "source explicitly prints '0 no entry'"
    pairs = [part.split("=", 1) for part in cell.split("¦")]
    labels = " | ".join(label for label, _ in pairs)
    forms = " | ".join(form for _, form in pairs)
    return forms, labels, "attested", "high", ""


def rows() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    assert set(ITEMS) == set(range(1, 28))
    for item, (gloss, cells) in ITEMS.items():
        assert len(cells) == len(SITES), (item, len(cells))
        for site_index, ((site_code, site_name), cell) in enumerate(zip(SITES, cells)):
            form, labels, status, confidence, uncertainty = parse_cell(cell)
            pdf_page, printed_page, column = coordinates(item, site_index)
            row = {
                "Item": str(item),
                "Gloss": gloss,
                "Site_Code": site_code,
                "Site_Name": site_name,
                "PDF_Page": pdf_page,
                "Printed_Page": printed_page,
                "Column": column,
                "Manual_Transcription": form,
                "Source_Cognate_Labels": labels,
                "Review_Status": status,
                "Confidence": confidence,
                "Uncertainty": uncertainty,
                "Reviewer_Method": "manual visual inspection of 400-dpi rendered PDF page",
                "Reviewed_At": REVIEWED_AT,
                "Reviewer_Declaration": DECLARATION,
            }
            out.append({key: unicodedata.normalize("NFC", value) for key, value in row.items()})
    assert len(out) == 459
    return out


def main() -> None:
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows())
    print(f"wrote {len(rows())} OCR-blind manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
