#!/usr/bin/env python3
"""Emit the Noira 2015 items 55–81 OCR-blind manual-review ledger.

Every value in ``ITEMS`` below was independently keyed while viewing the
400-dpi rendered source pages. This module does not read PDF text, OCR,
scaffold files, or another transcription at runtime.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
REVIEWED_AT = "2026-08-28"
OUT = Path(__file__).with_name("items_055_081_hand_keyed.tsv")

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
# responses in one site cell are separated by ``¦``. The order is SITES above.
ITEMS = {
    55: ("fire", [
        "1=ag", "1=ag", "1=ag", "1=ag", "1=ag", "1=age",
        "1=aːge¦1=aːgeʔe", "1=agʈhi", "1=ag", "1=ag", "3=wʌsʈel",
        "1=ag¦1=əgni", "1=ag¦1=agni¦3=wisʈo", "1=ag", "4=siŋgʌl",
        "4=siŋgʌl", "5=ɛŋger",
    ]),
    56: ("smoke", [
        "1=tumaɖi¦4=tumaɖi", "1=tumaɖʊ¦4=tumaɖʊ",
        "1=tumaɳo¦4=tumaɳo", "1=tumbro", "1=tumaɳo¦4=tumaɳo",
        "1=tumaɖɔ¦4=tumaɖɔ", "1=tʊmaɽʊ¦4=tʊmaɽʊ", "4=ɖuwaɖo",
        "5=dʊkʊla", "5=dɦukla", "1=dɦuaɖɔ¦4=dɦuaɖɔ", "4=dɦumaɖo",
        "3=dɦur", "2=dɦuã¦3=dɦuã", "2=dɦũja", "2=dũja", "2=dɦũja",
    ]),
    57: ("ash", [
        "5=khʌʔa", "5=khʌʔa", "5=kha¦5=khʌʔa", "5=kha¦5=khʌʔa",
        "5=khʌʔa", "2=nukhuɖu¦6=ʈapʌɳi", "2=nʊkhɽʊ", "9=rokhoɖo",
        "1=rak", "1=rak", "1=rak", "1=rakh¦1=rəkhju", "1=rakhʌ",
        "1=rakh", "7=oːp", "7=hoːp", "8=neʈo",
    ]),
    58: ("mud", [
        "2=garɔ", "2=garu", "1=doru¦2=garu¦7=doru¦7=garu", "9=rabʌrɔ",
        "2=garu¦7=doru", "2=garo¦3=poɳɔ", "5=sigʊʔu", "2=garo",
        "2=garo¦6=kaɖu", "2=gara", "2=garo", "6=kuɖ¦6=kuɖev¦7=kitʃəɖ",
        "8=tʃikhʌɭ", "7=kitʃəɖ", "8=tʃikal", "8=tʃikʌl", "10=bʊdi",
    ]),
    59: ("dust", [
        "4=rɔsʌɖɔ", "4=rɔsʌɖʊ", "1=ʈulo", "1=ʈulo", "1=ʈulo",
        "4=rosʌɖɔ", "1=ʈulɔ", "4=redzʌɖo", "3=pʌputa", "3=pʌputo",
        "3=pʌputa", "2=dɦul", "2=dɦuɭ¦3=phopʌʈʌ", "2=dɦul",
        "2=dɦuɭa", "2=dɦuɭa¦5=duri", "2=dulla",
    ]),
    60: ("gold", [
        "2=garɔ", "1=huno", "1=hono¦1=hũno", "1=hunɔ", "1=hono¦1=hũno",
        "1=hunɔ", "1=hũnɔ", "1=hano", "1=sona", "1=sona", "1=sunnu",
        "1=sonũ", "1=sone", "1=sona", "1=sona", "1=sona", "1=sona",
    ]),
    61: ("tree", [
        "1=saɖi", "1=saɖi", "1=saɖ", "1=saɽ", "1=saɖ", "1=saɖe",
        "1=saɽe", "1=dʒaɖ¦4=vik", "1=dʒaɖ", "1=dʒhaɽi", "1=dʒaɭ",
        "1=dʒaɖ", "1=dzhʌɖʌ", "1=peɖ", "5=sidʒ", "5=siɲdʒ", "3=ʌddo",
    ]),
    62: ("leaf", [
        "1=paɳ", "1=paɳ¦4=pʌɳɖho", "1=paɳ", "1=paɳ", "1=pʌɳ¦1=paɳ",
        "1=paɳ", "1=pʌne", "1=palo¦2=palo", "2=pʌʈʈa", "2=pʌʈa",
        "2=pʌʈa", "3=panɖəɖũ", "1=paɳ", "2=pətti", "1=pala¦2=pala",
        "1=pʌla¦2=pʌla", "1=pala¦2=pala",
    ]),
    63: ("root", [
        "1=mul", "1=mul", "1=mul", "1=mul", "1=mul", "1=mul", "1=mʊle",
        "1=muɭ", "3=mujanɖi", "3=mujaɖ", "1=muɭi", "1=muɭ", "1=muɭ",
        "2=dʒʌɖ", "2=dʒaɖi", "2=dʒʌri", "2=dʒari",
    ]),
    64: ("thorn", [
        "1=kaʈa", "1=kaʈʊ", "1=kaʈu", "1=kaʈa", "1=kaʈu", "1=kaʈu",
        "1=kaʈe", "1=kaʈo", "1=kaʈa", "1=kaʈo", "1=kaʈo", "1=kãʈo",
        "1=kãʈa", "1=kãʈa", "2=dʒanum", "2=dʒanum", "3=mɔr",
    ]),
    65: ("flower", [
        "1=phul", "1=phul", "1=ɸun", "1=phuɳ", "1=ɸul", "1=phun",
        "1=phuɳe", "1=ɸul", "1=phul", "1=phul", "1=phul", "1=ɸul",
        "1=ɸul", "1=ɸul", "1=phul", "1=phul", "1=phʊl",
    ]),
    66: ("fruit", [
        "1=phʌlvɔ", "1=phʌlvɔ", "1=ɸol¦1=phʌlvɔ¦5=ɸol", "1=phʌlvɔ",
        "1=ɸon¦1=phʌlvɔ", "1=phol¦6=phol", "1=phɔle¦6=phɔle",
        "1=ɸoɭ¦5=ɸoɭ", "3=phʌj¦5=phʌj", "3=phuj¦6=phuj",
        "1=phʌɭ¦5=phʌɭ", "1=ɸʌɭ¦5=ɸʌɭ", "1=ɸʌɭ¦1=ɸʌɭẽ¦5=ɸʌɭ",
        "1=phʌl¦5=phʌl", "4=dʒʌʊ", "4=dʒʌʊ", "1=phɔr¦6=phɔr",
    ]),
    67: ("mango", [
        "1=ambɔ", "1=ambo", "1=ambo", "1=ambo", "1=ambɔ¦1=ʌmbo",
        "1=ambo", "1=ɑmbɔ", "1=ambo", "1=ʌmbo", "1=ambo", "2=kiri",
        "2=kɛri", "1=ambɑ", "1=am", "1=ʌmbe", "1=ʌmbe", "3=bʌtko",
    ]),
    68: ("banana", [
        "1=kɛlɔ¦3=kɛlɔ", "1=kɛlʌ¦3=kɛlʌ", "1=kelo¦3=kelo",
        "1=kelɔ¦3=kelɔ", "1=kelo¦3=kelo", "1=kelo¦3=kelo",
        "1=kele¦1=kelɔ¦3=kele¦3=kelɔ", "1=kel¦4=kel", "2=keja¦3=keja",
        "2=kej¦4=kej", "1=kiɭu", "1=keɭu", "1=keɭe¦3=keɭe",
        "1=kela¦3=kela", "1=keɾe¦3=keɾe", "1=keɾe¦3=keɾe",
        "1=kere¦3=kere",
    ]),
    69: ("wheat", [
        "3=ghɔʔẽ¦5=ghɔʔẽ", "1=gʌũ", "1=gʌõ", "1=gɔu",
        "1=gɔvẽ¦2=gɔvẽ¦5=gɔvẽ", "2=gɔme", "2=gɔme", "1=gʌõ", "1=gɔv",
        "1=gʌu", "1=gʌu", "1=gɦəũ¦3=gɦəũ", "3=gɦʌhu¦4=gɦʌhu",
        "4=gehũ", "1=gʌʊ", "1=gʌʊ", "1=gohʊ",
    ]),
    70: ("millet", [
        "2=dʒuʌr", "2=dʒuvar", "2=zuwar", "2=dʒuvar", "2=dʒuwar",
        "2=dʒuʌr", "1=badʒʌri", "2=dzuwar", "2=dʒuvar", "2=dʒuvʌr",
        "2=dʒuvʌr", "2=dʒuari", "2=dʒuari", "2=dʒuvar", "3=orʌʊ",
        "3=orʌʊ", "3=oro",
    ]),
    71: ("rice", [
        "1=suka¦7=hal", "1=sɔka", "2=moria", "2=murijo", "2=moria",
        "1=suka", "1=sʊkhɑ¦3=kuɖiri", "1=tʃokha", "1=tʃoka", "1=tʃoka",
        "1=tʃoka", "1=tʃokɑ¦5=tʃavəl", "6=ʈanɖul", "5=tʃavəl",
        "8=tʃʌʊli", "8=tʃʌʊli", "6=tandur",
    ]),
    72: ("potato", [
        "1=bʌʈʌʈɔ", "1=bʌʈʌʈa", "1=bʌʈʌʈo", "1=boʈako", "1=bʌʈʌʈo",
        "1=bʌʈʌʈe", "1=bʌʈʌke", "2=alu", "1=bʌʈate", "1=bʌʈʌʈa",
        "1=bʌʈʌʈa", "1=bəʈaka", "1=bʌʈʌʈa", "2=alu", "2=hʌlʊ",
        "2=hʌlʊ", "2=alu",
    ]),
    73: ("eggplant", [
        "1=riŋɳɔ", "1=riŋɳɔ", "1=riŋgaɳe¦3=wenge", "1=riŋʌɳo",
        "1=riŋʌɳo", "1=eriŋgʌɳɔ¦1=riŋɳɔ", "1=riŋgʌɳe", "1=riŋgʌɳʌ",
        "4=vegʌna", "4=jegʌɳi", "3=waŋgu", "1=riŋgəɳə", "3=waŋge",
        "2=bẽigən", "1=eŋan¦3=eŋan", "1=eŋan¦3=eŋan",
        "1=eŋgan¦3=eŋgan",
    ]),
    74: ("groundnut", [
        "1=mugija", "1=mugje", "1=mũge", "5=muŋgʌrɔ", "1=mũge",
        "1=mugɔ¦5=mugɔ", "1=muŋge¦4=muŋge¦5=muŋge¦6=muŋge¦8=muŋge",
        "6=buimun-gjanɖana", "9=seŋgija", "5=çiŋgo¦8=çiŋgo",
        "5=çiŋgo¦8=çiŋgo", "2=məgɸʌɭi",
        "3=bwɦimu-gaʈʃa¦4=ʃeŋga¦5=ʃeŋga¦8=ʃeŋga¦9=ʃeŋga",
        "2=mũgɸʌli", "7=phellija", "7=phellija", "7=phʌlla",
    ]),
    75: ("chili", [
        "1=miriçɔ", "1=mirçe", "1=miriçɔ¦1=mirtse", "1=mirsa", "1=mirʃu",
        "1=mirsɔ", "1=mirise", "1=mirtʃo", "1=mirtʃe", "1=mirtʃ",
        "1=mirtʃu", "1=mərtʃũ", "1=mirtʃi", "1=mirtʃi", "1=miritʃa",
        "1=miritʃa", "1=miritʃa",
    ]),
    76: ("turmeric", [
        "1=elɖɔ¦5=elɖɔ", "1=elɖɔ¦5=elɖɔ", "1=elɖʌ¦5=elɖʌ",
        "1=elɖo¦5=elɖo", "1=elɖo¦5=elɖo", "1=elɖɔ¦5=elɖɔ",
        "1=elɖɔ¦5=elɖɔ", "1=eɭiɖ", "3=hʌiɖ", "2=ajjɖ",
        "1=hʌɭaɖi¦5=hʌɭaɖi", "1=həɭɖə¦5=həɭɖə", "1=hʌɭʌɖ",
        "1=hʌlɖi", "4=tʃasan", "4=sʌsan", "5=hʌrdo",
    ]),
    77: ("garlic", [
        "1=lɔhʌɳɔ", "1=lɔhʌɳɔ", "1=nohoɳo", "1=nohɳɔ", "1=nohoɳɔ",
        "1=lɔhʌɳɔ", "1=nɔhɔɳɔ", "1=ɭeheɳ", "1=lʌsin", "1=lʌsʌn",
        "1=lʌsʌn", "1=lʌsʌɳ", "1=lʌsun", "1=lʌhəsʊn¦1=lʌsʊn",
        "1=loson", "1=lʊsʊn", "1=lusun",
    ]),
    78: ("onion", [
        "2=kanɖo", "2=kanɖʊ", "2=kanɖu", "2=kanɖo", "2=kanɖu¦2=kanɖa",
        "1=ɖugli¦2=kanɖa", "2=kãɖa", "2=kanɖa", "2=kanɖo", "2=kanɖo",
        "2=kanɖo", "1=ɖuŋgəɭi", "2=kanɖa", "3=pjadʒ", "2=kande",
        "2=kande", "2=kande",
    ]),
    79: ("cauliflower", [
        "3=gɔbi", "3=gobi", "3=gobi", "3=kobi", "3=gobi", "3=gobi",
        "2=phʊlʊvʌre", "3=gobi", "3=kobi", "1=phul",
        "1=phul gobi¦3=phul gobi", "2=ɸlauwər",
        "1=ɸulkobi¦2=ɸlawər¦3=ɸulkobi¦3=gobi",
        "1=ɸulgobɦi¦1=phul gobɦi¦3=ɸulgobɦi¦3=phul gobɦi",
        "3=gobi", "3=gobi", "1=phul gobi¦3=phul gobi",
    ]),
    80: ("tomato", [
        "1=ɖɔmaʈʌr", "1=ʈʌmaʈʌr", "2=duʔune", "1=ʈʌmaʈo",
        "1=ʈʌmaʈʌr¦2=duʔule", "1=ʈʌmiʈo", "1=ʈamite", "4=irariŋʌɳo",
        "1=ʈʌmaʈe", "1=ʈʌmaʈa", "1=ʈʌmaʈʌr", "1=ʈomeʈo", "1=ʈʌmaʈe",
        "1=ʈʌmaʈʌr", "1=ʈʌmaʈʌr", "5=dʒirimiri", "3=bɦɛɖʌra",
    ]),
    81: ("cabbage", [
        "1=gadagopi", "1=gadagopi", "1=gobi", "1=gʌɖɖakobi",
        "1=gobi¦1=gʌɖakobi", "1=guʈagobi", "1=gobi", "1=paŋgobi",
        "1=gʌɖɖakobi", "1=kobi", "1=gadagopi", "1=kɔbidʒ",
        "1=gobi¦1=kobi", "1=gobɦi", "1=pala gobi", "1=pala gobi", "1=gobi",
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
        55: 44, 56: 44, 57: 45, 58: 45, 59: 45, 60: 45,
        61: 46, 62: 46, 63: 46, 64: 46, 65: 46, 66: 46,
        67: 47, 68: 47, 69: 47, 70: 48, 71: 48, 72: 48,
        73: 48, 74: 48, 75: 49, 76: 49, 77: 49, 78: 49,
        79: 49, 80: 50, 81: 50,
    }[item]
    if item == 56 and site_index >= 4:
        page = 45
    elif item == 60 and site_index >= 15:
        page = 46
    elif item == 66 and site_index >= 2:
        page = 47
    elif item == 69 and site_index >= 15:
        page = 48
    elif item == 74 and site_index >= 12:
        page = 49
    elif item == 79 and site_index >= 7:
        page = 50

    left_current_page = {
        55: set(), 56: set(), 57: set(range(17)), 58: set(range(6)),
        59: set(), 60: set(), 61: set(range(17)), 62: set(range(17)),
        63: set(range(5)), 64: set(), 65: set(), 66: set(),
        67: set(range(17)), 68: set(range(1)), 69: set(),
        70: set(range(17)), 71: set(range(17)), 72: set(range(9)),
        73: set(), 74: set(), 75: set(range(17)), 76: set(range(13)),
        77: set(), 78: set(), 79: set(), 80: set(range(17)),
        81: set(range(12)),
    }
    continuation_left = (
        (item == 56 and page == 45)
        or (item == 60 and page == 46)
        or (item == 66 and page == 47)
        or (item == 69 and page == 48)
        or (item == 74 and page == 49)
        or (item == 79 and page == 50)
    )
    column = "left" if continuation_left or site_index in left_current_page[item] else "right"
    return str(page), str(page - 6), column


def parse_cell(cell: str) -> tuple[str, str, str, str, str]:
    pairs = [part.split("=", 1) for part in cell.split("¦")]
    labels = " | ".join(label for label, _ in pairs)
    forms = " | ".join(form for _, form in pairs)
    return forms, labels, "attested", "high", ""


def rows() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    assert set(ITEMS) == set(range(55, 82))
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
    rows_to_write = rows()
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_to_write)
    print(f"wrote {len(rows_to_write)} OCR-blind manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
