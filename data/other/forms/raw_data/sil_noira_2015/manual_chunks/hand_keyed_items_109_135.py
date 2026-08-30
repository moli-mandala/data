#!/usr/bin/env python3
"""Emit the Noira 2015 items 109–135 OCR-blind manual-review ledger.

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
OUT = Path(__file__).with_name("items_109_135_hand_keyed.tsv")

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
# responses in one site cell are separated by ``¦``. ``BLANK`` means that the
# source explicitly prints ``0 no entry``. The order is SITES above.
ITEMS = {
    109: ("older sister", [
        "4=ɖʌji bɔʔhĩ", "3=oɖu bɔʔhi", "3=woɖiboʔhĩ", "3=voɖi boihʔi",
        "3=aoɖi boʔhĩ¦4=ɖʌji boʔhĩ", "3=oɖi bɔʔhĩ¦4=ɖʌiru bɔʔhĩ",
        "6=baje", "6=ɖawali bai¦8=ɖawali bai¦8=ɖawali boɳli",
        "7=moʈi bʌhʌis", "6=moʈi baj¦7=moʈi baj", "7=moʈi ben",
        "7=moʈiben", "3=woɖibʌ-hiɳ", "1=ɖiɖi¦2=bəɖibəhin",
        "6=bʌi¦7=bʌi", "9=bokodʒʌi", "6=baji",
    ]),
    110: ("younger sister", [
        "4=aiʈi bɔʔhĩ", "4=aiʈi bɔʔhi", "2=haɳiboʔhĩ¦4=haɳiboʔhĩ",
        "4=aiʈi boihʔi", "2=haɳi boʔhĩ¦4=haɳi boʔhĩ", "4=ʌiʈi bɔʔhĩ",
        "4=ajiʈibɔjiʔ-hi", "1=aʈʌlio bai¦1=aʈʌlio boɳli",
        "3=nʌhʌi bʌhʌis", "2=nani baj¦7=nani baj", "2=nʌnɖiɖi ben",
        "2=naniben¦8=naniben", "3=lahan-bhʌhin¦8=lahan-bhʌhin",
        "2=tʃoʈibəhin¦8=tʃoʈibəhin", "6=boko", "6=bokodʒʌi", "7=baji",
    ]),
    111: ("son", [
        "1=sɔʔu", "1=sɔʔu", "1=sɔʔu¦3=poriu",
        "1=suʔa¦3=pouʔija¦8=pouʔija", "1=soʔo¦3=poiu", "1=suʔu", "1=swʊ",
        "1=ʂero", "3=porija¦8=porija", "3=porija¦8=porija", "4=tʃokiri",
        "2=puʈəra¦4=ɖikəro¦4=tʃokro", "2=puʈrʌ¦6=mulga", "5=beʈa",
        "7=ɖokekoɳ", "7=koɳ", "8=nʌnʌ",
    ]),
    112: ("daughter", [
        "1=sɔʔi", "1=sɔʔi", "1=sɔʔi¦7=pori¦10=pori",
        "1=suʔi¦1=sui¦3=poiri¦7=poiri", "1=sɔʔi¦7=pori¦10=pori",
        "1=suʔi", "1=sʊi", "1=ʂori¦10=ʂori", "7=poʈi", "7=poʈi",
        "2=tʃokiri", "2=ɖikəri¦2=tʃokri¦3=puʈəri", "5=kʌnju¦6=mulgi",
        "3=puʈri¦4=beʈi", "9=koɳ", "9=kondʒʌi", "8=piridʒo",
    ]),
    113: ("husband", [
        "4=ɖɔhu", "1=maʈi¦7=koualu", "1=maʈi¦4=ɖɔʔu", "1=maʈi",
        "1=maʈi¦4=ɖɔhu", "4=ɖuhu", "1=maʈi", "10=eɖmi", "1=maʈi",
        "1=maʈi", "5=ghrwʌlo", "5=ɖɦʌɳi", "2=puʈi¦3=mulgi",
        "1=pəʈi¦2=pəʈi", "6=sana", "6=sana¦9=ɖoʈa", "8=atho",
    ]),
    114: ("wife", [
        "1=naɖi", "1=laɖi¦1=malaɖi", "4=bojõ", "4=buijo",
        "1=laɖi¦4=bojõ", "1=naɖi", "1=naɽi", "4=baiko", "8=t̪her",
        "4=bʌiko", "5=ghrwʌli", "2=pəʈni", "2=pʌʈni¦4=baiko",
        "2=pəʈni", "6=dʒapa", "6=dʒapaj", "9=kɔl",
    ]),
    115: ("boy", [
        "6=sɔʔu", "6=soʔu", "1=poriu¦6=sɔʔu", "1=porija¦6=suʔu",
        "1=poriu", "6=suʔu", "BLANK", "6=ʂoro", "1=porija", "1=porija",
        "2=tʃokaro", "2=tʃokro", "1=purga¦4=mulga", "3=ləɖka",
        "1=porija", "1=porija", "7=nʌnʌ",
    ]),
    116: ("girl", [
        "5=sɔʔi", "5=soʔi", "1=pori¦5=sɔʔi", "1=poiri¦5=suʔi¦5=sui",
        "1=pori", "5=suʔi", "BLANK", "5=ʂori", "1=poʈi", "1=porija",
        "2=tʃokari", "2=tʃokri", "1=porgi¦4=mulgi", "3=ləɖki",
        "7=t̪ʌrʌi", "7=t̪ʌrʌj", "6=piridʒo",
    ]),
    117: ("day", [
        "1=ɖihi", "1=ɖihi", "1=ɖihi", "1=ɖihi", "1=ɖihi", "1=ɖihi",
        "1=ɖihi", "1=ɖih¦5=ɖih", "1=ɖisa¦4=ɖisa", "5=ɖin", "3=ɖaɖo",
        "2=ɖivəs", "2=ɖiwʌs", "2=ɖivəs¦5=ɖin", "4=ɖija", "4=ɖija",
        "4=ɖina",
    ]),
    118: ("night", [
        "1=raʈ", "1=raʈ", "1=raʈ", "1=raʈ", "1=raʈ", "1=raʈe", "1=raʈe",
        "1=raʈ", "1=raʈ", "1=raʈ", "1=raʈ", "1=raʈ¦1=raʈri", "1=raʈra",
        "1=raʈ", "1=raʈo", "1=raʈo", "3=mindi",
    ]),
    119: ("morning", [
        "1=vehi", "1=vegi", "1=wegi", "1=vegidz", "1=vegi", "6=raʈi",
        "5=həɖãre¦6=raʈidʒe", "9=uʈjain", "2=sʌkʌi", "2=sʌkaj",
        "2=sʌkkale", "5=səvər", "2=sʌkaɭ", "3=səbera¦3=subeh¦5=səbera",
        "7=phidʒʌin¦8=ʌɲʌn", "7=phedʒʌr", "7=phɛdʒer",
    ]),
    120: ("noon", [
        "4=hirʌupe¦6=hirʌupe¦8=hirʌupe", "1=ɖihe¦4=hirape¦6=hirape",
        "1=ɖihɛ", "1=madzon-ɖihi", "1=ɖihɛ¦4=hirʌo¦8=hirʌo",
        "1=ɖihɔ¦8=hiraɳupe", "1=ɖihɔ", "1=boɽɖih", "5=mʌdjan",
        "5=mʌdhijan", "5=mʌɖijan", "3=bəpor", "3=ɖupʌr", "3=ɖopəhər",
        "7=ʌjipdʒʌn", "1=ɖija", "6=baripar",
    ]),
    121: ("evening/afternoon", [
        "1=hʌsʈivelli", "1=hasʈivelle", "1=hastiweɭe", "1=hʌsʈivello",
        "1=hʌsʈivell¦7=hãndze_po", "1=hʌsʈivell¦1=hʌsʈɔ", "1=hãsʈɔ",
        "11=weɭʈo¦12=weɭwar", "6=ɖimʌi", "2=ʌnɖija kʌj",
        "2=sʌnɖija kʌl", "5=sãdʒ", "2=saijʌnkʌɭ¦2=sʌnɖhja",
        "2=sõɖhja¦4=ʃam", "9=sikurup-dʒʌin", "10=ʌjubdʒen", "8=budo",
    ]),
    122: ("yesterday", [
        "1=kal", "1=kal", "1=kal", "1=kan", "1=kal", "1=kan", "1=kane",
        "1=kal", "4=kʌlɖi", "4=kalɖi", "1=kale", "1=kale¦3=gəikal",
        "1=kal", "1=kʌl", "4=kolɖin", "4=koldin", "5=çe",
    ]),
    123: ("today", [
        "1=adʒ", "1=adʒ", "1=adz", "1=adʒ", "1=adz", "1=adʒe", "1=adʒe",
        "1=aɖz", "1=adʒ", "1=adʒ", "1=adʒ", "1=adʒei", "1=adz",
        "1=adʒ", "3=ʈein", "3=ʈejndʒ", "2=baj",
    ]),
    124: ("tomorrow", [
        "1=hanɖa", "1=hanɖa", "1=handa", "1=handa", "1=handa", "1=hanɖa",
        "1=hãɖʌʔa", "5=hekeɭe", "4=kalɖi", "4=kalɖi", "2=kale", "2=kale",
        "3=uɖɦja", "2=kal", "7=paʈa", "7=paʈa", "6=kjam",
    ]),
    125: ("week", [
        "1=aʈhi¦2=hʌpʈi", "1=aʈh¦2=hʌpʈu", "1=aʈe¦1=aʈo", "1=aʈouɖo",
        "1=aʈ¦2=hʌpʈo", "1=aʈhiwaɖi", "1=aʈvarijʊ", "2=apʈo", "2=hʌpʈa",
        "1=aʈhoɖo", "3=saʈɖaɖa", "1=aʈhawaɖiũ¦2=səpʈah",
        "1=aʈhwʌɖa", "2=hʌphʈa¦2=səpʈah", "4=huʈi", "1=aʈdin", "2=hʌpʈa",
    ]),
    126: ("month", [
        "1=moinu", "1=moinɔ", "1=moinu", "1=moino", "1=moinu", "1=moinu",
        "1=mɔjinʊ", "1=mʌinɔ", "1=mʌhina", "1=mʌhino", "1=mʌhino",
        "1=mãhino", "1=mʌihina", "1=məhina", "1=mena", "1=mena", "1=mʌina",
    ]),
    127: ("year", [
        "1=orihi", "1=worihi", "1=wʌrih", "1=vʌrho", "1=worihi",
        "1=orihi¦1=worehe", "1=vɔrɔhe¦1=vɔrɔhɔ", "1=worih",
        "4=bʌrʌmʌ-hina", "3=sal", "3=sal", "2=vərʃ", "2=wʌrʂʌ¦3=sal",
        "2=vərʂ¦3=sal", "3=sal", "3=sal", "3=sal",
    ]),
    128: ("old", [
        "1=dʒʊnɔ", "1=dʒʊnɔ", "1=dzunʌ", "1=dʒuno", "1=dʒuna¦1=dʒunu",
        "1=dʒunno", "1=dʒunɔ", "1=ɖzunalo", "3=porɖi", "1=dʒuna",
        "1=dʒunu", "1=dʒunu", "1=dzunə", "2=purana", "1=dʒona",
        "1=dʒuna", "1=dʒuna",
    ]),
    129: ("new", [
        "1=nʌvɔ¦2=nʌvɔ¦4=nʌvɔ", "1=nʌvɔ¦2=nʌvɔ¦4=nʌvɔ",
        "1=naowa¦4=naowa", "1=nʌvo¦2=nʌvo¦4=nʌvo", "1=nowu", "3=nʌnno",
        "1=nʌvʊnɔ¦3=nʌvʊnɔ", "1=nʌwalo", "1=nʌva¦2=nʌva¦4=nʌva",
        "1=nʌva¦2=nʌva¦4=nʌva", "1=nou", "1=nəvũ¦4=nəvũ",
        "1=nʌwa¦2=nʌwa¦4=nʌwa", "1=nəvə¦2=nəja¦2=nəvə¦4=nəvə",
        "5=une", "5=une", "2=nahʊa¦4=nahʊa",
    ]),
    130: ("good", [
        "1=hadzo", "1=hadzo", "1=hadzo", "1=nʌdzo", "1=hadzu", "1=hadʒo",
        "1=hadʒɔ", "1=hadzo", "7=ovʌlʌs", "6=avɭ", "2=saru", "2=sarəs",
        "5=tsaŋgla", "3=ʌtʃtʃha¦4=bəɽhija", "3=atʃha", "8=aʊlka", "9=masʈo",
    ]),
    131: ("bad", [
        "2=kʌini hadʒo", "2=kʌini-hadzo", "2=kʌini hadʒo¦2=nehadzu",
        "2=nʌhʌdzo¦7=komi", "2=kʌini hʌdʒo¦2=ɳahadzu", "2=hadʒonʌi",
        "1=khʌrabe", "1=khʌrap", "3=kaninao-vʌlʌs¦4=sʌɖigʌja¦4=pidigʌja",
        "1=khʌrab", "1=khʌrab", "1=khərab", "1=khʌrab¦6=wait", "1=khərab",
        "8=surʌi", "8=surʌi", "9=bekar",
    ]),
    132: ("wet", [
        "1=piginɔ", "1=pigilo", "1=pignɔ", "1=pigino", "1=pigli",
        "1=piginɔ¦4=ʈiʈino", "1=pignɔ", "6=bidʒel", "2=nija", "2=nija",
        "3=lillu", "8=bɦinĩ¦9=pəɭa-rəvũ", "5=ola¦6=bhʌdʒlela", "5=gila",
        "5=ola", "5=ola", "5=ola",
    ]),
    133: ("dry", [
        "1=ugʌinɔ", "1=ugʌino", "1=uganɔ", "1=ugʌino", "1=ugalu",
        "1=ugʌinɔ", "1=ʊgajinɔ", "2=hukel", "2=sukʌila",
        "2=sukhʌl¦4=kojaɖa", "3=koɖu", "2=sukũ¦3=korũ",
        "2=sukha¦3=koɖa¦4=koɖa", "2=sukha", "5=lokor", "5=lokor", "6=pʌʈar",
    ]),
    134: ("long", [
        "1=lambɔ", "1=lambo", "1=nambɔ", "1=nambo", "1=lambo¦1=nambi",
        "1=nambɔ", "1=nambɔ", "1=lambo", "1=lʌmba", "1=lamba", "1=lambu",
        "1=lambũ", "1=lambʌ", "1=ləmbu", "1=lʌmba", "1=lʌmba", "1=lʌmba",
    ]),
    135: ("short", [
        "1=tukɔ", "7=aiʈo", "1=ʈokɔ¦12=hano¦8=hano", "7=aiʈo", "1=ʈoko",
        "1=tukɖo", "1=tʊkʌrɔ", "10=ʈiljo", "12=nʌhna¦4=nʌhna",
        "12=nana¦4=nana", "5=hʌnɖaɖu", "1=ʈũkũ",
        "2=tʃhoʈa¦3=akhuɖ¦4=lahan", "2=tʃhoʈa", "8=sʌnika", "8=sʌni",
        "9=bʌtʃka",
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
        109: 56, 110: 56, 111: 57, 112: 57, 113: 57, 114: 57,
        115: 58, 116: 58, 117: 58, 118: 58, 119: 58, 120: 59,
        121: 59, 122: 59, 123: 59, 124: 59, 125: 60, 126: 60,
        127: 60, 128: 60, 129: 60, 130: 61, 131: 61, 132: 61,
        133: 61, 134: 62, 135: 62,
    }[item]
    if item == 110 and site_index >= 15:
        page = 57
    elif item == 119 and site_index >= 15:
        page = 59
    elif item == 124 and site_index >= 12:
        page = 60
    elif item == 129 and site_index >= 6:
        page = 61

    left_sites = {
        109: set(range(5)), 110: set(), 111: set(range(17)),
        112: set(range(12)), 113: set(), 114: set(), 115: set(range(17)),
        116: set(range(17)), 117: set(range(7)), 118: set(), 119: set(),
        120: set(range(17)), 121: set(range(16)), 122: set(), 123: set(),
        124: set(), 125: set(range(17)), 126: set(range(17)),
        127: set(range(4)), 128: set(), 129: set(), 130: set(range(17)),
        131: set(range(4)), 132: set(), 133: set(range(9)),
        134: set(range(17)), 135: set(range(14)),
    }
    continuation_left = (
        (item == 110 and page == 57)
        or (item == 119 and page == 59)
        or (item == 124 and page == 60)
        or (item == 129 and page == 61)
    )
    column = "left" if continuation_left or site_index in left_sites[item] else "right"
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
    assert set(ITEMS) == set(range(109, 136))
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
