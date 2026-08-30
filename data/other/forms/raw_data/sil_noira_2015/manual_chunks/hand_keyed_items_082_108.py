#!/usr/bin/env python3
"""Emit the Noira 2015 items 82–108 OCR-blind manual-review ledger.

Every value in ``ITEMS`` below was independently keyed while viewing the
400-dpi rendered source pages (with selected difficult cells re-rendered at
800 dpi). This module does not read PDF text, OCR, scaffold files, or another
transcription at runtime.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
REVIEWED_AT = "2026-08-28"
OUT = Path(__file__).with_name("items_082_108_hand_keyed.tsv")

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
    82: ("oil", [
        "1=ʈel", "1=ʈel", "1=ʈen", "1=ʈen", "1=ʈel", "1=tene",
        "1=ʈene", "1=ʈeɭ", "1=ʈel", "1=ʈel", "1=ʈel", "1=ʈɛl",
        "1=ʈel", "1=ʈɛl¦1=ʈel", "2=sʊnum", "2=sʊnum", "1=ʈel",
    ]),
    83: ("salt", [
        "1=kharɔ", "1=kharo", "1=kharo", "1=kharo", "1=kharo",
        "1=kharo", "1=kharɔ", "2=miʈh", "1=khara", "2=miʈ",
        "2=miʈhu", "2=miʈhũ", "2=miʈh", "3=nʌmək", "4=bʊlʊm",
        "4=bʊlʊm", "5=tʃopo",
    ]),
    84: ("meat", [
        "1=mʌha", "1=maha", "1=maha", "1=maha", "1=mãhã", "1=mʌha",
        "1=mʌhã", "1=mah", "4=badʒi", "3=mas", "3=mas", "2=gos¦3=gos",
        "1=mãs¦3=mãs", "2=gosʈ", "5=gilʊ", "5=dʒilu", "6=kev",
    ]),
    85: ("fat", [
        "2=sʌrbi", "2=sʌrbi", "1=doɖo", "1=ɖoɖɔ", "1=doɖo¦2=sʌrbi",
        "1=ɖɔɖɔ", "1=ɖɔrɔ", "2=tʃerbi", "4=ʈadʒa", "2=tʃʌrbi",
        "2=tʃʌrbi", "2=tʃərbi", "2=tʃʌrəbi", "2=tʃərbi", "2=tʃerbi",
        "2=tʃerbi", "3=tem",
    ]),
    86: ("fish", [
        "1=masɔ", "1=masɔ", "1=masʊ", "1=maso", "1=masʊ", "1=masɔ",
        "1=mase¦1=masɔ", "1=maso", "1=mase", "1=masa", "2=matʃʌlʊ",
        "2=matʃəli", "1=masa¦2=matʃja", "2=mʌtʃhli", "4=kaku", "4=kaku",
        "3=tʃan",
    ]),
    87: ("chicken", [
        "1=kukɖɔ", "1=kukɖa¦1=kukɖi", "1=kukʌɖi", "1=kʊkʊɖi",
        "1=kukʌɖi", "1=kukɖo", "1=kʊkɽe¦1=kʊkɽi", "1=kukʌɖo",
        "1=kukɖi", "1=kukiɖi", "1=koŋʌɽi¦4=koŋʌɽi", "1=kukuɖi",
        "4=kombʌɖi", "3=mʊrgi", "2=çim", "2=çim", "1=kokʌr",
    ]),
    88: ("egg", [
        "1=inɖɔ", "1=inɖɔ", "1=inɖu", "1=inɖe", "1=inɖo",
        "1=ĩɖo¦1=inɖɔ", "1=ĩnɖɔ", "2=hakõ", "1=anɖa", "1=anɖa",
        "1=anɖa", "1=iɳɖũ", "1=ʌnɖe", "1=ʌnɖa", "2=ʌkkʌm",
        "2=ʌkkom", "3=kʌlen",
    ]),
    89: ("cow", [
        "2=gauɖi", "1=vasʌɖi¦2=gʌuɖi", "1=vasʌɖi¦2=gauɖi",
        "1=vasʌɖi¦2=gauɖi", "1=vasɖi¦2=gauɖi", "1=vasʌɖi",
        "1=vasɔrɔ¦1=vasʌɽi", "2=gai", "2=gavʌɖi", "2=gaj", "2=gaj",
        "2=gai", "2=gai", "2=gai", "2=gʌj", "2=gʌi", "3=dhor",
    ]),
    90: ("buffalo", [
        "1=paɖi", "1=paɖo¦5=paɖo", "1=paɖi", "1=paɖi", "1=paɖi",
        "1=paɖi", "1=paɽʔi", "6=ɖobo", "3=ɖobʌɖ¦6=ɖobʌɖ",
        "3=ɖobʌɖ¦6=ɖobʌɖ", "3=ɖobʌɽi", "2=bɦẽs", "2=mɦʌis",
        "2=bɦẽs", "4=bekkel", "4=bekkel", "5=odo",
    ]),
    91: ("milk", [
        "1=ɖuɖ", "1=ɖuɖ", "1=duɖ", "1=ɖuɖ", "1=ɖuɖ", "1=ɖuɖe",
        "1=ɖuɖe", "1=ɖuɖ", "1=ɖuɖ", "1=ɖuɖ", "1=ɖuɖ", "1=ɖuɖɦ",
        "1=ɖuɖɦ", "1=ɖuɖɦ", "3=ɖiɖʌm", "2=ɖiɖom¦3=ɖiɖom", "2=dudo",
    ]),
    92: ("horns", [
        "1=hiŋgʌɖɔ", "1=hiŋgʌɖɔ", "1=hiŋ", "1=hiŋg",
        "1=hiŋg¦1=hiŋgʌɖo", "1=hiŋge", "1=hiŋge", "1=hiŋg",
        "1=seŋgijo", "1=seŋgɖa", "1=çiŋga", "1=ʃiŋgəɖa", "1=siŋg",
        "1=siŋ", "1=siŋgi", "1=siŋgi", "1=siŋgi",
    ]),
    93: ("tail", [
        "4=sepɖɔ¦6=sepɖɔ", "4=sempʈi¦6=sempʈi", "4=ʂepʈi¦6=ʂepʈi",
        "4=sepʈo¦6=sepʈo", "4=sepʈo¦6=sepʈo", "4=semʈo", "1=pesʈɔ",
        "4=ʂemʈo", "4=sapʈa¦6=sapʈa", "5=tʃepa", "3=çepʈi¦4=çepʈi",
        "2=pũtʃʌɖi", "3=ʃepuʈ¦4=ʂelpʈi¦6=ʂelpʈi", "2=pũtʃh",
        "8=tʃu", "8=tʃu", "7=pago",
    ]),
    94: ("goat", [
        "1=bukɖɔ¦3=bukɖɔ", "1=bukuɽi¦2=bukuɽi¦3=bukuɽi",
        "1=bokʌɖi¦2=bokʌɖi¦3=bokʌɖi", "1=bukuɽi¦2=bukuɽi¦3=bukuɽi",
        "1=bokʌɖi¦2=bokʌɖi", "1=bukʌɖo¦2=bukʌɖo¦3=bukʌɖo",
        "1=bʊkhɽɔ¦2=bʊkhɽɔ¦3=bʊkhɽɔ", "1=bokʌɖi¦2=bokʌɖi",
        "1=bʌkʌri¦2=bʌkʌri", "1=bʌkʌri¦2=bʌkʌri",
        "1=bʌkʌri¦2=bʌkʌri", "1=bəkro¦2=bəkeri¦3=bəkro",
        "1=bʌkʌri¦2=bʌkʌri", "2=bəkeri", "4=çiri", "4=çiri", "4=çeri",
    ]),
    95: ("dog", [
        "1=kuʈro¦2=huɳɔ", "1=kuʈro", "2=hũɳi¦2=hũɳɔ", "2=hũɳo",
        "2=hũɳi¦2=hũɳɔ", "1=kuʈro", "1=kʊʈrɔ", "3=tʃiʈõ", "1=kuʈra",
        "1=kuʈra", "1=kuʈra", "1=kuʈər", "1=kuʈra", "1=kuʈʈa",
        "5=çiʈa", "5=çiʈa", "4=naj",
    ]),
    96: ("snake", [
        "7=goɖihu", "7=gorʌhu", "8=ivɔ", "1=hap¦7=goɖʊhu", "1=hap",
        "7=goɖoho", "2=kape", "6=geɖe", "3=sapɖa", "3=sapɽa", "1=sap",
        "1=sãp", "1=sap", "1=sãp", "4=bidʒ", "4=biɲ", "5=kogo",
    ]),
    97: ("monkey", [
        "1=makʌɖi", "1=makʌɖi", "1=makoɖ¦2=bhodʒijo", "1=makoɽ",
        "1=makoɖ", "1=makoɖe", "1=makɔɽe", "1=makoɖ",
        "3=vanɖra¦4=vanɖra", "3=waɖra", "3=wanɖrʊ¦4=wanɖrʊ",
        "3=vənɖərũ¦4=vənɖərũ", "3=wanʌr¦4=wanʌr", "4=bənɖər",
        "5=sara", "4=bʌnɖʌri¦5=sara", "6=tʃarko",
    ]),
    98: ("mosquito", [
        "1=mɔghɔ", "7=simiɳi", "1=mogɦe¦3=daɦẽ¦3=hãɦẽ",
        "1=mogho¦4=mokʈur", "1=mɔgɦe¦3=daɦẽ", "1=mogho¦4=mokʈo",
        "1=mɔgɦe", "10=tʃatʃʌɖia", "4=mʌkʈura", "2=mʌtʃrija",
        "2=mʌtʃʌrija", "2=mətʃhərə", "2=mʌtʃhʌr", "2=mət̪tʃhər",
        "11=tʃikini", "8=domdom¦11=tʃikini", "9=ɛdəgo",
    ]),
    99: ("ant", [
        "1=kivaɖɔ", "1=kiɖavi", "2=kiɖo", "2=kiɖo",
        "1=kiɖavo¦1=kiɖawi¦5=kiɖavo", "1=kiɖavo¦1=kivaɖɔ¦5=kiɖavo",
        "1=kivaɽɔ", "1=kiɖawõ¦5=kiɖawõ", "2=kiɖi", "2=kiɖjo¦5=kiɖjo",
        "2=kiɖi", "2=kiɖi", "3=mũŋgi", "4=tʃĩʈi", "4=tʃaʈi",
        "4=tʃaʈi", "6=kokʌi",
    ]),
    100: ("spider", [
        "1=bɔɖkiljɔ", "4=mukuru", "2=huʈaɖo", "9=boʈlia", "1=boʈkil",
        "1=boɖkulijo", "1=bɔɽkilijʊ", "2=huʈaɖo", "7=siʈaɖo",
        "8=gekɽa", "3=koɭi", "5=kəroɖio", "3=koɭi", "4=mʌkʌɖi",
        "6=dʒagli-malja", "6=dʒʌgʌ-limʌlʌj", "6=dʒʌgʌ-limalai",
    ]),
    101: ("name", [
        "1=nav", "1=nau", "1=nau", "1=nam", "1=nau", "1=nam", "1=name",
        "1=nau", "1=nam", "1=nav", "1=nam", "1=nam", "1=naw", "1=nam",
        "2=dʒʊmʊ", "2=dʒʊmʊ", "2=dʒʊmʊ",
    ]),
    102: ("man", [
        "4=maʈi", "4=maʈi", "4=maʈi", "4=maʈi", "4=maʈi", "4=maʈi",
        "5=maɦũʔʊ", "1=eɖmi", "7=t̪holja", "2=mʌnus", "2=mʌnus",
        "6=mʌnəs", "2=manuʃ¦3=purʊʂ", "1=aɖmi¦2=mənusjə¦3=purʊʂ",
        "8=ɖota", "8=ɖota", "2=manso¦4=aʈo¦8=aʈo",
    ]),
    103: ("woman", [
        "4=buʔĩ", "3=buĩ¦4=buʔĩ", "4=bojõ", "4=bojõ", "4=bojõ¦4=buʔĩ",
        "4=buʔĩ", "4=bʊjɛʔẽ", "3=baiku", "5=t̪her", "3=baj", "3=bʌi",
        "1=st̪ri", "1=st̪ri¦3=bai", "1=st̪ri¦2=aurət", "7=dʒʌpaj",
        "7=dʒʌpaj", "8=kɔl",
    ]),
    104: ("child", [
        "6=sɔʔɔ", "6=soʔe", "6=sɔʔɔ¦7=pɔrijɔ¦7=pori", "1=poiro¦7=poiro",
        "1=porijo¦6=soʔu", "1=puiro", "1=pʊjirɔ", "8=ʂero", "1=porija",
        "1=porija", "1=porijo", "2=tʃokrũ", "4=mul¦5=lekɦru", "3=bətʃtʃa",
        "10=kaʈʌria", "10=kaʈʌrja", "9=nʌnʌata",
    ]),
    105: ("father", [
        "1=bʌɦʔu", "1=bʌɦʔu", "1=bahku", "1=bʌɦʔu",
        "1=bahku¦1=bʌɦʔʊ", "1=bʌɦʔu", "1=bʌɦʔʊ", "4=aboxkhʌ",
        "3=abo", "2=babo", "2=bap", "2=bapə¦5=piʈa", "2=bap¦5=piʈa",
        "2=bap¦5=piʈa", "2=baʈa", "2=baʈe", "2=ba",
    ]),
    106: ("mother", [
        "1=jʌʔhi", "1=jʌhʔi", "1=jʌʔhi", "1=jʌʔhi¦1=jʌhʔi",
        "1=aihi¦1=jʌhʔi", "1=jʌhʔi", "1=jʌʔhi", "4=ax", "2=ma",
        "2=maj¦3=maj", "2=ma", "2=ma", "2=maʈa¦3=ai", "2=ma¦2=maʈa",
        "2=maj", "2=maj", "2=ma",
    ]),
    107: ("older brother", [
        "4=oɖupahi¦5=ɖʌju pahi", "4=oɖupʌhi¦5=ɖau pʌhi",
        "4=waɖu-pauhu¦5=ɖʌju pahi¦5=waɖupau-hu", "4=voɖu pʌuhʔi",
        "4=woɖu-pahi", "4=oɖu-pahju¦5=ɖʌiru pahju", "3=ɖaɖʊ",
        "8=ɖawalo baxkhʌ", "6=bʌɖo baʊs", "2=moʈo bav", "2=moʈo baj",
        "2=moʈobɦai", "2=moʈhbɦau", "1=bəɖəbɦai", "7=kʌɖaji",
        "7=ɖaji", "3=dada",
    ]),
    108: ("younger brother", [
        "1=aiʈupahi", "1=aiʈu pahi", "2=hanu-pauhu", "1=aiʈu pʌuhʔu",
        "2=hanupahi", "1=ʌiʈu pahju", "1=ajiʈupʌjhʊ", "7=aʈʌlio baxkhʌ",
        "2=nʌhno baʊs", "2=nanubav", "6=nʌnɖɖo bɦaj", "3=nanobɦai",
        "5=lahan-bʌhin", "4=tʃhoʈəb-ɦai", "10=ɖai", "8=boko",
        "9=bʌtʃkʌ-dada",
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
        82: 50, 83: 50, 84: 50, 85: 51, 86: 51, 87: 51, 88: 51,
        89: 51, 90: 52, 91: 52, 92: 52, 93: 52, 94: 52, 95: 53,
        96: 53, 97: 53, 98: 53, 99: 54, 100: 54, 101: 54, 102: 54,
        103: 55, 104: 55, 105: 55, 106: 55, 107: 55, 108: 56,
    }[item]
    if item == 84 and site_index >= 7:
        page = 51
    elif item == 89 and site_index >= 6:
        page = 52
    elif item == 94 and site_index >= 2:
        page = 53
    elif item == 98 and site_index >= 4:
        page = 54
    elif item == 107 and site_index >= 5:
        page = 56

    left_current_page = {
        82: set(), 83: set(), 84: set(), 85: set(range(17)),
        86: set(range(17)), 87: set(), 88: set(), 89: set(),
        90: set(range(17)), 91: set(range(17)), 92: set(), 93: set(),
        94: set(), 95: set(range(17)), 96: set(), 97: set(), 98: set(),
        99: set(range(17)), 100: set(range(17)), 101: set(), 102: set(),
        103: set(range(17)), 104: set(range(17)), 105: set(range(5)),
        106: set(), 107: set(), 108: set(range(17)),
    }
    continuation_left = (
        (item == 84 and page == 51)
        or (item == 89 and page == 52)
        or (item == 94 and page == 53)
        or (item == 98 and page == 54)
        or (item == 107 and page == 56)
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
    assert set(ITEMS) == set(range(82, 109))
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
