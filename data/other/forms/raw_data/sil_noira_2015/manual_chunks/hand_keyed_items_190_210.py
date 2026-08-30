#!/usr/bin/env python3
"""Emit the Noira 2015 items 190–210 OCR-blind manual-review ledger.

Every value in ``ITEMS`` was independently keyed while viewing 400-dpi
rendered source pages. Difficult glyphs were rechecked on 900-dpi renders.
This module never reads PDF text, OCR, scaffold files, or another
transcription at runtime.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
REVIEWED_AT = "2026-08-28"
OUT = Path(__file__).with_name("items_190_210_hand_keyed.tsv")

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

# Syntax: ``printed-cognate-number=form``; multiple source responses are
# separated by ``¦``. Values are literal manual decisions in SITES order.
ITEMS = {
    190: ("give!, he gave", [
        "1=ap¦1=apijo", "1=ap¦1=apijo", "1=apiho",
        "1=apiɖe¦1=apiɖeɖo", "1=ap¦1=apiho", "1=apje¦1=apino",
        "1=ape¦1=apino", "1=apil", "2=ɖʌidɛre¦2=ɖenol",
        "2=ɖidijʌnu", "2=dʌide¦2=dʌidiɖo", "1=ap", "2=ɖe", "2=ɖe",
        "4=sʌge¦4=dʒike", "4=sʌgedʒa¦4=dʒike", "3=ma¦3=beja",
    ]),
    191: ("it burns, it burned", [
        "2=peʈahe¦2=peʈaʈ gujo", "2=peʈaʔhe",
        "2=peʈahe¦2=peʈaʈ gujo¦4=balio", "4=bolʈoʈo¦4=boliʈgojo",
        "4=bolehe¦4=peʈahe", "4=bolʈouʈo¦4=boligoinu",
        "1=həlgave¦1=həlgavino", "6=bolil", "5=silgirʌna",
        "5=silgirʌnʌ", "5=ʃilgire¦5=ʃiligigeju", "4=bəl¦6=bəl",
        "3=dzʌɭ", "3=dʒəl¦4=dʒəl", "7=dʒʊljen",
        "7=dʒʊlʊba¦7=dʒʊljen", "8=ʌdgokin¦8=ʌdʌkin-dan",
    ]),
    192: ("don't die!, he died", [
        "1=ma mohi¦1=mohiʈguji", "1=ma mohi¦1=moiguju",
        "1=mamohi¦1=mɔhiʈ guji", "1=mamoho¦1=moiʈgojo",
        "1=maju¦1=moiʈguju", "1=mamɔhɔ¦1=moigɔinu",
        "1=moi¦1=moino", "2=morjel", "2=mʌrigʌja", "2=mʌrigʌjo",
        "2=mʌrigʌjo¦1=mʌrila-kijo", "2=mər", "2=mʌr", "2=mər",
        "4=gojen", "4=gojen¦4=gwojen", "3=beti",
    ]),
    193: ("don't kill!, he killed", [
        "1=mʌʈʌkio¦1=mʌʔʈurɔ-ku", "1=mʌiʈʌki-no¦1=maɖeha",
        "1=maɖʌhi¦1=ɖɛɖu", "1=mamoho¦1=mʌiɖeɖo",
        "1=maiʈʈakju¦1=ɖɛɖu", "1=mʌiʈʈʌka¦1=mʌiʈʈʌki-nu",
        "1=mʌiʈʈʌka¦1=mʌiʈʈʌ-kinu", "1=maril", "1=mʌriʈakno",
        "1=mariʈa-kano", "1=mʌrilakijo", "1=mʌr", "1=mʌr", "1=mʌr",
        "3=goikʌni", "3=godʒke-nedʒ", "2=pʌrdai",
    ]),
    194: ("fly!, it flew", [
        "2=uɖiʈdʒa¦2=uɖiʈgɔjo", "2=uɽehe¦2=uridʒa", "2=uɖio",
        "2=uɽehe", "2=uɖidʒa¦2=uɖio", "2=uɖeʔe¦2=uɖigoinu",
        "1=nahidʒo¦1=nahigojo", "2=uɖel", "2=uɖigʌja", "2=uriranu",
        "2=urirʌjo¦2=uriʌjo", "2=uɖi", "2=uɖ", "2=uɖ", "3=apiren",
        "3=aphirwa", "3=ʌphirka¦3=ʌphirkʌ-dan",
    ]),
    195: ("walk!, he walked", [
        "1=sal¦1=salju", "1=sal¦1=salju", "2=gojo", "1=sʌnɑ¦1=sʌna",
        "1=saliu¦1=sal", "1=sɑni¦1=sɑninu", "1=sa¦1=saninɔ",
        "1=tsalel¦1=tsalo", "1=tʃʌl", "1=tʌlno",
        "1=tʃʌlo¦1=tʃʌlgajo", "1=tʃal", "1=tsal", "1=tʃəl¦2=ghûm",
        "3=bo", "3=bo¦3=olen", "3=bo¦3=erikin",
    ]),
    196: ("run!, he ran", [
        "1=gugɖe¦1=gugɖju", "2=ɖʌuɖe¦2=ɖʌudiju", "1=gugɖiu",
        "1=gugɖe¦1=gugɖinu", "1=gugɖe¦1=gugɖiu",
        "1=gugɖeʔe¦1=gugɖinu", "1=gugɖe¦1=gugɖijo", "2=ɖowɖil",
        "2=ɖhovaɖ¦2=ɖhovɖi-gʌjo", "2=ɖʌuɖ", "2=ɖʌudʒo¦2=ɖʌuɖigʌjo",
        "2=ɖoɖ", "3=pʌl", "2=ɖʌoɖ", "4=sʌrve¦4=sʌrbdʒen",
        "4=sʌrube¦4=sʌrubdʒen", "5=tʃʌrgube¦5=tʃʌugi",
    ]),
    197: ("go!, he went", [
        "1=dʒa¦1=goju", "1=dza¦1=goju", "1=dʒo¦1=goinu¦2=gojo",
        "1=dʒo¦1=goinu", "2=goju¦2=dza", "1=dʒo¦1=goinu",
        "1=dʒo¦1=gojo", "1=giljo¦1=dzael¦2=giljo¦2=dzael",
        "1=dʒʌ¦1=dʒaʈirʌnu", "1=dza", "1=dʒʌo", "1=dʒa", "1=dza",
        "1=dʒa", "4=sene¦4=dʒolen", "4=sene¦4=olen", "3=ɛde¦3=eri",
    ]),
    198: ("come!, he came", [
        "1=au¦1=aviju", "1=aʊ¦1=aʊiju", "1=aiju¦1=avu",
        "1=ave¦1=avinu", "1=aiju¦1=av", "1=ave¦1=avinu",
        "1=avẽ¦1=avijo", "1=aijel", "2=ija", "1=a", "1=ajʌo", "1=av",
        "2=je", "1=au", "4=hadʒe¦4=hehen", "4=hadʒe¦4=hedʒken",
        "2=pja¦2=pati",
    ]),
    199: ("speak!, he spoke", [
        "2=bun¦2=buniju", "3=ko¦3=koju", "1=gogijo", "1=gog",
        "1=gogiju¦1=goge", "2=buneʔebu-niu", "1=goʈikəjə¦1=goʈikəja",
        "2=ke¦2=bolil", "2=bol", "2=bol", "2=bol", "2=bol", "2=bol",
        "2=bol", "4=ammʌdʒe¦4=mʌdʒike", "4=mʌde¦4=mʌdke",
        "4=mʌndibe¦4=mandija",
    ]),
    200: ("listen!, he heard", [
        "1=homle¦1=homlju", "1=homle¦1=homlju", "5=unaiju", "5=una",
        "5=unaiju¦5=una", "1=homle¦1=homilinu", "1=hamble¦1=hambəlja",
        "1=hombʌlil", "5=unav", "4=ajʌk", "2=sabʌl̪", "2=sabhə",
        "4=ʌikʌ", "3=sun", "6=iome", "6=iʊme¦6=iʊmke",
        "7=tʃʌknibe¦7=tʃʌknija",
    ]),
    201: ("look!, he saw", [
        "2=pal¦2=palju", "2=pal¦2=paliju", "2=pal¦2=palju¦2=palio",
        "2=pal¦2=palinu", "2=palio¦2=pal", "2=paleʔe¦2=palinu",
        "2=palə¦2=paljə", "3=ɖekhel", "3=ɖhek", "3=ɖek", "3=ɖek",
        "1=dʒojũ", "4=paha", "3=ɖekh", "5=doge¦5=doge",
        "5=dokedʒ¦5=ɖowen", "6=arabe¦6=araja",
    ]),
    202: ("I (1st sg)", [
        "4=aj", "4=aj", "3=ai¦4=ai", "4=aje", "3=ai¦4=ai", "4=aẽ",
        "1=mi", "1=mi", "2=hʌi¦3=hʌi", "2=hʌi¦3=hʌi", "?=hũ",
        "2=hũ", "1=mi", "1=mẽj¦1=mẽj", "6=in", "6=iɲ", "5=dʒo",
    ]),
    203: ("you (2nd sg, informal)", [
        "1=t̪u", "1=t̪u", "1=t̪u", "1=t̪e", "1=t̪u", "1=t̪u", "1=t̪ʊ",
        "1=t̪u", "1=t̪u", "1=t̪u", "1=t̪u", "1=t̪u", "1=t̪u",
        "1=t̪u¦1=t̪um", "2=am", "2=am", "3=ne",
    ]),
    204: ("you (2nd sg, formal)", [
        "2=t̪umu", "2=t̪um", "2=t̪umu", "2=t̪umi", "2=t̪umu", "2=t̪u",
        "2=t̪ʊmi", "2=t̪umo", "2=t̪umu", "2=t̪u", "2=tʌmi",
        "1=ap¦2=t̪əmẽi", "2=t̪umhi", "1=ap", "3=am", "3=am", "4=ne",
    ]),
    205: ("he (3rd sg, masculine)", [
        "2=t̪ɔ", "2=t̪oʔo", "2=t̪o", "6=jʌho", "2=t̪o", "5=honuʔu",
        "1=t̪ẽ", "1=t̪o", "2=t̪o", "6=jo", "1=t̪en", "1=t̪e", "2=t̪o",
        "3=wɔ¦3=wo", "8=ɖi", "8=ɖi", "9=ete",
    ]),
    206: ("she (3rd sg, feminine)", [
        "1=t̪eʔe", "1=t̪eʔe", "1=t̪ɛ", "4=jʌhi", "1=t̪ɛ", "3=honiʔi",
        "1=t̪ẽ", "1=t̪i", "1=t̪i", "4=ji", "1=t̪e", "1=t̪e", "1=t̪i",
        "2=wə¦2=wo", "6=ɖi", "6=ɖi", "7=ete",
    ]),
    207: ("we (1st pl, inclusive)", [
        "2=ʌmu", "2=ʌmu", "1=apu", "2=ʌmi", "1=apu¦1=apuhu¦2=ʌmu",
        "1=apũ", "1=apʊ", "2=ʌmu", "2=hʌmu", "3=ubʌrɛ", "1=apen",
        "1=apɳe", "1=apʌɳ", "2=həm", "4=ale", "4=ale", "5=iŋgin",
    ]),
    208: ("we (1st pl, exclusive)", [
        "1=ʌmu", "1=ʌmu", "1=amu", "1=ʌmi beni", "1=amu",
        "1=ʌme¦1=ʌmi", "1=amɪ", "1=amu", "1=hʌmu", "1=ʌm", "1=hʌmju",
        "1=ãmei", "1=amhĩ", "1=hʌm", "3=ale", "3=ale", "2=t̪e eko",
    ]),
    209: ("you (2nd pl)", [
        "1=t̪umu", "1=t̪umu", "1=t̪umu", "1=t̪umi", "1=t̪umu",
        "1=t̪ume¦1=t̪umi", "1=t̪ʊmi", "1=t̪umu", "1=t̪umu", "1=t̪umu",
        "2=t̪ebʌu", "1=t̪əmẽi", "1=t̪umhi", "1=t̪umlog", "4=ape",
        "4=ape", "3=ne",
    ]),
    210: ("they (3rd pl)", [
        "1=t̪eʔẽ", "1=t̪eʔẽ", "1=t̪ɛ", "2=jehɔ", "1=t̪ɛ", "1=t̪eʔẽ",
        "1=t̪ejẽ", "1=t̪a", "3=tja", "2=je", "1=t̪e", "1=t̪eo", "1=t̪e",
        "4=we", "6=ɖi", "6=ɖi", "5=etla",
    ]),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Site_Name", "PDF_Page", "Printed_Page",
    "Column", "Manual_Transcription", "Source_Cognate_Labels", "Review_Status",
    "Confidence", "Uncertainty", "Reviewer_Method", "Reviewed_At",
    "Reviewer_Declaration",
]


def coordinates(item: int, site_index: int) -> tuple[str, str, str]:
    if item == 190:
        page, column = 74, "left"
    elif item == 191:
        page, column = 74, "left" if site_index < 14 else "right"
    elif item == 192:
        page, column = 74, "right"
    elif item == 193:
        page, column = (74, "right") if site_index < 6 else (75, "left")
    elif item == 194:
        page, column = 75, "left"
    elif item == 195:
        page, column = 75, "left" if site_index < 7 else "right"
    elif item == 196:
        page, column = 75, "right"
    elif item == 197:
        page, column = (75, "right") if site_index < 5 else (76, "left")
    elif item == 198:
        page, column = 76, "left"
    elif item == 199:
        page, column = 76, "left" if site_index < 8 else "right"
    elif item == 200:
        page, column = 76, "right"
    elif item == 201:
        page, column = (76, "right") if site_index < 10 else (77, "left")
    elif item in {202, 203}:
        page, column = 77, "left"
    elif item in {204, 205}:
        page, column = 77, "right"
    elif item == 206:
        page, column = (77, "right") if site_index < 13 else (78, "left")
    elif item in {207, 208}:
        page, column = 78, "left"
    else:
        assert item in {209, 210}
        page, column = 78, "right"
    return str(page), str(page - 6), column


def parse_cell(cell: str) -> tuple[str, str, str, str, str]:
    pairs = [part.split("=", 1) for part in cell.split("¦")]
    labels = " | ".join(label for label, _ in pairs)
    forms = " | ".join(form for _, form in pairs)
    return forms, labels, "attested", "high", ""


def rows() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    assert set(ITEMS) == set(range(190, 211))
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
    assert len(out) == 357
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
