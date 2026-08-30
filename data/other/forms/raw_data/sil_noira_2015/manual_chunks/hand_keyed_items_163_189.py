#!/usr/bin/env python3
"""Emit the Noira 2015 items 163–189 OCR-blind manual-review ledger.

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
OUT = Path(__file__).with_name("items_163_189_hand_keyed.tsv")

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
# separated by ``¦``. ``BLANK`` means the source explicitly prints
# ``0 no entry``. Values are literal manual decisions in SITES order.
ITEMS = {
    163: ("twenty", [
        "1=vihi", "1=vihi", "1=vihi", "1=vihi", "1=vihi", "1=vihi",
        "1=vĩhi", "1=βih", "1=vis", "1=vis", "1=vis", "1=vis",
        "1=βis", "1=bis", "2=isʌ", "2=isa", "2=iso",
    ]),
    164: ("one hundred", [
        "1=ek hʌu", "1=ek hʌu", "1=ek hʌu", "1=hov", "1=ekhʌo",
        "1=ek hʌu¦1=ekhɔve", "1=ekhɔve", "1=ho", "3=sʌmbʌr",
        "3=sʌmbʌr", "1=hʌu", "2=so", "3=ʃʌmbhʌr", "2=so",
        "4=çeɖi", "4=çeɖi", "4=çedi",
    ]),
    165: ("who?", [
        "1=kɔɖo", "1=koɖo", "1=koɖo", "1=koɖo", "1=koɖu",
        "1=kɔɖo", "1=kɔrɔ", "1=kuɳ", "3=koi", "1=koɳ¦2=koɳ",
        "2=koɳ", "1=kɔɳ¦2=kɔɳ", "1=koɳ¦2=koɳ", "2=koun",
        "4=dʒe", "4=dʒe", "5=hɛre",
    ]),
    166: ("what?", [
        "3=ki", "3=ki", "3=ki", "1=ka", "3=ki", "1=kage", "1=kage",
        "3=kai", "3=kʌi", "3=kʌi", "2=su", "2=ʃũ", "3=kai",
        "3=kja", "5=tʃutʃ", "5=tʃutʃ", "4=nan",
    ]),
    167: ("where?", [
        "1=ka", "1=kʌ", "1=ka", "1=ka", "1=ka", "3=kahʔɔtɔhɔ¦2=kʌʔha",
        "1=kã", "1=kã", "0=t̪it̪a", "2=kʌha", "7=kaj", "7=kjã",
        "2=kothe", "2=kəhã", "5=tuvʌn", "9=olen¦5=toden¦5=tuvan",
        "8=miŋki",
    ]),
    168: ("when?", [
        "1=kʌʔha", "1=kʌʔha", "1=kedihi", "1=kʌha", "1=kẽhẽ¦1=kedihi",
        "1=kʌʔha", "1=kʌʔhã", "1=kedi", "8=ʌmi", "1=kʌvj",
        "2=kjarɛ", "2=kjarei", "1=kẽwhã¦1=kʌɖi", "3=kəb",
        "10=tʃola", "10=tʃola", "9=mieran",
    ]),
    169: ("how many?", [
        "1=keʈʌhɔ", "1=keʈʌhɔ", "1=keteho", "1=ket̪ija¦1=keʈʌhe",
        "1=keteho¦1=keʈuhu", "1=keʈʌhɔ", "1=keʈʌhu", "4=kolakh",
        "2=mukʈa", "1=keʈla", "1=keʈlu", "1=keiʈla", "1=kiʈi",
        "1=kiʈəne", "3=tʃoʈokuen", "3=tʃoʈo", "5=mijan",
    ]),
    170: ("what kind?", [
        "9=kɛhinʌhɔ", "9=kɛhinʌhɔ", "8=ket̪idzat̪¦9=hehinʌho",
        "9=kɛhinʌhɔi", "8=ket̪idzat̪¦9=kehʌloho", "9=kɛhinʌhɔ",
        "4=kɔnjariʈɔ", "12=kolakh-dzat̪iɳ", "9=kehʌɳɖiase",
        "11=kʌça", "10=kja", "3=kevi-prʌkar¦5=kevidʒat̪",
        "7=konʈiɖ-prʌkʌrtse",
        "3=eisprakar¦3=kisprʌkar¦3=kist̪ʌrʌh¦6=kəisa",
        "15=tʃupar", "15=tʃupar", "13=nuki¦14=nusan",
    ]),
    171: ("this", [
        "3=oʔõ", "3=oʔõ¦3=õ", "3=o¦3=oʔɔ", "4=i", "3=oʔõ¦3=õ",
        "5=eʔĩ¦3=eʔĩ¦3=oʔõ", "1=ãi", "2=jo", "2=ijha", "4=i",
        "2=ja", "1=ɑ", "1=hɑ", "2=jəh¦2=jɔ¦2=jih", "5=ini",
        "5=ini", "1=han",
    ]),
    172: ("that", [
        "1=phɔle", "1=phɔlo", "1=phɔlɔ¦2=to", "5=hono",
        "1=phɔlo¦2=t̪õ", "5=hono", "BLANK", "2=t̪o", "3=t̪it̪ʌha",
        "2=t̪a", "2=tja", "1=pelũ", "2=t̪o", "4=wɔh¦4=wo",
        "6=ɖi", "6=ɖi", "7=hʌuta",
    ]),
    173: ("these", [
        "9=eʔẽ¦7=eʔẽ", "9=eʔẽ¦7=eʔẽ¦4=ẽ¦3=ẽ¦7=ẽ¦6=ẽ", "4=i",
        "4=i", "6=e¦4=e¦3=e¦7=e¦9=eʔẽ¦7=eʔẽ", "7=eʔja", "BLANK",
        "6=jõ", "8=ʌt̪ʌli", "4=i", "3=ai¦4=ai", "3=ɑ", "5=he",
        "6=je", "9=ini", "9=ini", "5=han",
    ]),
    174: ("those", [
        "1=phɔle", "1=phɔle", "4=te", "3=hono", "4=t̪e", "3=onja",
        "BLANK", "4=t̪a", "4=t̪e", "4=t̪a", "4=t̪i", "1=pelã",
        "4=t̪e", "5=βe¦5=ve", "7=ɖi", "7=ɖi", "6=hʌuta",
    ]),
    175: ("same", [
        "1=sarke", "1=ek sarko", "1=harko", "1=ek halko", "1=sackha",
        "1=ek harko", "1=hʌrikɔdʒe", "1=harkas̪", "1=ek sʌrkoi",
        "1=sʌrkijo", "1=ek sʌrka", "1=sərkũ", "1=sarkha", "2=səman",
        "3=mʌtʃika", "3=mjaka-kidʒa¦3=mjatʃika", "1=bisʌrika",
    ]),
    176: ("different", [
        "6=ɖihiro", "6=ɖihiro", "4=alog¦6=ɖihiro", "5=ʌŋgʌŋgo",
        "1=dʒuɖu¦6=biru", "1=dʒuɖo", "1=dʒoɖɔ", "3=pharek",
        "1=dʒuɖa", "4=ʌlʌg", "7=njaru", "1=dʒuɖi",
        "3=φʌrʌk¦4=alʌgʌlʌg¦2=wegʌlʌ",
        "3=φʌrk¦4=əlʌgəlʌg¦6=bhinə", "4=ʌlʌgʌlgo",
        "4=ʌlʌgʌlgo¦8=neranera", "4=ʌlʌgʌlgo",
    ]),
    177: ("whole", [
        "1=akhɔ", "1=akhu", "1=akhuwo", "1=akho", "1=akhwo", "1=akho",
        "1=ɑːkhɔ", "1=akhwalo", "2=ovʌlʌs", "BLANK", "1=akho",
        "1=ɑkhũ", "3=purnə", "3=pura¦3=purnə", "BLANK", "4=sʌdʒʌka",
        "BLANK",
    ]),
    178: ("broken", [
        "2=phuʈinɔ", "2=phuʈlu", "2=puʈno", "2=phuʈunu", "2=puʈiio",
        "2=phuʈunu", "3=ʈuʈnɔ", "1=phuʈel", "2=phuʈlu",
        "2=phuʈigʌjo", "2=phuʈija", "3=ʈuʈelu¦4=bhãgelu",
        "1=φuʈʌlele", "3=ʈuʈa", "5=tja", "5=tja", "6=ʌrom",
    ]),
    179: ("few", [
        "1=t̪huɖɔ", "1=thuɖo", "1=t̪oɖo", "1=thuɖo", "1=thuɖo",
        "1=thuɖo", "1=t̪huɽɔ", "1=thorʌs̪", "4=vaj", "2=dʒʌraka",
        "2=dʒʌrasoku", "1=t̪hoɖũ", "1=t̪hoɖa¦3=kahi", "1=t̪hoɖa",
        "5=dʒisa", "2=dʒarasa¦5=dʒisa", "2=dʒirisa",
    ]),
    180: ("many", [
        "3=dʒʌst̪i", "1=dʒaʔko¦3=dʒast̪i", "1=dzaʔakho", "1=dʒako",
        "1=dzaʔakho", "1=dʒaʔko", "1=dʒaʔkhɔ", "10=dzobed̪",
        "8=mukʈa", "8=mukʈa", "3=dʒast̪i", "4=ghʌɳu",
        "5=puʂkʌl¦6=bʌrets", "7=bəhut̪", "11=gonika",
        "11=gonedʒka", "9=khobo",
    ]),
    181: ("all", [
        "5=akhe", "5=akhe", "5=akhe", "5=akho", "1=baɖe¦5=akhe",
        "5=akhe", "2=bɔʈedʒe", "5=akha¦7=boʈha", "6=hogai",
        "2=bʌʈu¦7=bʌʈu", "3=sʌu", "5=akho", "3=sʌrwʌ", "4=səb",
        "4=seb", "4=seb", "8=pura",
    ]),
    182: ("eat!, he ate", [
        "1=kha¦1=khaɖu", "1=kha¦1=khaɖo", "1=khaɖo", "1=khɔ¦1=kaɖo",
        "1=kha¦1=khaɖo", "1=khaɖo¦1=kho", "1=kho¦1=keɖo", "1=khaije",
        "1=khʌile¦1=khʌlʌna", "1=khava¦1=kha", "1=khao¦1=khaɖu",
        "1=kha", "1=kha", "1=kha", "2=dʒome¦2=dʒojen",
        "2=dʒome¦2=dʒowen", "3=tebe¦3=teja",
    ]),
    183: ("bite!, he bit", [
        "1=sʌu¦1=sʌvijo", "1=sau¦1=sʌvjo", "1=saulio¦1=sʌu",
        "1=sʌve¦1=savino", "1=sauwio¦1=sau", "1=sʌve¦1=sʌvinɔ",
        "1=save", "4=tsaiel¦5=tsaiel¦5=saiel", "4=tʃav¦4=tʃʌvilʌna",
        "4=tʃavijena", "4=tʃajliɖu", "2=khurɖ", "4=tsau", "3=kaʈa",
        "6=kʌvedʒ¦6=kʌpkine", "6=kabedʒ¦6=kakenedʒ",
        "7=bʌrube¦7=hʌruj",
    ]),
    184: ("he is, he was hungry", [
        "2=hukinu-hui¦2=hukinu-hʌʈu", "1=buk lʌgi he", "2=pukh̪lagi",
        "2=phknʌgʌi¦2=phuknʌ-giʈi", "2=pukh̪lagi",
        "2=phinoʈu¦2=phkinuho je", "1=bhûknʊdʒ ɾeh¦1=bhûkinədʒ rəino",
        "1=buklagi", "1=bhuklʌgni¦1=bhuklʌgnil", "1=bhuklagni",
        "1=bhuklʌgi", "1=bhûkjo", "1=bhukela", "1=bhûkh",
        "3=rʌŋgʌdʒen¦3=rʌŋgedʒka", "3=rʌŋgʌdʒen¦3=rʌŋgedʒen",
        "4=tʃatpʌkka¦4=tʃatpʌkkʌ dan",
    ]),
    185: ("drink!, he drank", [
        "1=piɖu", "1=paĩ piʈnu", "1=piɖo", "1=piʈuʈu¦1=piɖo", "1=piɖo",
        "1=pi¦1=piɖno", "1=pi¦1=piɖno", "1=pil¦1=piɖo",
        "1=pijle¦1=pilina", "1=pirʌnu", "1=piro", "1=pi", "1=pi",
        "1=pi", "2=nʊnʊb¦2=nʊen", "2=nʊnʊba¦2=nʊnudan",
        "3=delenka delenʌ dan",
    ]),
    186: ("he is, he was thirsty", [
        "1=t̪ɔrho¦1=t̪or", "1=tɔro legi hɛ", "1=toroholagi",
        "1=t̪oro nʌgiʈi", "1=toroholagi", "1=torʌho¦1=torʌhinu",
        "1=tərne reho", "2=t̪ihlagi", "3=pʌipis-lʌgini¦3=pʌipislʌ-ginil",
        "2=t̪islagni", "1=t̪ʌrʌslagi", "1=t̪hərəʃo", "2=t̪ʌhanel",
        "3=pjasuhe¦3=pjasat̪ha", "5=ɖaʈaʈʌŋ-dʒʌn",
        "5=daʈaʈʌŋ-ken¦5=ɖaʈaʈʌŋ-kenɖan",
        "4=bʌtamka¦4=bʌtʌmkʌ-dan",
    ]),
    187: ("sleep!, he slept", [
        "1=huvidʒa¦1=huvidʒu", "1=huidʒa¦1=huvehe", "1=huwio",
        "1=hovʌhe¦1=houviɖ goju", "1=huvidʒa¦1=huvidʒu¦1=huwio",
        "1=huvidʒo¦1=huvigoi-nu¦1=huvinu", "1=huv¦1=huvejo", "1=hujel",
        "1=suirʌnu¦1=suirʌnal¦3=suirʌnu¦3=suirʌnal", "1=suvigʌjo",
        "1=sujo¦1=suidʒa", "1=sui¦3=sui", "2=dzĥop", "3=so",
        "4=giʈidʒe¦4=giʈijen", "4=giʈidʒba¦4=giʈijen",
        "5=kʌpoka¦5=kʌpokka",
    ]),
    188: ("lie down!, he lay down", [
        "9=huʈʈidʒʌ¦9=huʈʈiguju",
        "9=luʈʈidʒa¦9=luʈʈuguju¦1=luʈʈidʒa¦1=luʈʈuguju¦2=luʈʈidʒa¦2=luʈʈuguju",
        "9=huʈʈidʒʌ¦9=huʈʈiguju",
        "9=nuʈʈidʒa¦9=nuʈʈiɖ gojuhu¦1=nuʈʈidʒa¦1=nuʈʈiɖ gojuhu",
        "9=loʈu¦1=loʈu¦2=loʈu",
        "9=nuʈʈidʒo¦9=nuʈʈigoi-nu¦1=nuʈʈidʒo¦1=nuʈʈigoi-nu",
        "3=aɖu pəɖə¦3=nuʈiʈ və", "9=loʈel¦1=loʈel", "6=hendo suhino",
        "5=kʌŋligʌgo", "4=gʌrbʌɖi-dʒa¦4=gʌrbʌɖi-gʌjo", "3=pəɖir",
        "2=lek", "1=leʈ¦2=leʈ", "1=leʈeʈen¦1=leʈeʈjen",
        "8=tekaken¦8=kekʌi", "7=kʌpɔbe¦7=kʌpi",
    ]),
    189: ("sit down!, he sat down", [
        "2=bɔ¦2=bɔhju", "2=boidʒʌ¦2=boihija", "2=bɔhju¦2=bɔho",
        "2=bohidʒa bohinu¦2=bohiʈgoju", "1=boʈhuhe¦1=biodʒa",
        "2=bohidʒo¦2=bohinu", "2=bohiʈrə¦2=bohiʈvəhũ", "2=boho¦2=bohel",
        "1=bʌʈidʒa¦1=bʌʈigʌjo", "1=bʌʈno", "2=bʌso¦2=bʌsigʌjo",
        "2=bes", "2=bʌs", "1=bejʈh", "4=suba¦4=subaŋen",
        "4=subʌi¦4=subanken", "3=pətebe¦3=pəte",
    ]),
}

FIELDS = [
    "Item", "Gloss", "Site_Code", "Site_Name", "PDF_Page", "Printed_Page",
    "Column", "Manual_Transcription", "Source_Cognate_Labels", "Review_Status",
    "Confidence", "Uncertainty", "Reviewer_Method", "Reviewed_At",
    "Reviewer_Declaration",
]


def coordinates(item: int, site_index: int) -> tuple[str, str, str]:
    page = {
        163: 67, 164: 67, 165: 67, 166: 68, 167: 68, 168: 68,
        169: 68, 170: 68, 171: 69, 172: 69, 173: 69, 174: 69,
        175: 70, 176: 70, 177: 70, 178: 70, 179: 70, 180: 71,
        181: 71, 182: 71, 183: 71, 184: 72, 185: 72, 186: 72,
        187: 72, 188: 73, 189: 73,
    }[item]
    thresholds = {
        165: (5, 68), 167: (15, 68), 170: (3, 69), 172: (2, 69),
        174: (3, 70), 176: (12, 70), 179: (3, 71), 181: (12, 71),
        183: (12, 72), 185: (8, 72), 187: (3, 73), 189: (16, 74),
    }
    if item in thresholds and site_index >= thresholds[item][0]:
        page = thresholds[item][1]

    left_sites = {
        163: set(), 164: set(), 165: set(), 166: set(range(17)),
        167: set(range(15)), 168: set(), 169: set(), 170: set(),
        171: set(range(17)), 172: set(range(2)), 173: set(),
        174: set(), 175: set(range(17)), 176: set(range(12)),
        177: set(), 178: set(), 179: set(), 180: set(range(17)),
        181: set(range(12)), 182: set(), 183: set(), 184: set(range(17)),
        185: set(range(8)), 186: set(), 187: set(), 188: set(range(6)),
        189: set(),
    }
    continuation_left = item in thresholds and site_index >= thresholds[item][0]
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
    assert set(ITEMS) == set(range(163, 190))
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
