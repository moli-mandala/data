#!/usr/bin/env python3
"""Emit the Noira 2015 items 28–54 OCR-blind manual-review ledger.

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
OUT = Path(__file__).with_name("items_028_054_hand_keyed.tsv")

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
    28: ("door", [
        "2=baʔaɳo¦2=baʔaɳɔ¦3=baʔaɳo¦3=baʔaɳɔ",
        "2=baʔaɳo¦2=baʔaɳɔ¦3=baʔaɳo¦3=baʔaɳɔ",
        "2=bʌʔnɔ¦3=bʌʔnɔ", "6=barija",
        "2=baʔa¦2=baʔaɳo¦3=baʔaɳo", "2=bʌʔa", "2=bʌʔa",
        "2=baiɳo¦3=baiɳo", "3=bʌrna", "3=barɳa", "2=banɔ¦3=banɔ",
        "3=bãrɳũ¦5=ɖərvudʒo", "5=ɖʌrwadʒa¦6=ɖar", "5=ɖərvaza",
        "7=ɖordʒa", "8=kiwar", "7=ɖʌrdʒə",
    ]),
    29: ("firewood", [
        "1=nakɖi", "1=lʌkʌɽo", "1=nakʌɖo¦1=naʌɖʌ", "1=nʌkuɽi",
        "1=lakɖu", "1=nakuɖe", "1=nakʌre", "1=lakoɖ", "1=lʌkʌɖi",
        "1=lakɽa", "1=lakʌru", "1=ləkəɖũ", "1=lakuɖʌ", "1=ləkəɖi",
        "3=tʃakan", "3=tʃakan", "2=apo",
    ]),
    30: ("broom", [
        "1=bʌʔɖi", "1=baʔrɪ", "1=baʌɖi", "1=bʌʔɖi",
        "1=baʔaɖi¦1=baʔaɖo", "1=bʌʔɖi", "1=baʔrɪ", "1=baiɖi",
        "1=bɦʌiɖi", "1=bɦʌiɖo", "4=haɖʌni", "2=dʒaɽɖũ¦3=savaraɳi",
        "2=dʒɦaɖu", "2=dʒɦaɖu", "4=dʒʊnʊ", "2=dʒɦaɖu", "4=dʒʊnu",
    ]),
    31: ("mortar", [
        "1=ukhʌl", "1=ukhʌl", "1=ukhʌl¦5=khaɖɳo", "1=ukhʌl",
        "1=ukhʌl¦7=khaɳʌɖoʔ", "1=ukhʌle", "2=mʊhɔle", "1=ukheɭ",
        "1=ukuj", "1=ukhijo", "5=khalɳʊ", "3=pəʈʌro¦5=khəiɳi",
        "1=ukhʌɭi", "1=okhəli¦4=khərəl", "8=kʊɳɖija", "8=kʊɳɖija",
        "9=oson",
    ]),
    32: ("pestle", [
        "4=muhlɔ¦5=muhlɔ", "5=muhlo", "1=mukʌl¦5=mukʌl", "5=muhul",
        "4=muhʌl¦5=muhʌl¦8=ɖoɳ", "5=mohole", "1=ʊkhɔle¦2=ʊkhɔle",
        "5=muheɭ", "7=musija", "7=mucija", "4=musʌnu",
        "2=khul¦9=pəʈʌr", "4=musʌɭi", "3=lõɖɦa¦4=musai",
        "10=goɖʌl", "10=goɖʌl¦12=tʊku", "5=mosor",
    ]),
    33: ("hammer", [
        "1=hʌʈeɖɔ", "1=hʌʈhəɖʊ", "1=aʈʌɖi", "3=pahʔʈoɳ",
        "1=hʌʈhuɖi", "1=hʌʈhoɖi", "1=aːʈhʊrɪ", "1=aʈhoɖa",
        "1=hʌʈoɖi", "1=haʈhoɖi", "1=haʈhoɖi", "1=həʈhoɖi",
        "1=hatoɖʌ", "1=həʈhoɖi¦2=gɦən", "1=hʌʈəra", "1=hʌʈoɽa",
        "2=ghʌn",
    ]),
    34: ("knife", [
        "1=soku¦2=soku¦5=soku", "1=soku¦2=soku¦5=soku",
        "1=saku¦2=saku¦5=saku¦5=ʂuru¦7=ʂuru", "1=saku¦2=saku¦5=saku",
        "1=saku¦2=saku¦5=saku¦5=ʂuru¦7=ʂuru", "1=soku¦2=soku¦5=soku",
        "1=sɔpko¦2=sɔpkʊ", "4=tʃaku", "1=tʃaku¦3=tʃaku¦4=tʃaku",
        "4=tʃaku", "4=tʃaku",
        "1=tʃəku¦3=tʃəku¦3=tʃəro¦3=tʃʊri¦4=tʃəku¦4=tʃəro¦7=tʃəro¦7=tʃʊri",
        "2=suri¦5=suri¦7=suri", "1=tʃaku¦3=tʃaku¦3=tʃʊri¦4=tʃaku¦7=tʃʊri",
        "4=tʃaku", "2=sura¦5=sura¦7=sura", "4=tʃaku",
    ]),
    35: ("axe", [
        "1=kuwʌɖi", "1=kuʊaɽi", "1=kuwaɖo", "1=kuvaɽi", "1=kuwaɖɛ",
        "1=kuwaɖɔ", "1=kʊvaɽɔ", "1=kuraɖ", "1=kuɖad",
        "1=kʊraɖ¦2=kʊraɖ", "1=kuraɖi", "1=kuhaɖi¦2=kori",
        "1=kuraɖʌ", "1=kʊlhaɖi", "4=akhʌl", "4=akhʌi", "3=tʃɛkʈo",
    ]),
    36: ("rope", [
        "1=ɖuʔɖi¦3=ɖuʔɖi¦6=ɖuʔɖi", "1=ɖɦuʔɖi¦1=ɖɔro¦3=ɖɦuʔɖi",
        "1=ɖoɖɔ¦1=ɖuʔɖi", "1=ɖuri", "1=ɖuʔɖi¦7=humb", "3=ɖuʔɖo",
        "1=ɖuɽɪ", "1=ɖoiɖo", "1=ɖojʌɖa", "1=ɖoiɖa", "1=ɖor",
        "1=ɖori¦2=ɖoru", "1=ɖori", "5=rəssi", "2=ɖora", "2=ɖora",
        "2=ɖora",
    ]),
    37: ("thread", [
        "6=huʈi", "1=huʈ", "1=huʈ¦6=huʈ", "1=huʈ¦6=huʈ",
        "1=huʈ¦2=ɖoru¦6=huʈ", "1=huʈe¦6=huʈe", "1=huʈe¦6=huʈe",
        "1=huʈ¦6=huʈ", "2=ɖoro", "2=ɖoro", "2=ɖori", "2=ɖoro",
        "1=suʈ¦2=ɖora¦6=suʈ", "1=suʈ¦2=ɖora¦4=ɖɦaga¦6=suʈ",
        "5=seʈʌm", "5=setʌm", "1=suʈo",
    ]),
    38: ("needle", [
        "1=hui", "1=hui", "1=hwi", "1=hui", "1=hui", "1=hue", "1=hʊje",
        "1=hui", "1=suj", "1=suj", "1=suj", "1=soi", "1=sui", "1=soi",
        "1=sũj", "1=sũj", "1=sũi",
    ]),
    39: ("cloth", [
        "3=kapɽi", "3=kapʌɽi", "1=sako", "1=saʔko¦4=kuɽiʈe",
        "1=saʔkhõ¦3=kʌpʌɽ", "2=nukʌɖe", "2=nʊgɔrɔ", "5=ɸaɖko",
        "3=kʌpʌɖa", "3=kʌpʌɽa", "3=kʌpəɽu", "3=kapaɖ¦3=kopəɖũ",
        "3=kapʌɖʌ", "3=kəpəɖa", "4=aŋgi", "4=aŋgi", "3=kʌpʌɽa",
    ]),
    40: ("ring", [
        "1=muɖhi", "1=munɖi", "1=munɖi", "1=munɖi", "1=munɖi",
        "1=munɖi", "1=muɖɪ", "1=munɖʌɖo", "1=munɖi", "1=munɖi",
        "2=iʈi", "2=wĩʈi", "1=munɖi¦4=ʌŋgʌʈhi", "3=mũɖəri¦4=ãguʈhi",
        "1=munʈi", "1=munɖi", "1=mʊnɖi",
    ]),
    41: ("sun", [
        "1=ɖihi", "1=ɖihi", "1=dihi", "1=ɖihi", "1=dihi", "1=ɖihi",
        "1=ɖɪhɪ", "1=ɖih", "1=ɖis", "1=ɖin", "2=surj",
        "2=surədʒ¦2=surjə", "2=surijʌ", "2=surədʒ", "5=gomit",
        "5=gomedʒ", "6=ɖevta",
    ]),
    42: ("moon", [
        "1=saɳɖ", "1=saɳʈ", "1=sãɖ", "1=saɳɖ", "1=sãɖ",
        "1=saɳɖ¦1=tʃʌnɖe", "1=sãɖe", "1=tʃaɳɖ", "1=tʃanɖ", "1=tʃanɖ",
        "1=tʃanɖo", "1=tʃanɖo¦1=tʃəndrə", "1=tʃʌnɖrʌ",
        "1=tʃãɖ¦1=tʃəndrəma", "1=ʈhenɖedʒ",
        "1=ʈhenɖedʒ¦2=tʃʌnnigo-gedʒ", "2=mindi devta",
    ]),
    43: ("sky", [
        "1=dʒʊg", "1=dʒʊg", "1=dzʊg", "1=dʒʊg", "1=dzʊg",
        "2=dʒʊge¦4=ʊaɖʌlo", "1=hɔrɔg¦2=dʒʊg", "2=dzug", "5=vaɖja",
        "5=waɖja", "4=waɖʌɭu", "3=akas", "3=akaʃʌ", "3=akaʃ",
        "6=bʌɖra", "6=baɖʌra", "6=bʌɖʌra",
    ]),
    44: ("star", [
        "1=ʈara", "1=ʈara", "1=ʈara¦1=ʈaru", "3=sanʌɳa", "1=ʈaru",
        "1=ʈara", "1=ʈara", "1=ʈara", "2=tʃaɖʌni", "1=ʈaro", "1=ʈara",
        "1=ʈara¦1=ʈaro", "1=ʈara", "1=ʈara", "4=ipil", "4=ipil",
        "5=pipindʒor",
    ]),
    45: ("rain", [
        "4=poɖɛ hɛp¦5=pʌĩ", "1=ʊʌrhaʈ¦4=pʌĩ poɖɛ hɛ¦5=pʌĩ poɖɛ hɛ",
        "1=wohʌraʈ¦5=paĩ", "1=vʌrhaʈ",
        "1=wohʌraʈ¦4=pʌĩ poɖɛ hɛ¦5=pʌĩ poɖɛ hɛ¦7=pãʔi",
        "1=vʌrhaʈ poɖɛ he¦1=vʌrhaʈe¦4=vʌrhaʈ poɖɛ he", "1=vʌrhaʈe",
        "5=paɳi¦6=paɳi", "6=pani pʌɽina¦7=pani pʌɽina", "6=paniʌirɔnʊ",
        "6=paniʌiro", "1=vərsaɖ", "3=pausʌ", "2=bərəs¦2=wərsa",
        "8=ɖa¦9=barsʌrɔ", "11=dagʌma", "10=manɖo",
    ]),
    46: ("water", [
        "1=paĩ", "1=paĩ", "1=paĩ", "1=paĩ", "1=paʔi", "1=pʌĩ", "1=pãʔĩ",
        "1=paɳi", "1=pani", "1=pani", "1=pani", "1=paɳi", "1=paɳi",
        "1=pani¦3=dʒəl", "3=ɖa", "3=ɖa", "4=dʒʌpə",
    ]),
    47: ("river", [
        "3=khaɖi", "3=khaɖi", "3=khaɖ", "3=khaɖi", "3=khaɖi",
        "1=noje¦3=khaɖi", "1=nɔje¦3=kaɽi", "6=njeɳɖ¦7=namili",
        "1=nʌɖi", "1=nʌɖi", "1=nʌɖi", "1=nəɖi", "1=nʌɖi", "1=nəɖi",
        "4=gaɖa", "4=gaɖa", "5=pʌrai",
    ]),
    48: ("cloud", [
        "1=vaʈlɔ", "1=ʊaɖlɔ", "1=wadʌlo", "1=vaɖʌlo", "1=waɖʌlo",
        "1=ʊaɖlɔ", "1=vaɖəlɔ", "1=waɖeɭ", "1=vaɖija", "1=waɖija",
        "1=vaɖʌɭu", "1=vaɖəɭ", "2=ɖhʌg", "1=baɖəl", "1=bʌɖʌɖo",
        "1=baɖʌra", "1=bʌɖʌra",
    ]),
    49: ("lightning", [
        "1=vidʒ", "1=vidʒ", "1=widz", "1=vidʒ", "1=widz",
        "1=ʊidʒe¦1=ʊidʒunu", "1=vidʒorʊʈhɔ", "1=βidzʌɭe", "1=vidʒ",
        "1=vidʒ", "1=idʒ", "1=widʒʌɭi", "1=βidz", "1=bidʒəli",
        "1=bidʒli", "1=bidʒli", "1=bidʒʌli",
    ]),
    50: ("rainbow", [
        "1=baɳi", "1=baɳi", "1=baɳ", "1=baɳ", "1=baɳi", "1=baɳe",
        "1=bãɳe", "1=banɖ", "1=banɖ", "3=ɖhʌnuʃ baɳ", "3=ɖhʌnuʃ",
        "2=meigɦɖ-ɦanuʃa", "2=indrʌɖ-ɦanuʂʌ", "2=indrəɖ-ɦənuʂ",
        "4=ʈʌmʊk loɖʌdʒ", "4=ʈʌmʊk loɖʌdʒ", "4=ʈʌmko lodedʒ",
    ]),
    51: ("wind", [
        "3=varo", "3=varʊ", "3=waro", "3=varo", "3=varɔ¦3=waru", "3=vaje",
        "3=vaje", "3=waro", "3=vargo", "3=varo", "3=varo",
        "1=pauvən¦1=pəvən¦2=havar¦3=wairo", "3=wara", "2=həva",
        "5=kojo", "5=kojo", "6=ora",
    ]),
    52: ("stone", [
        "2=ɖɔɳi", "3=ɖogʊɖʊ", "2=ɖoɳ", "2=ɖoɳ", "2=ɖoɳ", "2=ɖɔɳe",
        "2=ɖɔɳe", "3=ɖegoɖ¦5=ɖegoɖ", "3=ɖʌgʌɖ¦5=ɖʌgʌɖ",
        "3=ɖʌgʌɖo", "3=ɖʌgʌɖo", "1=pəʈʈhər", "3=ɖʌgʌɖ¦5=ɖʌgʌɖ",
        "1=pəʈʈhər", "4=ɖega¦5=ɖega", "4=ɖega¦5=ɖega", "4=tʃago",
    ]),
    53: ("path", [
        "2=vaʈi", "2=vaʈi", "2=waʈ", "2=vaʈ", "2=pagwaʈ¦2=waʈ", "2=vaʈe",
        "1=rəhʊ¦2=vaʈe", "2=waʈ¦9=marg", "2=vaʈ", "6=gaɖvaʈ", "2=vaʈi",
        "1=rəsʈo¦2=waʈ¦4=keɖi", "1=rʌsʈa¦2=paiwaʈ", "1=rasʈa",
        "7=kora", "7=kora", "8=daj",
    ]),
    54: ("sand", [
        "3=rɛʔkhɔ", "1=reʔʈi¦3=reʔko¦3=reʔʈi", "1=reʈo¦3=rɛʔkhɔ",
        "1=reʈo", "1=reʈo¦3=rɛʔkho", "1=reʈɔ", "1=rẽʔʈɔ¦3=rẽʔʈɔ",
        "2=welʈo", "1=reʈu", "1=reʈu", "1=reʈu", "1=rɛʈi",
        "1=reʈi¦2=walʊ", "1=rɛʈi¦2=balu", "1=reʈi¦4=beʈʃil",
        "1=reʈi", "1=reʈi",
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
        **{item: 38 for item in range(28, 31)},
        **{item: 39 for item in range(31, 35)},
        **{item: 40 for item in range(35, 38)},
        **{item: 41 for item in range(38, 43)},
        **{item: 42 for item in range(43, 47)},
        **{item: 43 for item in range(47, 52)},
        **{item: 44 for item in range(52, 55)},
    }[item]
    if item == 30 and site_index >= 5:
        page = 39
    elif item == 34 and site_index >= 6:
        page = 40
    elif item == 37 and site_index >= 14:
        page = 41
    elif item == 42 and site_index == 16:
        page = 42
    elif item == 47:
        page = 42 if site_index == 0 else 43

    left_current_page = {
        28: set(), 29: set(), 30: set(), 31: set(range(10)),
        32: set(range(11)), 33: set(), 34: set(), 35: set(range(17)),
        36: set(), 37: set(), 38: set(range(17)), 39: set(range(17)),
        40: set(range(8)), 41: set(), 42: set(), 43: set(range(17)),
        44: set(range(2)), 45: set(), 46: set(), 47: set(),
        48: set(range(17)), 49: set(range(13)), 50: set(), 51: set(),
        52: set(range(17)), 53: set(range(17)), 54: set(range(2)),
    }
    continuation_left = (
        (item == 30 and page == 39)
        or (item == 34 and page == 40)
        or (item == 37 and page == 41)
        or (item == 42 and page == 42)
        or (item == 47 and page == 43)
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
    assert set(ITEMS) == set(range(28, 55))
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
