#!/usr/bin/env python3
"""Write visually checked Adi Appendix B cells for items 279--307."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_279_307_hand_keyed.tsv"
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

# Item, gloss, then MN/BR/RM/ML/PL/AS/PD/SM/BK cells as form=label.
# Pipe-separated text preserves separately labelled printed responses.
RAW = """\
279\tabove\tareŋ=1\tare | are=1 | 2\tage=2\tao=3\tage=2\tare | are=1 | 2\tt̪əjoŋ=4\tt̪əjoŋ=4\tt̪əjoŋ=4
280\tbelow\tamoŋ=1\tbapuuk=2\tkhɛmuuk=3\tamo=1\tʃokpuuk=4\tləŋkuuŋ=5\trumkuuŋ=6\thokko=7\tʃokko=7
281\tfar\tmod̪o=1\tod̪ə=2\tad̪o=2\tmorom=3\tad̪o=2\tmot̪əŋ=3\tmot̪əŋ=3\tmot̪əŋ=3\tad̪o=2
282\tnear\toŋe=1\tŋenuuk=2\tnetʃe=3\taŋiaŋ=4\tnəʃe=3\tŋənuuŋ=2\tmoŋio=5\tmoŋe=1\tnetʃe=3
283\tright\tlagbuuk=2\tlabbe | labbe=1 | 4\tlokhbikh=3\tlagd̪aŋ=2\tlaːbuuk=1\tlagbe | lagbe=2 | 4\tlagbuuk=2\tlagbuuk=2\tlukbik=3
284\tleft\tlakke=1\tlatʃe=1\tlokhtʃe=1\tlakkhe=1\tlatʃe=1\tlakke=1\tlakke=1\tlakke=1\tlokhtʃe=1
285\tblack\tjaka=1\tjaka=1\tjaka=1\tjegiaŋ=2\tjaka=1\ttʃəmar=3\tjoraŋ=4\tjaka=1\tkajaŋ=5
286\twhite\tjahuuŋ=1\tjaluŋ=1\tjɛpuŋ | jɛpuŋ=1 | 3\tjetʃi=4\tjapu=3\tjaluŋ=1\tjaʃuuŋ=1\tjaluŋ=1\tpũpõ=6
287\tred\tjaluuŋ=1\tpəjiŋ=2\tjəluuŋ=1\tjelaŋ=1\tjalu=1\tpəjik=2\tjaluuŋ=1\tjaluuŋ=1\tjəluuŋ | luŋkaŋ=1 | 3
288\tgreen\tjajuuŋ=1\tjajuuŋ=1\tjaghe=2\tjetʃaŋ=3\tjadʒe=2\tjajuuŋ=1\tjajuuŋ=1\tjajuuŋ=1\tdʒenio=4
289\tyellow\tjage=1\tjage=1\tnoge=1\tnogẽ=1\tnoge=1\tjage=1\tjage=1\tjage=1\tjage=1
290\twhen (near future)\təd̪ulola=1\thəd̪uulo=1\thed̪d̪uəm=2\thadʒelum=2\thəd̪uulo=1\təd̪uulo=1\təd̪uulo=1\təd̪ut̪lo=1\tarero=3
291\twhere\tkolola=1\thokola=2\thelo=1\tkalal=4\tholo=1\tuuŋkolo=1\tuuŋkolo=1\tkolo=1\thələbo=3
292\twho\thəkola=1\thuula=2\thuupe?=3\tbuu=4\tuula=2\tʃəkola=1\tʃəko=1\thəko=1\thuulo=2
293\twhat\tuuŋkokola=1\thohkola=2\thegolo=3\thakha=4\thogola=3\tkapə?=5\tuuŋko=1\tuuŋkoko=1\thego=3
294\thow many\təd̪ut̪kola=1\thəd̪uut̪kola=1\thed̪d̪uugo=1\tkajkika=2\thəd̪uugola=1\təd̪ut̪ko=1\təd̪uut̪ko=1\təd̪uut̪ko=1\thid̪d̪igo=1
295\tthis thing\thuuːat̪uu=1\tʃuuːat̪o=1\tʃuuat̪uu=1\taguat̪ogu=2\tʃiat̪oʃi=3\tʃuuat̪uu=1\thiat̪uuʃi=3\thiat̪uuʃi=3\tʃum | gilas=4 | 5
296\tthat thing\təat̪uu=1\tanat̪o=2\tahat̪uu=1\tjoat̪ogu=3\taat̪o=1\td̪əat̪uu=1\td̪əat̪uud̪ə=1\təat̪uu=1\tam=4
297\tthese things\thuat̪uu=1\tan at̪o=2\tʃi at̪uu=3\tagu at̪oregu=4\tʃi guud̪uuʃi=5\td̪əat̪uu=6\thi at̪uukuud̪ar=7\thikehi=8\tʃũgũd̪uuŋsim=5
298\tthose things\td̪əat̪uu=1\tan at̪o=2\tʃi at̪uu=4\tjo at̪oregu=5\ta guud̪uua=6\td̪əat̪uu=1\thi at̪uukuud̪ar=7\tətabuŋ=8\ta gũd̪ũŋaa=6
299\t1st sg.(I)\tŋo=1\tŋo=1\tŋo=1\tŋa=1\tŋo=1\tŋo=1\tŋo=1\tŋo=1\tŋo=1
300\t2nd sg.(familiar)\tno=1\tno=1\tno=1\tŋi=2\tno=1\tno=1\tno=1\tno=1\tno=1
301\t2nd sg.(honorific)\tno=1\tno=1\tno=1\tŋi=2\tno=1\tno=1\tno=1\tno=1\tno=1
302\t3rd sg.(generic/male)\tbuu=1\tbuu=1\tʃi=2\tdʒi=3\tŋuu=1\tbuu=1\tbuu=1\tbuu=1\tmeju, mu=1
303\t3rd sg.(female)\tbuu=1\tbuu=1\tʃi=2\tdʒi=3\tŋuu=1\tbuu=1\tbuu=1\tbuu=1\tmuu | me j=1 | 4
304\t1st pl.\tŋolu=1\tŋolu=1\tnonu=1\tŋadʒi=2\tŋonu=1\tŋolu=1\tŋolu=1\tŋolu=1\tũlu=1
305\t2nd pl.(familiar)\tnolu=1\tnolu=1\tmennokəbaŋ=2\tŋadʒi=3\tnonũ=1\tnolu=1\tnolu=1\tnolu=1\tnolu=1
306\t2nd pl.(honorific)\tnolu=1\tnolu=1\tmənnoguuguuŋ=2\tŋija=3\tnonũ=1\tnolu=1\tnolu=1\tnolu=1\tmənnoguuguuŋ=2
307\t3rd pl.\tbulu=1\tbulu=1\tmənu=2\tdʒadʒi=3\tmanu=2\tbulu=1\tbulu=1\tbulu=1\tmaluguud̪uuŋ=4
"""

FIELDS = [
    "Item", "Gloss", "Site_Code", "Site_Name", "PDF_Page", "Printed_Page",
    "Column", "Manual_Transcription", "Source_Cognate_Labels",
    "Review_Status", "Confidence", "Uncertainty", "Reviewer_Method",
    "Reviewed_At", "Reviewer_Declaration",
]


def coordinates(item, code):
    if item == 279:
        return "36", "32", "left"
    if 280 <= item <= 283:
        return "36", "32", "middle"
    if item == 284:
        return ("36", "32", "middle") if code != "BK" else ("36", "32", "right")
    if 285 <= item <= 288:
        return "36", "32", "right"
    if item == 289:
        return ("36", "32", "right") if code in {"MN", "BR", "RM", "ML", "PL", "AS"} else ("37", "33", "left")
    if 290 <= item <= 293:
        return "37", "33", "left"
    if item == 294:
        return ("37", "33", "left") if code in {"MN", "BR", "RM", "ML", "PL", "AS"} else ("37", "33", "middle")
    if 295 <= item <= 298:
        return "37", "33", "middle"
    if item == 299:
        return ("37", "33", "middle") if code in {"MN", "BR", "RM", "ML", "PL"} else ("37", "33", "right")
    if 300 <= item <= 303:
        return "37", "33", "right"
    if item == 304:
        return ("37", "33", "right") if code in {"MN", "BR", "RM", "ML"} else ("38", "34", "left")
    if item == 305:
        return ("38", "34", "left") if code in {"MN", "BR", "RM", "ML", "PL", "AS"} else ("38", "34", "middle")
    if item == 306:
        return ("38", "34", "middle") if code != "BK" else ("38", "34", "right")
    return "38", "34", "right"


def source_data():
    data = {}
    codes = list(SITES)
    for line in RAW.splitlines():
        item_text, gloss, *cells = line.split("\t")
        assert len(cells) == len(codes), (item_text, len(cells))
        parsed = {}
        for code, cell in zip(codes, cells):
            form, labels = cell.rsplit("=", 1)
            parsed[code] = (form, labels)
        data[int(item_text)] = (gloss, parsed)
    return data


def build_rows():
    rows = []
    for item, (gloss, cells) in source_data().items():
        assert set(cells) == set(SITES)
        for code, name in SITES.items():
            form, labels = cells[code]
            pdf_page, printed_page, column = coordinates(item, code)
            row = {
                "Item": str(item), "Gloss": gloss, "Site_Code": code,
                "Site_Name": name, "PDF_Page": pdf_page,
                "Printed_Page": printed_page, "Column": column,
                "Manual_Transcription": form,
                "Source_Cognate_Labels": labels,
                "Review_Status": "attested", "Confidence": "high",
                "Uncertainty": "", "Reviewer_Method": METHOD,
                "Reviewed_At": "2026-08-28",
                "Reviewer_Declaration": DECLARATION,
            }
            assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
            rows.append(row)
    return rows


def main():
    rows = build_rows()
    assert len(rows) == 29 * 9 == 261
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
