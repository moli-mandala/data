#!/usr/bin/env python3
"""Write visually checked Adi Appendix B cells for items 219--248."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_219_248_hand_keyed.tsv"
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
# Pipe-separated text preserves repeated printed responses and label sequences.
RAW = """\
219\tto take\tlanam=1\tlanam=1\tlõnam=1\tdʒakt̪uŋ=2\tlanam=1\tlanam=1\tlanam=1\tlanam=1\tlõŋnam=1
220\tto give\tbinam=1\tbinam=1\tbinam=1\tramt̪uŋ=2\tdʒinam=1\tbinam=1\tbinam=1\tbinam=1\tbinam=1
221\tto kill\tpənnam=1\tikenam=4\tmukhenam=2\tluat̪ma=5\tmokenam=2\tkejnam=3\tpət̪nam=1\tpənnam=1\tmennam=1
222\tto die\thunam=1\tʃinam | ʃinam=1 | 2\tʃinam | ʃinam=1 | 2\tʃima=2\tʃinam | ʃinam=1 | 2\tʃuunam | ʃuunam=1 | 2\tʃinam | ʃinam=1 | 2\thinam=1\tʃinam | ʃinam=1 | 2
223\tto love\tajanam=1\tajanam=1\tajanam=1\tajanma=2\tajanam=1\tajanam=1\tajanam=1\tajanam=1\tajanam=1
224\tto hate\tmuurenam=1\tkaŋkinam=2\tkõlõrnam=3\tmiaŋuuma=4\tkarornam=3\tajamanam=5\tkaŋkinam=2\tkaŋkinam=2\tkõlõrnam=3
225\tone\taʔir=1\takon=2\takhin=2\takan=2\takhen=2\takon=2\tat̪əl=3\tat̪əl=3\takhen=2
226\ttwo\taŋɲi=1\taŋɲi=1\teɲi=1\tnə=2\taɲi=1\taŋɲi=1\taŋɲi=1\taŋɲi=1\taɲi=1
227\tthree\taum=3\taun=3\teum=3\tham=4\taum=3\tad̪un=1\taŋum=2\taum=3\taum=3
228\tfour\taʔi=2\tappi=3\tepi=3\tpə=1\tappi=3\tappi=3\tappi=3\tappi=3\tappi=3
229\tfive\taŋɲo=1\taŋɲo=1\taŋo=1\tpaŋu=2\taŋo=1\təŋo=1\tpilŋo=1\tpirŋo=1\tõŋo=1
230\tsix\takkeŋ=1\takke=1\tekhi=1\tsaːp=2\takkhe=1\takkhuu=1\takkeŋ=1\takkeŋ=1\takkhuu=1
231\tseven\tkənnut̪=1\tkinit̪=1\tkũnnu=1\traŋal=2\tkənə | khuunuu=1 | 1\tkuunut̪=1\tkuunut̪=1\tkuunut̪=1\tkũnuu=1
232\teight\tpiɲi=1\tpiɲi=1\tpiɲi=1\trajeŋ=2\tpiɲi=1\tpiɲuu=1\tpuuɲuu=1\tpiɲuu=1\tpiɲi=1
233\tnine\tkonaŋ=1\tkonaŋ=1\tkonoŋ=1\tkaɲiem=2\tkona=1\tkonaŋ=1\tkonaŋ=1\tkonaŋ=1\tkonoŋ=1
234\tten\təjuŋ=1\təjuŋ=1\tuujuŋ=1\thaŋt̪ak=2\tuuju=1\təjuŋ=1\tuujuŋ=1\tuujuŋ=1\tuujuŋ=1
235\televen\təjuŋkola ako=1\təjuŋkola ako=1\tuujuŋkolaakin=1\thaŋt̪akkola at̪əl=2\tuujulaken=1\takonkou=3\tuujuŋkolaat̪əl=1\tuujuŋkolaat̪əl=1\tuujulakhen=1
236\ttwelve\təjuŋkola aŋɲi=1\təjuŋkola aŋɲi=1\tuujuŋlokaeɲi=1\thaŋt̪akkola nə=2\tuujulaɲi=1\taŋɲikou=3\tuujuŋkolaaŋɲi=1\tuujuŋkolaaŋɲi=1\tuujulaɲi=1
237\ttwenty\tejuŋaŋɲi=1\tejuŋaŋɲi=1\ttʃaŋɲi=2\thaŋtaŋɲə=3\ttʃaŋɲi=2\təjuŋaŋɲi=1\tuujuŋaŋɲi=1\tuujuŋaŋɲi=1\ttʃaŋɲi=2
238\thundred\tluuŋko=1\tluuŋkon=1\tluuŋgo=1\tluuŋko=1\tlu=1\tluuŋkou=1\tluuŋko=1\tluuŋko=1\tluuŋ=1
239\tthousand\thədʒar=1\tədʒar=1\thedʒar=1\thadʒar=1\thedʒer=1\thadʒar=1\tədʒar=1\tədʒar=1\thadʒar=1
240\tfew\taŋoŋ=1\taŋoŋko=1\tatʃonago=2\taɲio=1\tmatʃi=3\tamenəko=4\taŋoŋ=1\taŋoŋ=1\tadʒi=3
241\tsome\taŋoŋoko=1\taŋoŋoko=1\thegope=2\taɲio=1\tadʒer=3\taŋoŋ=1\taŋoŋgam=1\taŋoŋgam=1\tadʒi=3
242\tmany\tbodʒe=1\tammo=2\teuu=3\tbudʒi=1\tauu=3\tbodʒə=1\tbodʒə=1\tbodʒe=1\tauu=3
243\tall\tt̪akam=1\tmaluŋ=2\tappuuŋ=5\tkəbaŋ=3\tt̪akam=1\tkəba=3\tt̪akam=1\tt̪abuŋ=4\tkəbaŋ=3
244\tbig\tbot̪t̪əna=1\tbot̪t̪əna=1\tt̪əbən=2\tnəbə=3\tat̪t̪ə=4\tbot̪t̪əna=1\tbot̪t̪əna=1\tbot̪t̪əna=1\tt̪ənə̃=5
245\tsmall\tamena=1\tamena=1\tatʃona=2\tule=3\tame=1\tamena=1\tame=1\tamena=1\tatʃona=2
246\tlong\tbod̪oŋ=1\tt̪əd̪ənə=2\tjaro=3\tbod̪o=1\taso=4\tbod̪oŋ=1\tbod̪oŋ=1\tbod̪oŋ=1\tjaro=3
247\tshort (length)\tand̪əna=1\tad̪d̪əŋ=1\tad̪d̪əŋ=1\tad̪d̪e=1\tad̪d̪e=1\tand̪əŋ=1\tand̪əŋ=1\tand̪əŋ=1\tad̪əŋ=1
248\theavy\tt̪əbək=1\təjid̪=2\təjid̪=2\td̪abakh=3\təjid̪=2\təjid̪=2\tt̪əbək=1\tt̪əbək=1\taji=2
"""

FIELDS = [
    "Item", "Gloss", "Site_Code", "Site_Name", "PDF_Page", "Printed_Page",
    "Column", "Manual_Transcription", "Source_Cognate_Labels",
    "Review_Status", "Confidence", "Uncertainty", "Reviewer_Method",
    "Reviewed_At", "Reviewer_Declaration",
]


def coordinates(item, code):
    if 219 <= item <= 221:
        return "32", "28", "left"
    if 222 <= item <= 225:
        return "32", "28", "middle"
    if item == 226:
        return ("32", "28", "middle") if code in {"MN", "BR", "RM", "ML", "PL"} else ("32", "28", "right")
    if 227 <= item <= 230:
        return "32", "28", "right"
    if item == 231:
        return ("32", "28", "right") if code in {"MN", "BR", "RM", "ML"} else ("33", "29", "left")
    if 232 <= item <= 235:
        return "33", "29", "left"
    if item == 236:
        return ("33", "29", "left") if code in {"MN", "BR", "RM"} else ("33", "29", "middle")
    if 237 <= item <= 240:
        return "33", "29", "middle"
    if item == 241:
        return ("33", "29", "middle") if code in {"MN", "BR", "RM"} else ("33", "29", "right")
    if 242 <= item <= 245:
        return "33", "29", "right"
    if item == 246:
        return ("33", "29", "right") if code in {"MN", "BR"} else ("34", "30", "left")
    return "34", "30", "left"


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
    assert len(rows) == 30 * 9 == 270
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} manually reviewed cells to {OUT}")


if __name__ == "__main__":
    main()
