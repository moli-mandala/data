#!/usr/bin/env python3
"""Write visually checked Adi Appendix B cells for items 249--278."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_249_278_hand_keyed.tsv"
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
# The sole source blank uses the nonlexical generator marker ∅=0; it is emitted
# with an empty Manual_Transcription, never as a form.
RAW = """\
249\tlight (not heavy)\tət̪saŋ=1\təʃʃaŋ=1\thõdʒup=2\toh=3\tɛʃʃa=1\təʃʃaŋ=1\tət̪saŋ=1\tət̪saŋ=1\thõdʒuk=2
250\tfat\tdʒuunam=1\tdʒuunam=1\tpõt̪ə=2\tlẽma=3\tpõt̪ə=2\tdʒuunam=1\tŋəlnam=4\tdʒuunam=1\tpõt̪ə=2
251\tthin\thaŋt̪ar=1\thome=2\tpoŋɲi=3\tsoma | so ma=2 | 4\tpõŋa=3\tʃona=4\tʃoŋkək=5\thaŋkuu=1\thõme=2
252\twide, broad\tbort̪ak=1\tat̪ak=2\tt̪akt̪ə=3\tat̪ak=2\tt̪akt̪ə=3\tat̪ak=2\tbort̪ak=1\tat̪ak=2\tt̪akt̪ə=3
253\tnarrow\tadʒok=1\tat̪akt̪ame=2\tt̪akt̪ʃona=3\tadʒok=1\tadʒok=1\tadʒok=1\tadʒok=1\tadʒok=1\tt̪akme=4
254\tdeep\toruuŋ=1\turuuŋ=1\tərupona=2\thaluum=3\taruu=1\turuuŋ=1\təruu=1\taruu=1\taruuŋ=1
255\tshallow\toruumana=1\turumanə=1\təruumo=2\tʃamar=3\taruumana=1\turumanə=1\təmək=4\taruumana=1\tbetʃõr=5
256\tfull\tbiːnam=1\tbiːnam=1\tjuurkuud̪o=2\tbaŋɲuu=3\tdʒupok=4\tbiːna=1\tbuna=1\tbuna=1\tjuurkuuk=2
257\tempty\taruk=1\taruk=1\td̪oma=2\ttʃuŋə̃=3\taro=1\tməraŋ=5\tkamaŋ=6\tbumana=4\tarom=1
258\thungry\tkənoŋ=1\tkənoŋ=1\tkinnoŋ=1\tbanu=2\tkəno=1\tkənoŋ=1\tkəno=1\tkənoŋ=1\tkinno=1
259\tthirsty\tt̪uuluŋ=1\tt̪umuŋ=1\tt̪ũnũŋ=1\ttʃaŋmi=2\thapuu=3\tt̪uuluŋ=1\tt̪uuluŋ=1\tt̪uuluŋ=1\thõpũ=3
260\tsweet\tt̪inam=1\tt̪ipo=2\tt̪ipo=2\tauuu=3\tt̪ipo=2\tt̪ipo=2\tt̪unam=1\tkunam=4\tt̪ipo=2
261\tsour\tkuna=1\tkutʃuk=2\tlika=3\tahar=4\tkutʃuk=2\tkutʃi=2\tkuna=1\tkuna=1\tkutʃuk=2
262\tbitter\tkoʃaŋ=1\tkotʃak=1\tkatʃakh=1\taka=2\tkatʃakh=1\tkotʃaŋ=1\tkoʃaŋ=1\tkoʃaŋ=1\tkatʃakh=1
263\tspicy, hot\thauna=1\tau=1\təd̪uuk=3\tamar=2\tau=1\tau=1\tmannam=2\tmarnam=2\tad̪uuk=3
264\tripe\tminna | minna=1 | 2\tminne | minne=1 | 2\tminna | minna=1 | 2\tmaŋma=2\tɲinne=1\tminne | minne=1 | 2\tminna | minna=1 | 2\tminna | minna=1 | 2\tminne | minne=1 | 2
265\trotten (fruit)\tjana=1\tjane=1\tjane=1\tkaŋ=2\tjane=1\tjane=1\tjana=1\tjana=1\tjãnə̃=1
266\tfast\td̪ugd̪ugna=1\tæmpə=2\tenukh=3\ttʃakon=4\tanuk=3\tlagan=5\tbat̪t̪uuk | be gi=6 | 8\tmənaŋ=7\tanuk=3
267\tslow\tət̪ət̪=1\thət̪ət̪=1\tkole=2\tad̪ol=3\thəruu=4\tət̪ət̪=1\tad̪ol=3\təːt̪ət̪=1\tkole=2
268\tsame\tnəjuuhuuna=1\tləkon=2\takhend̪a=3\takham=4\takənd̪əd̪ə=3\takjam=4\takham=4\takjam=4\thə̃guŋ=5
269\tdifferent\taŋuna=1\taŋu=1\təŋud̪a=1\taŋutʃu=1\taŋoaŋo=1\taŋu=1\taŋu=1\taŋu=1\taŋõ=1
270\tdry\tput̪urna=1\tput̪urna=1\tʃəmpuu=2\tʃaɲi=3\tput̪uur=1\tput̪urna=1\tput̪ul=1\tput̪ul=1\tput̪urna | ʃe nnə=1 | 3
271\twet\tdʒunam=3\taʃikane=2\tdʒudʒaŋ=3\ttʃuɲi=4\tdʒudʒa=3\tjapiom=5\tjapiom=5\taʃinam=1\tdʒũdʒaŋ=3
272\thot\tgunam=1\tgutʃi=1\tɛgu=1\takal=2\tagu=1\tukkhi=3\tpamki=4\tgunam=1\tagu=1
273\tcold\trəjuŋ=1\trəjuŋ=1\thuutʃuur=2\tantʃiŋ=3\theŋɲik=4\tanʃuuŋ=3\tanʃuuŋ=3\tantʃiŋ=3\theŋjik=4
274\tgood\tajnam=1\thajna=1\tpod̪a=2\tajid̪ɲi=3\tajd̪u | aj d̪u=1 | 3\tajnam=1\tajna=1\tajnam=1\tpone=2
275\tbad\tajmana | t̪arukmana=1 | 4\thajmanə=1\tpomoŋ=2\tajid̪ŋaɲi=3\tajma=1\tajmana=1\tloruumana=4\tmalanam=5\tkaru=6
276\tnew\tanuu=1\tanuu=1\tnit̪i=2\tanũl=1\tani=1\tanu=1\tanuu=1\tanuu=1\tnit̪i=2
277\told\taku=1\taku=1\t∅=0\taku=1\taku=1\taku=1\taku=1\taku=1\tamen=2
278\tbroken\tbənna | d̪uurne=1 | 2\tbənna=1\td̪uurtu=2\tborma=3\tbərt̪ak=4\td̪uurnə=2\tbət̪na=5\tbənna=1\td̪uurne=2
"""

FIELDS = [
    "Item", "Gloss", "Site_Code", "Site_Name", "PDF_Page", "Printed_Page",
    "Column", "Manual_Transcription", "Source_Cognate_Labels",
    "Review_Status", "Confidence", "Uncertainty", "Reviewer_Method",
    "Reviewed_At", "Reviewer_Declaration",
]


def coordinates(item, code):
    if 249 <= item <= 250:
        return "34", "30", "left"
    if item == 251:
        return ("34", "30", "left") if code in {"MN", "BR"} else ("34", "30", "middle")
    if 252 <= item <= 255:
        return "34", "30", "middle"
    if item == 256:
        return ("34", "30", "middle") if code == "MN" else ("34", "30", "right")
    if 257 <= item <= 260:
        return "34", "30", "right"
    if item == 261:
        return ("34", "30", "right") if code == "MN" else ("35", "31", "left")
    if 262 <= item <= 264:
        return "35", "31", "left"
    if item == 265:
        return ("35", "31", "left") if code in {"MN", "BR", "RM", "ML", "PL"} else ("35", "31", "middle")
    if 266 <= item <= 269:
        return "35", "31", "middle"
    if item == 270:
        return ("35", "31", "middle") if code in {"MN", "BR", "RM", "ML"} else ("35", "31", "right")
    if 271 <= item <= 274:
        return "35", "31", "right"
    if item == 275:
        return ("35", "31", "right") if code in {"MN", "BR"} else ("36", "32", "left")
    return "36", "32", "left"


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
            blank = form == "∅"
            pdf_page, printed_page, column = coordinates(item, code)
            row = {
                "Item": str(item), "Gloss": gloss, "Site_Code": code,
                "Site_Name": name, "PDF_Page": pdf_page,
                "Printed_Page": printed_page, "Column": column,
                "Manual_Transcription": "" if blank else form,
                "Source_Cognate_Labels": labels,
                "Review_Status": "source_blank" if blank else "attested",
                "Confidence": "high",
                "Uncertainty": (
                    "Source prints cognate label 0 and ‘no entry’." if blank else ""
                ),
                "Reviewer_Method": METHOD, "Reviewed_At": "2026-08-28",
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
