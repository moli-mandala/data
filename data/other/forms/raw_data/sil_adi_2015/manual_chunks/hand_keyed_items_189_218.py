#!/usr/bin/env python3
"""Write visually checked Adi Appendix B cells for items 189--218."""

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "items_189_218_hand_keyed.tsv"
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
189\tto drink\tt̪unam=1\tt̪unam=1\tt̪unam=1\ttʃaŋma=2\tt̪unam=1\tt̪unam=1\tt̪unam=1\tt̪unam=1\tt̪ũnam=1
190\tto sing\tgoknam=1\tgonam=1\tgãnã=2\tgioŋma=3\tmenmennam=4\tgoŋnam=1\tmirilunam=5\tgoŋnam=1\tbennam=6
191\tto bite\trəknam=3\tgonnam=1\tgamnam=1\tŋonma=2\tgomnam=1\tgannam=1\trəgnam=3\tgamnam=1\tgamnam=1
192\tto laugh\tɲirnam=1\tɲirnam=1\tɲirnam=1\tŋalma=2\tɲirnam=1\tɲirnam=1\tɲirnam=1\tɲirnam=1\tɲirnam=1
193\tto speak\tlunam=1\tlunam=1\tbennam=2\tgaŋma=3\tdʒabnam=4\tlunam=1\tponam=5\tmannam=2\tbennam=2
194\tto tell\tlubinam=3\tlubinam=3\tbenbinam=1\tpoluma=2\tmendʒinam=4\tlubinam=3\tlubinam=3\tt̪ombinam=1\tbenbinam=1
195\tto know\tkennam=1\tkinnam=1\ttʃennam=2\thuːma=3\ttʃennam=2\tkennam=1\tkennam=1\tkennam=1\ttʃennam=2
196\tto forget\tkensikumanam=5\tmit̪pennam=1\tmũpõŋkumanam=3\tmiaŋpok=4\tmuuppennam=1\tmit̪pennam=1\tkenʃijimanam=5\tmuuŋokhinam=2\tmit̪pennam=1
197\tto sleep\tjupnam=2\tjunnam=2\tjupnam=2\tuumma=3\tjupnam=2\tjunnam=2\tipnam=1\timnam=4\tjupnam=2
198\tto dream\tjummamanam=2\tjummaŋmanam=2\tjupmõŋmanam=2\tuumma=1\tjupmamanam=2\tjummamanam=2\timmaŋmanam=2\timmaŋmanam=2\tjupmõŋmanam=2
199\tto do/make\tinam=1\tinam=1\tinam=1\tluoma=2\tinam=1\tinam=1\tinam=1\tinam=1\tinam=1
200\tto work\tagerinam=3\tagergernam=3\tleginam=2\tagerluma=3\taŋoinam=4\tagerinam=3\tagerinam=3\tagerinam=3\tinam=1
201\tto play\timannam=2\timennam=2\tsomennam=2\tkelima=3\tsomennam=2\timennam=2\timannam=2\timannam=2\tsõnam=1
202\tto dance\tnit̪ommonam=6\tponuŋmonam=1\tponumonam=1\tmad̪arluma=2\tsomennam=4\tsomennam=4\tpaksomonam=3\tponuŋmonam=1\tnaʃinam=5
203\tto throw\tərʔaknam=1\tjoppaknam=2\tnanam=3\tjurma=4\tərpanam=1\tjoppaknam=2\tərpaknam | ərpaknam=1 | 2\tbjarnam=5\tornam=6
204\tto lift\tdʒoŋonnam=1\tlatʃaŋnam=5\td̪eiennam=2\tdʒojaphma=1\tdʒorəpnam=1\tlarepnam=3\tdʒoŋonnam=1\tdʒonam=1\tʃeːnam=4
205\tto push\tnuunam=2\tnuunam=2\tnumanam=4\tnamma=1\tnuunam=2\tnuupaknam=3\tnuunam=2\tnuunam=2\tnũnnam=5
206\tto pull\thonam=1\tʃonam=1\ttʃẽt̪unam=3\tsehma=2\tʃonam=1\tʃonam=1\tʃonam=1\thonam=1\tʃenam=1
207\tto tie\trunnam=1\tjenam=3\tt̪aʔpanam=4\tjama=5\tpuʃumnam=6\tjənam=3\teejnam=2\trunnam=1\tpũnam=1
208\tto wipe\tt̪ud̪bunnam=2\tt̪ipbinnam=2\tt̪ikkhanam=1\tpekkut̪ma=4\tt̪ippaknam=3\tt̪ud̪bunnam=2\tɲot̪bunnam=5\tt̪ud̪bunnam=2\tt̪it̪kaknam=3
209\tto weave (on loom)\thumnam=1\ttʃunnam=1\tuʃumsumnam=1\ttʃimma=2\tʃumnam=1\ttʃunnam=1\tʃumnam=1\thumnam=1\ttʃumnam=1
210\tto sew\tomnam=1\thonnam | honnam=1 | 2\thomnam=2\thomma=2\tamnam=1\tonnam=1\tomnam=1\tomnam=1\thõmnam=2
211\tto wash\təd̪bunnam=1\thuurbinnam=2\tnikʔkhapha=3\tpamma=4\tuurkaknam=5\tnunam=6\təd̪bunnam=1\təd̪bunnam=1\tuurkaknam=5
212\tto take bath\tuurhuunam=1\tmosusunam=3\tiʃunam=1\thamtʃuma=2\tiʃisunam=1\tuurʃunam=1\tuuʃunam=1\tuurʃinam=1\thuurʃunam=1
213\tto cut something\tlonnam=2\tpeːnam=3\tpeːnam=3\tpima=1\tganam=4\tgannam=4\tlot̪nam=2\tgannam=4\tpeːnam=3
214\tto burn\tromnam=1\tronnam=1\tromnam=1\tgiuma=2\tramnam=1\tromnam=1\tromnam=1\tparnam=3\trumnam=1
215\tto buy\trənam=1\trənam=1\trənam=1\tdʒaŋma=2\trənam=1\trənam=1\trənam=1\trənam=1\trənam=1
216\tto sell\tkonam=2\tkonam=2\tphuʔʃenam=3\tkuma=1\tpuknam=4\tkonam=2\tkonam=2\tkonam=2\tpuknam=4
217\tto steal\td̪oʔionam=1\td̪otʃonam=1\td̪opioŋnam=1\tt̪iuma=3\td̪otʃonam=1\tpjonam=2\tpjonam=2\tpjonam=2\td̪opioŋnam=1
218\tto lie, fib\tmənam=1\tmənam=1\tmenam=1\tjat̪ma=2\tmenam=1\tmənam=1\tjad̪nam=3\tjad̪nam=3\tmənam=1
"""

FIELDS = [
    "Item", "Gloss", "Site_Code", "Site_Name", "PDF_Page", "Printed_Page",
    "Column", "Manual_Transcription", "Source_Cognate_Labels",
    "Review_Status", "Confidence", "Uncertainty", "Reviewer_Method",
    "Reviewed_At", "Reviewer_Declaration",
]


def coordinates(item, code):
    if 189 <= item <= 191:
        return "30", "26", "left"
    if item == 192:
        return ("30", "26", "left") if code in {"MN", "BR"} else ("30", "26", "middle")
    if 193 <= item <= 196:
        return "30", "26", "middle"
    if item == 197:
        return ("30", "26", "middle") if code in {"MN", "BR"} else ("30", "26", "right")
    if 198 <= item <= 201:
        return "30", "26", "right"
    if item == 202:
        return ("30", "26", "right") if code in {"MN", "BR"} else ("31", "27", "left")
    if 203 <= item <= 206:
        return "31", "27", "left"
    if item == 207:
        return ("31", "27", "left") if code == "MN" else ("31", "27", "middle")
    if 208 <= item <= 211:
        return "31", "27", "middle"
    if 212 <= item <= 216:
        return "31", "27", "right"
    return "32", "28", "left"


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
