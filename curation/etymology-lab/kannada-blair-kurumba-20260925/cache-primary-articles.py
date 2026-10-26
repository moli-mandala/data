"""Cache full primary DEDR articles for exploratory Blair cross-pass candidates."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P.parents[2]
candidates = json.loads((P / "cross-pass-candidates.json").read_text())
ids = {row["parent"] for row in candidates}
ids.update({
    "d386", "d1224", "d3683", "d5538", "d4616", "d4157", "d1407",
    "d4649", "d5486", "d4626", "d5155", "d2902", "d1584", "d1871",
    "d3486", "d3309", "d3687", "d4620", "d1873", "d1911", "d1884",
    "d4715", "d1249", "d2680", "d1114", "d3268",
    "d3986", "d4345", "d4039", "d4494",
    "d4303", "d5224",
})
articles = {}
with gzip.open(DATA / "data/dedr/entry-texts-audit.csv.gz", "rt") as source:
    for row in csv.DictReader(source):
        if row["Entry_ID"] in ids:
            articles[row["Entry_ID"]] = {
                key: row[key] for key in ("Page", "Citation", "Raw_Article")
            }
            articles[row["Entry_ID"]]["URL"] = (
                "https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page=" + row["Page"]
            )
assert set(articles) == ids, sorted(ids - set(articles))
(P / "candidate-article-evidence.json").write_text(
    json.dumps(articles, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps({"articles": len(articles), "candidates": len(candidates)}))
