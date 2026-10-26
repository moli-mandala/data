"""Exploratory Kota DEDR form/gloss leads; never automatic assignments."""
import collections
import csv
import difflib
import json
import re
import unicodedata
from pathlib import Path

P = Path(__file__).resolve().parent
R = P.parents[2]
records = json.loads((P / "survey-inventory.json").read_text())
assert len(records) == 907
forms = [x for x in csv.DictReader((R / "cldf/forms.csv").open())
         if x["Language_ID"] == "Kota" and x["Source"].startswith("dedr")]
ids = {x["ID"] for x in forms}
parents = collections.defaultdict(list)
for edge in csv.DictReader((R / "cldf/edges.csv").open()):
    if edge["Child_ID"] in ids and edge["Rank"] == "1":
        parents[edge["Child_ID"]].append(edge["Parent_ID"])

trans = str.maketrans({"ɨ": "i", "ṳ": "u", "ɛ": "e", "ɔ": "o", "ʰ": "h", "ʔ": "",
                       "ŋ": "n", "ᵊ": "a", "ṛ": "r", "ḍ": "d", "ṭ": "t", "ṇ": "n",
                       "ḷ": "l", "ś": "s", "š": "s", "ñ": "n", "ɸ": "p"})

def fold(s):
    s = s.lower().translate(trans)
    return re.sub("[^a-z]", "", "".join(c for c in unicodedata.normalize("NFD", s)
                                   if not unicodedata.combining(c)))

stop = {"the", "of", "a", "an", "to", "or", "and", "with", "is", "in", "as", "for",
        "on", "one", "thing", "small", "whole", "my", "he", "she", "it"}

def tokens(s):
    return set(re.findall("[a-z]+", s.lower())) - stop

index = collections.defaultdict(list)
for form in forms:
    for token in tokens(form["Gloss"]):
        index[token].append(form)
out = []
for i, record in enumerate(records, 1):
    pool = {f["ID"]: f for token in tokens(record["Gloss"]) for f in index.get(token, [])}
    target = fold(record["Form"])
    matches = []
    for form in pool.values():
        f = fold(form["Form"])
        if not f or len(f) < 2:
            continue
        score = difflib.SequenceMatcher(None, target, f).ratio()
        if score >= 0.72:
            matches.append([form["Form"], form["Gloss"], parents[form["ID"]], round(score, 3)])
    if matches:
        matches.sort(key=lambda x: (-x[3], len(x[1])))
        out.append({"index": i, "form": record["Form"], "gloss": record["Gloss"],
                    "matches": matches[:5]})
(P / "dedr-kota-form-candidates.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(dict(candidateRecords=len(out), candidateTypes=len({(x["form"], x["gloss"]) for x in out})))
