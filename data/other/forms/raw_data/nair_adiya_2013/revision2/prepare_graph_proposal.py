"""Source-explicit free/slash alternative graph proposal; no CSV mutation."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV = HERE / "proposed.csv"
FIRST = HERE / "first-reading.jsonl"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

rows = {r[10]: r for r in csv.reader(CSV.open(newline=""))}
units = {r["unit_key"]: r for r in map(json.loads, FIRST.open())}
groups = [
    ("p5:pronoun-paradigm", 5, (6,), "Source prints nii/niiyu as one nominative singular slash group."),
    ("p5:first-person-variation", 1, (2,), "Source expressly states naanu and naa are free variants."),
    ("p5:first-person-variation", 3, (4,), "Source prints naanu/naa poyuḷey as a whole-clause slash group."),
    ("p7:kinship-terms", 1, (2, 3), "Father terms are printed appa/appan/appey in one gloss scope."),
    ("p7:kinship-terms", 6, (7,), "Ascending-generation kin term is a printed slash pair."),
    ("p7:kinship-terms", 8, (9,), "Grandmother term is a printed slash pair."),
    ("p7:kinship-terms", 12, (13,), "Father's elder brother term is a printed slash pair."),
    ("p7:kinship-terms", 14, (15,), "His wife term is a printed slash pair."),
    ("p7:kinship-terms", 16, (17,), "Father's younger brother term is a printed slash pair."),
    ("p7:kinship-terms", 18, (19,), "The separately named cittappan/cittappey option is itself a printed slash pair."),
    ("p7:kinship-terms", 20, (21,), "Elder brother term is a printed slash pair."),
]

proposals = []
for suffix, head, tails, reason in groups:
    unit_key = "nair2013adiya:" + suffix
    unit = units[unit_key]
    head_key = f"{unit_key}:slot{head}"
    assert rows[head_key][0] == "Ravula"
    for tail in tails:
        child_key = f"{unit_key}:slot{tail}"
        assert rows[child_key][0] == "Ravula"
        assert rows[child_key][3] == rows[head_key][3]
        proposals.append({"child_key": child_key, "child_form": rows[child_key][2],
                          "variant_of_key": head_key, "head_form": rows[head_key][2],
                          "source_unit_key": unit_key, "printed_page": unit["printed_page"],
                          "source_reason": reason,
                          "status": "proposed only; original and edge semantics require independent review"})
assert len(proposals) == 12
holds = [
    {"source_unit_key": "nair2013adiya:p7:kinship-terms",
     "slots": [4, 5], "reason": "amma is a reference term, avva an address term; not freely interchangeable."},
    {"source_unit_key": "nair2013adiya:p7:kinship-terms",
     "slots": [16, 18], "reason": "kuḷiyappan and cittappan share a referent but are distinct lexemes; no source variant identity is asserted."},
    {"source_unit_key": "nair2013adiya:p7:kinship-terms",
     "slots": [22, 23], "reason": "akki and akkiya differ in address/reference and extended relation scope."},
    {"source_unit_key": "nair2013adiya:p7:kinship-terms",
     "slots": [27, 28], "reason": "ṛaatti and kuḷilavaḷu have different speaker/reference restrictions."},
    {"source_unit_key": "nair2013adiya:p5:pronoun-paradigm",
     "slots": [14, 19, 22], "reason": "Repeated era occupies distinct feminine/non-feminine/honorific cells; no cross-cell variant edge."},
]
out = {"status": "source-explicit variant proposal; none applied to CSV",
       "csv_sha256": sha(CSV), "first_reading_sha256": sha(FIRST),
       "proposals": proposals, "held_non_edges": holds,
       "borrowed_or_derived_edges": 0}
(HERE / "graph-proposal.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
print(len(proposals),"proposed variant edges,",len(holds),"held classes")
