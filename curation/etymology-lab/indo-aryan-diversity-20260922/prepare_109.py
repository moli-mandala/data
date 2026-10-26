"""Prepare four source-checked Andersen links, leaving suspected contamination aside.

Do not rerun after save: accepted targets are required to be unlinked.
"""

import csv
import json
from pathlib import Path

D = Path(__file__).resolve().parent
DATA = D.parents[2]
records = json.loads((D / "andersen-records.json").read_text())
by_form = {r["Form"]: r for r in records}
needed = (
    "yādisa", "hacʰāroha", "hatʰiyāroha", "hatʰāroha", "hadʰa",
    "hida", "budʰaśake", "śaka", "bumhanāni", "bamhana",
)
assert all(sum(r["Form"] == form for r in records) == 1 for form in needed)
sidecar = DATA / "data/other/forms/etymologies/20260804-andersen-mre.csv"
overlay = {row["Form_ID"]: row["Etymon_ID"] for row in csv.DictReader(sidecar.open()) if row["Status"] == "accepted"}
compiled = {row["Child_ID"]: row["Parent_ID"] for row in csv.DictReader((DATA / "cldf/edges.csv").open()) if row["Rank"] == "1"}
linked = compiled | overlay
for form, parent in (("hatʰāroha", "14047"), ("hida", "1605"), ("bamhana", "9327")):
    assert linked.get(by_form[form]["ID"]) == parent, form


def record(form):
    r = by_form[form]
    assert r["ID"] not in linked, form
    return {key: r[key] for key in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}


accepted = []


def simple(form, parent, kind, article, evidence):
    r = record(form)
    accepted.append({
        "record": r,
        "parent": parent,
        "kind": kind,
        "citation": f"{r['Source']};CDIAL[{article}]",
        "evidence": evidence,
    })


simple("yādisa", "10458", "reflex", "10458", "Andersen prints yādisa 'which like, what like, whichever' beside RV yādṛś and Pali yādisa; the continuation lists regional yārisa. Turner 10458 derives this family from yādrśa and explicitly lists Pali yādisa and Ashokan yadiśaṁ/ādise/yārise.")
simple("hacʰāroha", by_form["hatʰāroha"]["ID"], "variant", "14047", "Andersen's printed hachāroha line says 'see hatthāroha' and the head includes its Nittur inflection. The source head is already linked to Turner 14047 hastyārōha 'elephant rider'.")
simple("hatʰiyāroha", by_form["hatʰāroha"]["ID"], "variant", "14047", "Andersen's printed hathiyāroha line says 'see hatthāroha' and the head includes its Erragudi inflection. Preserve the source-marked spelling variant of the already-linked elephant-driver head.")
simple("hadʰa", by_form["hida"]["ID"], "variant", "1605", "Andersen's printed hadha line says 'see hida' for the Rupnath spelling. Turner 1605 lists Ashokan hida under ihá 'here'; retain the source's explicit cross-reference rather than infer a separate etymon.")

held = [
    {"record": record("budʰaśake"), "reason": "Andersen says the attested form is 'for upāsake'; a visual Buddha+saka compound would conflict with that textual correction and the Buddhist saka homograph's suspect compiled ancestry."},
    {"record": record("śaka"), "reason": "The printed line only says 'cf. budhaśaka'; it does not identify this record with the adjective śakya or resolve the Buddhist/upāsaka reading."},
    {"record": record("bumhanāni"), "reason": "The printed line says 'see bambhana', but the installed head is bamhana. Hold this cross-reference until the b/m spelling or OCR discrepancy is reviewed."},
]
assert len(accepted) == 4 and len({x["record"]["ID"] for x in accepted}) == 4
assert len(held) == 3
(D / "pass109-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
print("prepared 4 rows across 4 records and 3 holds")
