"""Prepare image-checked Andersen heads, printed variants, and compounds.

Preparation is read-only apart from writing the decision file. Do not rerun after
save: the accepted target assertions deliberately reject already-linked forms.
"""

import csv
import json
from pathlib import Path

D = Path(__file__).resolve().parent
DATA = D.parents[2]
records = json.loads((D / "andersen-records.json").read_text())
by_form = {r["Form"]: r for r in records}
needed = (
    "dāni", "dāṇi", "vacana", "vaḍʰi", "silāṭʰambʰa", "silātʰambʰa",
    "silā", "tʰambʰa", "suvaṁṇagiri", "suvaṁṇa", "giri", "suaga", "svaga",
    "sātireka", "savapaṭʰavi", "asokarāja",
)
assert all(sum(r["Form"] == form for r in records) == 1 for form in needed)

sidecar = DATA / "data/other/forms/etymologies/20260804-andersen-mre.csv"
overlay = {
    row["Form_ID"]: row["Etymon_ID"]
    for row in csv.DictReader(sidecar.open()) if row["Status"] == "accepted"
}
with (DATA / "cldf/edges.csv").open() as handle:
    compiled = {
        row["Child_ID"]: row["Parent_ID"]
        for row in csv.DictReader(handle) if row["Rank"] == "1"
    }
linked = compiled | overlay
for form, parent in (
    ("silā", "12459"), ("tʰambʰa", "13682"),
    ("suvaṁṇa", "13519"), ("giri", "4161"), ("svaga", "13910"),
):
    assert linked.get(by_form[form]["ID"]) == parent, form


def record(form):
    r = by_form[form]
    assert r["ID"] not in linked, form
    return {key: r[key] for key in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}


accepted = []


def simple(form, parent, kind, articles, evidence):
    r = record(form)
    accepted.append({
        "record": r,
        "parent": parent,
        "kind": kind,
        "citation": r["Source"] + ";" + ";".join(f"CDIAL[{n}]" for n in articles),
        "evidence": evidence,
    })


def compound(form, parts, articles, evidence):
    r = record(form)
    parents = [by_form[p]["ID"] for p in parts]
    accepted.append({
        "record": r,
        "parent": parents[0],
        "components": parents,
        "kind": "component",
        "citation": r["Source"] + ";" + ";".join(f"CDIAL[{n}]" for n in articles),
        "evidence": evidence,
    })


simple("dāni", "1567", "reflex", (1567,), "Andersen prints dāni 'now' and compares RV idā́nīm and Pali idāni/dāni, explicitly explaining initial-vowel loss. Turner 1567 section 1 lists Ashokan Rupnath dāni beside the Pali and Prakrit forms.")
simple("dāṇi", by_form["dāni"]["ID"], "variant", (1567,), "The printed dāṇi line says 'see dāni'. It is an inscriptional spelling of the source head newly linked to Turner 1567.")
simple("vacana", "11199", "reflex", (11199,), "Andersen prints vacana 'word' beside RV vacaná and Pali vacana. Turner 11199 explicitly lists Ashokan vacana 'order' from vacaná; the narrower document usage fits the source's instr. vacanena.")
simple("vaḍʰi", "12076", "reflex", (12076,), "Andersen prints vaḍhi 'increase, growth, welfare, profit' beside RV vṛ́ddhi and Pali vaḍḍhi. Turner 12076 explicitly lists Ashokan vaḍhi/vadhi/vuḍhi 'growth' in the vṛ́ddhi article.")
compound("silāṭʰambʰa", ("silā", "tʰambʰa"), (12459, 13682), "Andersen prints silāṭhambha 'stone pillar' with Sanskrit stambha and Pali silātthambha, and explicitly cross-refers to silā and thambha. Turner 12459 lists Ashokan silā 'stone'; 13682 lists Ashokan pillar forms from stambha. Preserve both lexical components in source order.")
simple("silātʰambʰa", by_form["silāṭʰambʰa"]["ID"], "variant", (12459, 13682), "Andersen's separate dental silāthambha line says 'see silāṭhambha'; link it to the source's retroflex compound head, newly assigned both lexical components.")
compound("suvaṁṇagiri", ("suvaṁṇa", "giri"), (13519, 4161), "Andersen prints suvaṁṇagiri 'gold mountain' with Sanskrit suvarṇagiri and cross-references both suvaṁṇa and giri. Turner 13519 gives Pali suvaṇṇa 'gold' and 4161 gives Pali/Prakrit giri 'mountain'; both source heads already have these ancestors.")
simple("suaga", by_form["svaga"]["ID"], "variant", (13910,), "The printed suaga line says 'see svaga'; Andersen's svaga entry lists the Sahasram suag[e] spelling. Turner 13910 explicitly lists Ashokan svaga under svargá; preserve the source-marked graphic variant.")

held = [
    {"record": record("sātireka"), "reason": "Printed comparison and segmentation require distinguishing sātireka from related sādʰika and the etymological atireka; do not attach by meaning alone."},
    {"record": record("savapaṭʰavi"), "reason": "The printed 'entire world' expression has a first component sava without a separate source head; defer decomposition rather than infer an unattested local parent."},
    {"record": record("asokarāja"), "reason": "Its king component is linked, but the Aśoka proper-name source head is not etymologised; hold the compound until that name's source treatment is reviewed."},
]
assert len(accepted) == 8 and len({x["record"]["ID"] for x in accepted}) == 8
assert sum(len(x.get("components", [x["parent"]])) for x in accepted) == 10
assert len(held) == 3
(D / "pass094-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
print("prepared 10 rows across 8 records and 3 holds")
