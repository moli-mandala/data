"""Prepare source-checked Andersen compounds and pronoun variants for pass090.

Preparation is read-only apart from writing decisions; etymology_lab.py owns saves.
Do not rerun after save because the accepted target check will then fail.
"""

import csv
import json
from pathlib import Path

D = Path(__file__).resolve().parent
DATA = D.parents[2]
records = json.loads((D / "andersen-records.json").read_text())
by_form = {r["Form"]: r for r in records}
needed = (
    "aya", "ayaputa", "puta", "amisa", "ammisa", "mātāpitu", "mātā",
    "pitu", "dʰammaguna", "dʰammayuta", "dʰamma", "guna", "yuta",
    "misamdeva", "misa", "deva", "iyam", "ayam", "amisamdevā",
    "mahatata", "apacāy",
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
    ("puta", "8265"), ("mātā", "10016"), ("pitu", "8179"),
    ("dʰamma", "6753"), ("guna", "4190"), ("yuta", "10479"),
    ("misa", "10135"), ("deva", "6523"),
):
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


simple("aya", "1347", "reflex", "1347", "Andersen prints aya 'worthy, honourable' with RV ā́rya and Pali ayya. Turner 1347 section 1 lists Pali ayya 'worthy' and explicitly cites Ashokan aya-puta, confirming this element in the source's following title.")
compound("ayaputa", ("aya", "puta"), ("1347", "8265"), "Andersen prints ayaputa 'prince' beside Sanskrit āryaputra and Pali ayyaputta and cross-refers to aya and puta. Turner 1347 explicitly cites Ashokan aya-puta and 8265 lists Ashokan puta from putrá; preserve both title components in order.")
simple("amisa", "567", "reflex", "567", "Andersen prints amisa 'unmixed, unmingled' with ŚB ámiśra and Pali a-missa. Turner 567 explicitly lists Ashokan amisā nom. pl. under ámiśra.")
simple("ammisa", by_form["amisa"]["ID"], "variant", "567", "The printed ammisa line says 'see amisa' and gives the Sa inscriptional spelling; link it to the source's amisa head, newly linked to Turner 567.")
compound("mātāpitu", ("mātā", "pitu"), ("10016", "8179"), "Andersen prints mātāpitu 'mother and father' beside Sanskrit mātāpitr and Pali mātāpitar, cross-referring to both heads. Turner 10016 explicitly cites Ashokan matapituṣu and 8179 includes Ashokan father forms; preserve ordered kinship components.")
compound("dʰammaguna", ("dʰamma", "guna"), ("6753", "4190"), "Andersen prints dhammaguna 'Dharma-virtue' beside Sanskrit dharma-guṇa and Pali dhamma-guṇa, then cross-refers to both heads. The source's dhamma and guna are already linked to Turner 6753 and the quality section of 4190.")
compound("dʰammayuta", ("dʰamma", "yuta"), ("6753", "10479"), "Andersen prints dhammayuta 'engaged in Dharma' beside Sanskrit dharmayukta and Pali dhamma-yutta, with explicit cross-references to dhamma and yuta. Their source heads already reach Turner 6753 and 10479.")
compound("misamdeva", ("misa", "deva"), ("10135", "6523"), "Andersen prints misamdeva 'god associated (with)' beside Sanskrit miśra-deva and cross-refers to misa and deva. Turner 10135 explicitly cites Ashokan misa and 6523 cites Ashokan devā; retain both lexical components.")
simple("iyam", "587", "reflex", "587", "Andersen prints iyam 'this here' beside RV iyám/ayám/idám and Pali ayaṁ. Turner 587 explicitly lists Ashokan gir. iyaṁ and the related ayaṁ regional forms under the same inflected demonstrative.")
simple("ayam", by_form["iyam"]["ID"], "variant", "587", "Andersen's separate ayam line says 'see iyam'. Turner 587 contains both iyám and ayám and their Ashokan forms, so preserve the printed source cross-reference to the iyam head.")

holds = [
    ("amisamdevā", "Printed Gu/Sa shows fused ammisamdevā while other localities write amisā devā; the source gives no gloss for the composite. Hold the phrase/compound boundary and case agreement for further review rather than assign a single inherited stem."),
    ("mahatata", "Andersen compares Sanskrit mahātmatva and Pali mahantatā for 'greatness'; Turner 9933 mahattva is a related but different formation, and the exact source suffix needs separate analysis."),
    ("apacāy", "The printed source compares ŚB/Pali apacāyati, but no exact Turner entry for that verb was found. Do not attach the broad honour verb to a similar-looking unrelated head."),
]
held = [{"record": record(form), "reason": reason} for form, reason in holds]
assert len(accepted) == 10 and len({x["record"]["ID"] for x in accepted}) == 10
assert sum(len(x.get("components", [x["parent"]])) for x in accepted) == 15
assert len(held) == 3
(D / "pass090-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
print("prepared 15 rows across 10 records and 3 holds")
