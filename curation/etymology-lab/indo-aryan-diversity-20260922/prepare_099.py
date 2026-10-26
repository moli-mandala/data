"""Prepare source-checked Andersen lexical heads and explicit printed variants.

Do not rerun after save: assertions require all targets to remain unlinked.
"""

import csv
import json
from pathlib import Path

D = Path(__file__).resolve().parent
DATA = D.parents[2]
records = json.loads((D / "andersen-records.json").read_text())
by_form = {r["Form"]: r for r in records}
needed = (
    "kata", "kaṭa", "kitā", "no", "pakiti", "pakati", "saka",
    "nātika", "dīgʰāvus", "śaka", "kiti", "kiṇti",
)
assert all(sum(r["Form"] == form for r in records) == 1 for form in needed if form != "saka")
assert sum(r["Form"] == "saka" for r in records) == 2
by_form["saka"] = next(r for r in records if r["Form"] == "saka" and r["Gloss"] == "able, possible")
sidecar = DATA / "data/other/forms/etymologies/20260804-andersen-mre.csv"
overlay = {row["Form_ID"] for row in csv.DictReader(sidecar.open()) if row["Status"] == "accepted"}
compiled = {row["Child_ID"] for row in csv.DictReader((DATA / "cldf/edges.csv").open()) if row["Rank"] == "1"}
linked = overlay | compiled


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


simple("kata", "2814", "reflex", "2814", "Andersen prints kata 'made' beside RV kṛtá and Pali kata, including the Rupnath and other regional forms. Turner 2814's kṛtá subentry explicitly lists Girnar kata and other Ashokan kaṭa, kiṭa forms; select the participial subentry, not the article's finite karóti head.")
simple("kaṭa", by_form["kata"]["ID"], "variant", "2814", "The printed kaṭa line says 'see kata'. Turner 2814 explicitly lists Ashokan kaṭa alongside Girnar kata in its kṛtá participial subentry.")
simple("kitā", by_form["kata"]["ID"], "variant", "2814", "The printed kitā line says 'see kata' and Andersen's kata article records Panguraria kitā. Turner 2814 includes Ashokan northwestern kiṭa in the same kṛtá participial subentry.")
simple("no", "7607", "reflex", "7607", "Andersen prints no 'nor, but not' beside RV nṓ and Pali no, explicitly analysing Sanskrit no < na-u. Turner 7607 derives nṓ from ná + u and directly lists Ashokan no 'not'.")
simple("pakiti", "8446", "reflex", "8446", "Andersen prints pakiti 'rule, usage' beside GB prakṛti and Pali pakati. Turner 8446 directly lists Ashokan pakiti 'rule, standard' under prakṛti.")
simple("pakati", by_form["pakiti"]["ID"], "variant", "8446", "The printed pakati line says 'see pakiti'; its Udegolam spelling belongs to the newly linked rule/usage head, matching Turner's Pali pakati comparison.")
simple("saka", "12253", "reflex", "12253", "Andersen prints the adjective saka 'able, possible' beside Sanskrit śakya and Pali sakka. Turner 12253 directly lists Ashokan Girnar saka, Shahbazgarhi śaka, and Jaugada sakiya under śakya 'possible'. The distinct printed Buddhist noun saka is a separately installed and already-linked source record.")
simple("nātika", "5276", "reflex", "5276", "Andersen prints nātika 'relative' beside Sanskrit jñātika and Pali ñāti-ka. Turner 5276 under jñātí explicitly includes Ashokan Girnar ñātika and Kalsi nātikye, supporting the -ka relative form.")
simple("dīgʰāvus", "6371", "reflex", "6371", "Andersen prints dīghāvus 'long life' beside RV dīrghā́yus and Pali dīghāyu/dīghāvu, analysing the attested ending as dative. Turner 6371 directly cites Ashokan dīghāvuse, either nominative 'long-lived' or dative 'for a long life'; preserve that inflectional ambiguity.")

held = [
    {"record": record("śaka"), "reason": "The printed śaka line refers to budhaśaka 'Buddhist', whereas the separate saka adjective is 'possible'; despite Turner's regional śaka adjective, these source senses must not be conflated."},
    {"record": record("kiti"), "reason": "The printed line says 'for kiṃti', but the installed kiṇti head needs its compound particle analysis before a source variant edge can safely be attached."},
    {"record": record("kiṇti"), "reason": "Andersen compares kim + iti 'how then'; no exact Turner head was identified for this combined particle, so hold rather than attach a broad interrogative."},
]
assert len(accepted) == 9 and len({x["record"]["ID"] for x in accepted}) == 9
assert len(held) == 3
(D / "pass099-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
print("prepared 9 rows across 9 records and 3 holds")
