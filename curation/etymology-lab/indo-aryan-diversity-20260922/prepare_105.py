"""Prepare image-checked Andersen variants and a transparent compound.

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
    "ciraṭʰitika", "cira", "tʰitika", "caka", "saka", "cʰavacʰara",
    "samvacʰara", "sāvāna", "sāvana", "jānapada", "janapada",
    "udala", "udālaka", "hīyam",
)
assert all(sum(r["Form"] == form for r in records) == 1 for form in needed if form != "saka")
assert sum(r["Form"] == "saka" for r in records) == 2
by_form["saka"] = next(r for r in records if r["Form"] == "saka" and r["Gloss"] == "able, possible")
sidecar = DATA / "data/other/forms/etymologies/20260804-andersen-mre.csv"
overlay = {row["Form_ID"]: row["Etymon_ID"] for row in csv.DictReader(sidecar.open()) if row["Status"] == "accepted"}
compiled = {row["Child_ID"]: row["Parent_ID"] for row in csv.DictReader((DATA / "cldf/edges.csv").open()) if row["Rank"] == "1"}
linked = compiled | overlay
for form, parent in (
    ("cira", "4824"), ("tʰitika", "13770"), ("saka", "12253"),
    ("samvacʰara", "13011"), ("sāvana", "12698"), ("janapada", "5101"),
):
    assert linked.get(by_form[form]["ID"]) == parent, form


def record(form):
    r = by_form[form]
    assert r["ID"] not in linked, form
    return {key: r[key] for key in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}


accepted = []


def relation(form, parent_form, kind, articles, evidence):
    r = record(form)
    accepted.append({
        "record": r,
        "parent": by_form[parent_form]["ID"],
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


compound("ciraṭʰitika", ("cira", "tʰitika"), (4824, 13770), "Andersen prints ciraṭhitika 'perpetual, lasting long' beside Sanskrit cirasthitika and Pali ciratthitika, explicitly cross-referring to cira and thitika. Turner 4824 documents cira in Ashokan compounds and 13770 explicitly lists Ashokan cira/cila-thitika forms. Preserve both ordered lexical components.")
relation("caka", "saka", "variant", (12253,), "Andersen's printed caka line says 'see saka'. The source's adjective saka 'able, possible' is linked to Turner 12253; the separate Buddhist noun saka has a different ID and is excluded.")
relation("cʰavacʰara", "samvacʰara", "variant", (13011,), "Andersen's printed chavachara line says 'see saṃvachara'. The year head is the linked source samvacʰara, from Turner 13011 saṁvatsará; preserve the documented Rupnath spelling as a source variant.")
relation("sāvāna", "sāvana", "variant", (12698,), "Andersen's printed sāvāna line says 'see sāvana' for the Sahasram spelling. The source head sāvana 'announcement, edict' is already linked to Turner 12698 śrāvaṇa 'announcing'.")
relation("jānapada", "janapada", "derived", (5101,), "Andersen prints jānapada 'belonging to the country' beside ŚB jānapada and Pali janapada, cross-referring to the source's janapada 'country'. Turner 5101 explicitly distinguishes Ashokan Kalsi janapade from Girnar jānapadasa 'living in the country' < jānapadá. Record the adjective's transparent derivation from the already-linked same-source noun rather than collapse the nominal and adjectival formations.")

held = [
    {"record": record("udala"), "reason": "The scan clearly prints retroflex uḍala while the installed source form is dental udala; despite Turner's exact uḍāla 1935, hold until this source transcription mismatch is adjudicated."},
    {"record": record("udālaka"), "reason": "The scan prints uḍālaka but the installed form has dental udālaka. Turner 1935 explicitly gives uḍālake, yet the transcription discrepancy warrants review before linking."},
    {"record": record("hīyam"), "reason": "The printed line cross-refers to both hi and iyam; it needs a particle/pronoun boundary decision rather than a single automatic variant edge."},
]
assert len(accepted) == 5 and len({x["record"]["ID"] for x in accepted}) == 5
assert sum(len(x.get("components", [x["parent"]])) for x in accepted) == 6
assert len(held) == 3
(D / "pass105-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
print("prepared 6 rows across 5 records and 3 holds")
