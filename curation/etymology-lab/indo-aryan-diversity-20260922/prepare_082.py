"""Prepare the source-checked Andersen glossary decisions for pass082.

This is research preparation only. Saving is delegated to etymology_lab.py.
"""

import csv
import json
from pathlib import Path

D = Path(__file__).resolve().parent
records = json.loads((D / "andersen-records.json").read_text())
by_form = {r["Form"]: r for r in records}
assert all(sum(r["Form"] == form for r in records) == 1 for form in (
    "garu", "guru", "guna", "guṇa", "tiya", "di", "budʰa",
    "manusa", "māṇusa", "mānusa", "saca", "mahalaka", "bamhana",
    "maha", "munisa", "tatʰā", "likʰ",
    "raṭʰika", "rajuka",
))
sidecar = D.parents[2] / "data/other/forms/etymologies/20260804-andersen-mre.csv"
linked = {
    row["Form_ID"]
    for row in csv.DictReader(sidecar.open())
    if row["Status"] == "accepted"
}
with (D.parents[2] / "cldf/edges.csv").open() as handle:
    compiled = {
        row["Child_ID"]: row["Parent_ID"]
        for row in csv.DictReader(handle) if row["Rank"] == "1"
    }
    linked.update(compiled)
for form, parent in (("garu", "4209"), ("guna", "4190"), ("manusa", "9828")):
    source_id = by_form[form]["ID"]
    assert compiled.get(source_id) == parent, form


def record(form):
    r = by_form[form]
    assert r["ID"] not in linked, form
    return {key: r[key] for key in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}


proposals = [
    ("guru", by_form["garu"]["ID"], "variant", "The printed guru line says 'see garu'; preserve that source-defined spelling relation to its garu head, which is already linked to Turner 4209."),
    ("guṇa", by_form["guna"]["ID"], "variant", "The printed guṇa line says 'see guna'. This is a source cross-reference to the separately recorded virtue head, already linked to Turner 4190."),
    ("tiya", "5912", "reflex", "Andersen compares tiya 'third' with RV tr̥tī́ya and Pali tatiya. Turner 5912 section 1 includes Pk tiiya and other reduced third-ordinal forms."),
    ("di", "6648", "reflex", "Andersen's di 'two' compares RV dvi and Pali dvi/di, with duve in the inscription. Turner 6648 explicitly cites Ashokan duve among the continuations of dva 'two'."),
    ("budʰa", "9276", "reflex", "Andersen prints budha 'Buddha' and compares RV buddhá, Pali buddha. Turner 9276 section 1 names the Buddha and explicitly cites Ashokan Budhe."),
    ("māṇusa", "10049", "reflex", "The separate māṇusa 'man' head compares RV mā́nuṣa and Pali mānusa. Turner 10049 explicitly gives Ashokan mānusa and Pk māṇusa, separating it from shorter mánuṣa 9827."),
    ("mānusa", by_form["māṇusa"]["ID"], "variant", "The printed mānusa line says 'see māṇusa'; link the source's cross-reference to its already identified māṇusa head under Turner 10049."),
    ("saca", "13112", "reflex", "Andersen prints saca 'truth' beside RV satyá and Pali sacca. Turner 13112 explicitly has Ashokan saca 'truth' under satyá."),
    ("mahalaka", "9935", "reflex", "Andersen calls mahalaka 'noble, old' and compares Pali mahallaka; Turner 9935 explicitly gives Ashokan mahalaka/mahālaka 'great, old'."),
    ("bamhana", "9327", "reflex", "Andersen prints bamhana 'Brahman' beside RV brāhmaṇá. Turner 9327 cites Ashokan bamaṇa/baṁbhana/bāmhaṇa regional forms under brāhmaṇá section 1."),
    ("raṭʰika", "10724", "reflex", "Andersen prints raṭhika for an official of the kingdom and compares Skt rāṣṭrika, Pali raṭṭhika. Turner 10724 gives rāṣṭrika 'governor' and Pali raṭṭhika under that exact branch."),
]

accepted = []
for form, parent, kind, evidence in proposals:
    citation_parent = {
        "guru": "4209",
        "guṇa": "4190",
        "mānusa": "10049",
    }.get(form, parent)
    r = record(form)
    accepted.append(
        {
            "record": r,
            "parent": parent,
            "kind": kind,
            "citation": f"{r['Source']};CDIAL[{citation_parent}]",
            "evidence": evidence,
        }
    )

holds = [
    ("maha", "Turner 9946 is mahānt, but its full article does not explicitly give the source's independent maha head; avoid equating a reduced adjective with that specific stem without fuller analysis."),
    ("tatʰā", "The source clearly compares RV/Pali tathā 'thus', but the local Turner inventory lacks an independent tathā head; do not attach to the later compound tathāpi 5647."),
    ("likʰ", "The glossary is a broad write-root, while Turner 11048 is a particular likháti formation; hold rather than flatten verbal morphology into that finite head."),
    ("rajuka", "The installed p. 164 record is only a 'see rājuka' cross-reference; the actual rājuka head begins on the following printed page but has no distinct installed record. Hold rather than flatten this source-boundary gap."),
]
held = [{"record": record(form), "reason": reason} for form, reason in holds]

out = {"accepted": accepted, "held": held}
(D / "pass082-decisions.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
assert len(accepted) == 11 and len({a["record"]["ID"] for a in accepted}) == 11
assert len(held) == 4
print("prepared 11 assignments and 4 holds")
