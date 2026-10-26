"""Prepare image-checked Andersen adverb, pronoun, and numeral forms.

Do not rerun after save: accepted targets must still be unlinked.
"""

import csv
import json
from pathlib import Path

D = Path(__file__).resolve().parent
DATA = D.parents[2]
records = json.loads((D / "andersen-records.json").read_text())
by_form = {r["Form"]: r for r in records}
needed = ("tatʰā", "tatʰa", "ve", "sata", "vā", "va", "tu", "heta")
assert all(sum(r["Form"] == form for r in records) == 1 for form in needed)
sidecar = DATA / "data/other/forms/etymologies/20260804-andersen-mre.csv"
overlay = {row["Form_ID"] for row in csv.DictReader(sidecar.open()) if row["Status"] == "accepted"}
compiled = {row["Child_ID"] for row in csv.DictReader((DATA / "cldf/edges.csv").open()) if row["Rank"] == "1"}
linked = compiled | overlay


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


simple("tatʰā", "5646", "reflex", "5646", "Andersen prints tathā 'so, thus' beside RV/Pali tathā and regionally varying tatha/tathā forms. Turner 5646 táthā directly gives Pali tathā and Ashokan tatha with variable final length; preserve this attested adverb as its own source head.")
simple("tatʰa", by_form["tatʰā"]["ID"], "variant", "5646", "The Nittur line on the preceding scan page says 'see tatha' and is the source's unglossed counterpart to the 'so, thus' head. Turner 5646 explicitly notes Ashokan variable final vowel length.")
simple("ve", "11183", "reflex", "11183", "Andersen's separate ve line says '(Skt vaḥ) see tu-' and its pronoun paradigm labels ve accusative/dative plural. Turner 11183 vaḥ is the enclitic second-person plural pronoun and explicitly lists Ashokan ve dative.")
simple("sata", "12278", "reflex", "12278", "Andersen's separately bounded printed sata entry is glossed 'hundred', compares RV śatá and Pali sata, and cites Rupnath. Turner 12278 śatá '100' lists Pali sata and Ashokan śata/ṣata regional series; direct scan inspection resolves the earlier cached-OCR boundary concern.")

held = [
    {"record": record("vā"), "reason": "The printed vā line says 'see va (= Skt eva)', but the installed va record merges the eva particle and a distinct wish/choose verb; hold the source variant until that homograph boundary is resolved."},
    {"record": record("tu"), "reason": "The installed tu record merges the printed 'but' particle and second-person 'you' pronoun heads; these have separate etymologies and cannot take one edge."},
    {"record": record("heta"), "reason": "Andersen's 'here' head compares Sanskrit atra and cites a proposed *etra; no specific Turner etymon covering this Ashokan formation was established."},
]
assert len(accepted) == 4 and len({x["record"]["ID"] for x in accepted}) == 4
assert len(held) == 3
(D / "pass115-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
print("prepared 4 rows across 4 records and 3 holds")
