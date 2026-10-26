"""Prepare image-checked Andersen (1990) source variants and lexical heads.

Research preparation only; the shared etymology_lab helper performs graph validation
and sidecar writing. Do not rerun after saving because targets cease to be unlinked.
"""

import csv
import json
from pathlib import Path

D = Path(__file__).resolve().parent
DATA = D.parents[2]
records = json.loads((D / "andersen-records.json").read_text())
by_form = {r["Form"]: r for r in records}
needed = (
    "tī", "ti", "śaca", "saca", "śvaga", "svaga", "prāṇa", "prāna",
    "pāna", "pi", "pī", "velā", "salila", "kʰudaka",
    "kʰudakamahalaka", "mahalaka", "kʰugdaka", "hevaha", "sa", "pipule",
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
    ("ti", "1562"), ("saca", "13112"), ("svaga", "13910"),
    ("pāna", "8928"), ("mahalaka", "9935"),
):
    assert linked.get(by_form[form]["ID"]) == parent, form


def record(form):
    r = by_form[form]
    assert r["ID"] not in linked, form
    return {key: r[key] for key in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}


proposals = [
    ("tī", by_form["ti"]["ID"], "variant", "Andersen's printed tī line says 'see ti'. The source's ti head is already linked to Turner 1562, which explicitly includes Ashokan ti under íti."),
    ("śaca", by_form["saca"]["ID"], "variant", "Andersen's printed śaca line says 'see saca'. The source's saca 'truth' head is already linked to Turner 13112, which explicitly includes Ashokan saca."),
    ("śvaga", by_form["svaga"]["ID"], "variant", "The printed śvaga line says 'see svaga'. The source's svaga 'heaven' head is already linked to Turner 13910, which explicitly lists Ashokan svaga."),
    ("prāṇa", by_form["pāna"]["ID"], "variant", "The printed prāṇa line says 'see pāna'. The source's pāna 'living being' head is already linked to Turner 8928; Turner explicitly contrasts Ashokan praṇa/prāṇa and kāl. pāna."),
    ("prāna", by_form["pāna"]["ID"], "variant", "The printed prāna line also says 'see pāna'. Keep this source spelling as a sibling variant of the linked pāna head, ultimately Turner 8928."),
    ("pi", "485", "reflex", "Andersen prints pi 'also' beside RV ápi and Pali pi. Turner 485 explicitly lists Ashokan pi for the same particle; Turner 486 is an unrelated expressive 'also' form."),
    ("pī", by_form["pi"]["ID"], "variant", "The printed pī line says 'see pi'. It points to this source's pi 'also' head, linked in the same pass to Turner 485."),
    ("velā", "12115", "reflex", "Andersen prints velā 'limit of time' beside ŚB vḗlā and Pali velā. Turner 12115 section 1 gives the matching limit/time sense; its possible ancient Dravidian input does not identify a direct Ashokan donor."),
    ("salila", "12335", "reflex", "Andersen prints salila 'body, relics' beside RV śárīra and Pali sarīra. Turner 12335 gives the same body noun in Pali and Prakrit; the unusual r>l form is supported by this source comparison, not confused with an unrelated water word."),
    ("kʰudaka", "3712", "reflex", "Andersen prints khudaka 'lowly person' beside Sanskrit kṣudraka and Pali khuddaka. Turner 3712 explicitly cites Ashokan khuda/khudaka under the small/lowly branch."),
]
accepted = []
for form, parent, kind, evidence in proposals:
    r = record(form)
    c = {
        "tī": "1562", "śaca": "13112", "śvaga": "13910",
        "prāṇa": "8928", "prāna": "8928", "pī": "485",
    }.get(form, parent)
    accepted.append({
        "record": r,
        "parent": parent,
        "kind": kind,
        "citation": f"{r['Source']};CDIAL[{c}]",
        "evidence": evidence,
    })

r = record("kʰudakamahalaka")
components = [by_form["kʰudaka"]["ID"], by_form["mahalaka"]["ID"]]
accepted.append({
    "record": r,
    "parent": components[0],
    "components": components,
    "kind": "component",
    "citation": f"{r['Source']};CDIAL[3712];CDIAL[9935]",
    "evidence": "Andersen prints khudakamahalaka 'lowly and great person', compares Sanskrit kṣudraka plus mahallaka and Pali khuddaka-mallaka, and cross-refers to both lexical heads. The installed form is composite, so preserve its two source-defined components in order.",
})

holds = [
    ("kʰugdaka", "The printed source line is khuḍaka 'see khudaka', but the installed form is kʰugdaka; a likely transcription/OCR issue must be reviewed separately before a variant link."),
    ("hevaha", "Andersen says 'see hevam and ah-': this is not a simple spelling variant of hevam, and the ah- verbal component lacks a secure accepted ancestor here."),
    ("sa", "The installed record combines source-distinct sa 'six' and sa 'he, she, it, that'; assigning one parent would erase this homonym boundary."),
    ("pipule", "The printed line says 'see vipule', but there is no distinct installed vipule source head to use as a verified variant target."),
]
held = [{"record": record(form), "reason": reason} for form, reason in holds]

assert len(accepted) == 11 and len({x["record"]["ID"] for x in accepted}) == 11
assert len(held) == 4
out = {"accepted": accepted, "held": held}
(D / "pass086-decisions.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print("prepared 12 rows across 11 records and 4 holds")
