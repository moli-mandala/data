"""Prepare direct Zargari much/many and cock comparisons."""
import csv
import json
from pathlib import Path

from etymology_assignments import read_assignments

D = Path("curation/etymology-lab/indo-aryan-diversity-20260922")
SOURCE = "rezaibaghbidi2003zargari"
all_forms = {r["ID"]: r for r in csv.DictReader(open("cldf/forms.csv")) if SOURCE in r["Source"]}
linked = {r["Form_ID"] for r in read_assignments() if r["Status"] == "accepted"}
linked.update(r["Child_ID"] for r in csv.DictReader(open("cldf/edges.csv")) if r["Rank"] == "1")
identities = {r["Form_ID"]: r["Source_Key"] for r in csv.DictReader(open("data/form-identities.csv")) if r["Source_Key"].startswith(SOURCE + ":")}
source_rows = list(csv.reader(open("data/other/forms/20260825-rezai-baghbidi-zargari.csv")))
raw_rows = list(csv.DictReader(open("data/other/forms/raw_data/20260825-rezai-baghbidi-zargari-audit.csv")))

families = [
    (
        "much / very", "9190", "bahutva", ["f_z5nxvkr2b2wsy", "f_cod4jwu2eakdw"],
        "Turner 9190 bahutva explicitly lists European Romani but/būt 'much, many'. The source prints Zargari but 'many; much; several' under quantitative pronouns (p. 137 §3.3.7.1) and but 'very' under degree adverbs (p. 143 §3.20.1.5). The latter is a source-documented intensifier use of the same inherited quantity word; its gloss is preserved separately.",
    ),
    (
        "cock", "11589", "vā́śyatē", ["f_v4fbe4isitw6o"],
        "Turner 11589 vā́śyatē explicitly gives European Romani baš- 'to bark, crow' and, in the same parenthesis, German Romani bašno 'cock'. The source prints Zargari bāśno 'cock' among masculine nouns (p. 132 §3.1.9). The -no bird noun is a documented Romani derivative within this family, not a claimed independent Sanskrit noun.",
    ),
]

accepted = []
audit = []
review = [
    "# Zargari: much, very and cock",
    "",
    "Three source-checked reflex records in two Turner families. Standing autonomous authorization. Original PDF printed pp. 132, 137, 143, glyph audit and imported keys, homepage searches and full CDIAL articles/addenda were checked. No lexical import or rebuild.",
    "",
    "| # | Zargari forms and meanings | Proposed etymology | Evidence |",
    "|---|---|---|---|",
]
for number, (meaning, parent, headword, ids, evidence) in enumerate(families, 1):
    forms = []
    for fid in ids:
        r = all_forms[fid]
        assert fid not in linked, fid
        assert r["Language_ID"] == "Zarg", (fid, r["Language_ID"])
        key = identities[fid]
        source_matches = [x for x in source_rows if x[0] == "Zarg" and x[3] == r["Gloss"] and x[7] == r["Source"] and x[10] == key]
        assert len(source_matches) == 1, (fid, source_matches)
        raw_matches = [x for x in raw_rows if x["Status"] == "ingest" and key in x["Emitted_Keys"].split() and x["Final_Gloss"] == r["Gloss"]]
        assert len(raw_matches) == 1, (fid, raw_matches)
        accepted.append({
            "record": {k: r[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")},
            "parent": parent, "kind": "reflex", "citation": f"CDIAL[{parent} and addenda];{r['Source']}",
            "evidence": evidence + " Original PDF, glyph-decoded source audit and imported key checked; no transcription or identity change.",
        })
        audit.append({
            "id": fid, "source_key": key, "raw_unit": raw_matches[0]["Unit_ID"],
            "raw_form": raw_matches[0]["Raw_Form"], "final_form": raw_matches[0]["Final_Forms"],
            "printed_page": raw_matches[0]["Printed_Page"], "record_hash": raw_matches[0]["Record_SHA256"],
        })
        forms.append(r["Form"])
    review.append(f"| {number} | **{' / '.join(forms)}** ‘{meaning}’ ({len(ids)}) | **{headword}**, CDIAL **{parent}** | {evidence} |")

held = []
for form, reason in [
    ("tʰāv", "The source prints 'string' (p. 131) and Turner has European Romani thav 'thread', but places the form under both 6203 and 6770 with historical crossing and an alternative reconstruction in the addendum. The precise parent is unresolved."),
    ("sir", "The source prints sir 'all' (p. 137); Turner 13276 lists Romani saro/savoro 'all', but the Zargari vowel and absent -v/-o require a specific source comparison before assigning this branch."),
]:
    matches = [r for r in all_forms.values() if r["Form"] == form and r["ID"] not in linked]
    assert len(matches) == 1, (form, matches)
    held.append({"record": {k: matches[0][k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}, "reason": reason})

assert len(accepted) == 3 and len(held) == 2
(D / "pass129-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(D / "pass129-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
review += ["", "Two held records:"]
review += [f"- **{h['record']['Form']}** ‘{h['record']['Gloss']}’: {h['reason']}" for h in held]
review += ["", "Three assignment rows affect three records, all `reflex`. Discovery: homepage `zargari-search030`, full CDIAL `zargari-primary027`, PDF images and `pass129-source-audit.json`. Save and validation pending."]
(D / "pass129-review.md").write_text("\n".join(review) + "\n")
print("accepted", len(accepted), "held", len(held))
