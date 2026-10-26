"""Prepare Zargari God, other, and other-day etymologies."""
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

other = "f_4sxak5n6l24b6"
day = "f_izw3vw3vizd6o"
assert other not in linked and day in linked
proposals = [
    (
        "f_qz6qnwlv35mwk", "6530", "reflex", None, "CDIAL[6530 and addenda]",
        "Turner 6530 dēvátā explicitly cites European Romani devel 'God'. The source prints Zargari dévlā 'O God!' as a vocative stress example (p. 130 §2.5.2), and its noun morphology identifies the vocative -ā as probably Persian-derived (p. 130 §3.1.2). The Romani stem is assigned to Turner's family while the local vocative suffix and its contact history remain in the note.",
    ),
    (
        other, "434", "reflex", None, "CDIAL[434 and addenda]",
        "Turner 434 ápara explicitly lists European Romani aver 'other'. The source prints Zargari āvur 'another; another one' in the indefinite-pronoun list (p. 136 §3.3.6.1). The vowel form is retained as printed; no immediate borrowing route is inferred.",
    ),
    (
        "f_psgoses5vysfw", other, "component", [other, day], "CDIAL[434];CDIAL[6333]",
        "The source prints the two-word Zargari phrase āvur dis 'the day after tomorrow' (p. 143 §3.20.1.2). Its lexical constituents are source āvur 'another' (newly linked to Turner 434) and dis 'day' (already linked to Turner 6333). This records two inherited components and preserves the idiomatic whole gloss; it does not posit an ancient whole-phrase etymon.",
    ),
]

accepted = []
audit = []
review = [
    "# Zargari: O God, another and another day",
    "",
    "Three source-checked records: two reflexes and one two-part phrase. Standing autonomous authorization. Original PDF printed pp. 130, 136, 143, glyph audit and imported keys, homepage search and full CDIAL articles/addenda were checked. No lexical import or rebuild.",
    "",
    "| # | Zargari form and meaning | Proposal | Evidence |",
    "|---|---|---|---|",
]
for number, (fid, parent, kind, components, citation_prefix, evidence) in enumerate(proposals, 1):
    r = all_forms[fid]
    assert fid not in linked, fid
    assert r["Language_ID"] == "Zarg", (fid, r["Language_ID"])
    key = identities[fid]
    source_matches = [x for x in source_rows if x[0] == "Zarg" and x[3] == r["Gloss"] and x[7] == r["Source"] and x[10] == key]
    assert len(source_matches) == 1, (fid, source_matches)
    raw_matches = [x for x in raw_rows if x["Status"] == "ingest" and key in x["Emitted_Keys"].split() and x["Final_Gloss"] == r["Gloss"]]
    assert len(raw_matches) == 1, (fid, raw_matches)
    decision = {
        "record": {k: r[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")},
        "parent": parent, "kind": kind, "citation": f"{citation_prefix};{r['Source']}",
        "evidence": evidence + " Original PDF, glyph-decoded source audit and imported key checked; no transcription or identity change.",
    }
    if components is not None:
        decision["components"] = components
    accepted.append(decision)
    audit.append({
        "id": fid, "source_key": key, "raw_unit": raw_matches[0]["Unit_ID"],
        "raw_form": raw_matches[0]["Raw_Form"], "final_form": raw_matches[0]["Final_Forms"],
        "printed_page": raw_matches[0]["Printed_Page"], "record_hash": raw_matches[0]["Record_SHA256"],
    })
    review.append(f"| {number} | **{r['Form']}** ‘{r['Gloss']}’ | **{kind}**, {citation_prefix} | {evidence} |")

held = []
for form, reason in [
    ("m-āc", "The source analyzes the prohibitive mā- before ač 'stay' (p. 128 §2.3.11), and Turner 1031 gives Romani ač; a complete two-component source-form analysis is unavailable because the affirmative stem is not separately installed."),
    ("mā́kʰuv", "The source identifies prohibitive mā- before khuv 'weave' (p. 130 §2.5.3), while Turner 4205 gives Romani khuv. The affirmative source stem is not installed separately, so the exact component edge should await a representation review."),
]:
    matches = [r for r in all_forms.values() if r["Form"] == form and r["ID"] not in linked]
    assert len(matches) == 1, (form, matches)
    held.append({"record": {k: matches[0][k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}, "reason": reason})

assert len(accepted) == 3 and len(held) == 2
(D / "pass137-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(D / "pass137-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
review += ["", "Two held records:"]
review += [f"- **{h['record']['Form']}** ‘{h['record']['Gloss']}’: {h['reason']}" for h in held]
review += ["", "Three affected records yield four assignment rows: two reflexes and two component edges. Discovery: homepage `zargari-search032`, full CDIAL `zargari-primary030` plus prior `zargari-primary021` for dis, PDF images and `pass137-source-audit.json`. Save and validation pending."]
(D / "pass137-review.md").write_text("\n".join(review) + "\n")
print("accepted", len(accepted), "held", len(held))
