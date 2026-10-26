"""Prepare direct Turner/Romani matches from Zargari phonology examples."""
import csv
import json
from pathlib import Path

from etymology_assignments import read_assignments


D = Path("curation/etymology-lab/indo-aryan-diversity-20260922")
SOURCE = "rezaibaghbidi2003zargari"
all_forms = {
    r["ID"]: r
    for r in csv.DictReader(open("cldf/forms.csv"))
    if SOURCE in r["Source"]
}
linked = {r["Form_ID"] for r in read_assignments() if r["Status"] == "accepted"}
linked.update(
    r["Child_ID"]
    for r in csv.DictReader(open("cldf/edges.csv"))
    if r["Rank"] == "1"
)
identities = {
    r["Form_ID"]: r["Source_Key"]
    for r in csv.DictReader(open("data/form-identities.csv"))
    if r["Source_Key"].startswith(SOURCE + ":")
}
source_rows = list(csv.reader(open("data/other/forms/20260825-rezai-baghbidi-zargari.csv")))
raw_rows = list(csv.DictReader(open("data/other/forms/raw_data/20260825-rezai-baghbidi-zargari-audit.csv")))

families = [
    (
        "name", "7067", "nā́man", "f_ngzflwilszfj2",
        "The source explicitly derives ānāv from Old Indo-Aryan *nāman and discusses v from medial m (p. 128 §2.3.18). Turner 7067 independently lists European Romani anav/nav 'name', distinguishing Palula nam as a Persian loan.",
    ),
    (
        "tear", "919", "áśru", "f_tw4v77ymuee62",
        "The source identifies stressed áwsa 'tear' as genuine Romani (p. 129 §2.5.1). Turner 919 §1 gives Greek Romani asva, Hungarian yasva and Bohemian avs/avsa, matching the Zargari stem without invoking the distinct nasalized or contaminated branches.",
    ),
    (
        "good", "10888", "lakṣmaka", "f_i3k7yp3egunts",
        "The source identifies lāčhó 'good' as genuine Romani (p. 129 §2.5.1). Turner 10888 directly gives southeastern European Romani lačhó under lakṣmaka, while flagging a possible crossing with accha; retain that caveat rather than treating the older root as certain.",
    ),
    (
        "white", "8051", "pāṇḍú", "f_5xqyzrrh6kcjw",
        "The source identifies pārnó 'white' as genuine Romani (p. 129 §2.5.1). Turner 8051 §1 explicitly gives European Romani parno 'white'; the -rn- form belongs to this main branch, not his separate Kashmiri *paṇḍra subsection.",
    ),
    (
        "bride / daughter-in-law", "11251", "vadhūṭī", "f_vkl6qkndfcmkg",
        "The source prints bori 'bride; daughter-in-law' in its irregular noun paradigms (p. 132 §3.1.9). Turner 11251 gives Greek Romani borí 'bride, son's wife' under vadhūṭī, and 12177 explicitly redirects European Romani bori to this etymon rather than vyavahārikā.",
    ),
    (
        "shoulder", "13839-2", "*sphiyakk-", "f_cyv7qlq6f4xkg",
        "The source prints pukko/puqqo 'shoulder' and expressly compares Early Romani phiko in its medial-gemination discussion (p. 127 §2.3.7); §2.3.1 notes Zargari deaspiration. Turner 13839 places European Romani phiko specifically in the -kk- shoulder extension, represented by CDIAL 13839-2. The local u vowel is preserved, not normalized to the comparandum.",
    ),
]

accepted = []
audit = []
review = [
    "# Zargari: six inherited phonology and noun examples",
    "",
    "Six reflex records, each with a direct Romani comparison in Turner and an independently checked source form. Standing autonomous authorization. Original PDF printed pp. 127–129, 132 (PDF pp. 5–7, 10), source glyph audit, installed source keys, and full CDIAL articles/addenda checked. No lexical import or rebuild.",
    "",
    "| # | Zargari form and meaning | Proposed etymology | Evidence |",
    "|---|---|---|---|",
]
for number, (meaning, parent, headword, fid, evidence) in enumerate(families, 1):
    r = all_forms[fid]
    assert fid not in linked, fid
    assert r["Language_ID"] == "Zarg", (fid, r["Language_ID"])
    key = identities[fid]
    source_matches = [
        x for x in source_rows
        if x[0] == "Zarg" and x[3] == r["Gloss"]
        and x[7] == r["Source"] and x[10] == key
    ]
    assert len(source_matches) == 1, (fid, source_matches)
    raw_matches = [
        x for x in raw_rows
        if x["Status"] == "ingest"
        and key in x["Emitted_Keys"].split()
        and x["Final_Gloss"] == r["Gloss"]
    ]
    assert len(raw_matches) == 1, (fid, raw_matches)
    citation = f"CDIAL[{parent} and addenda];{r['Source']}"
    accepted.append({
        "record": {k: r[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")},
        "parent": parent,
        "kind": "reflex",
        "citation": citation,
        "evidence": evidence + " Original PDF, glyph-decoded source audit and imported key checked; no transcription or identity change.",
    })
    audit.append({
        "id": fid, "source_key": key, "raw_unit": raw_matches[0]["Unit_ID"],
        "raw_form": raw_matches[0]["Raw_Form"],
        "final_form": raw_matches[0]["Final_Forms"],
        "printed_page": raw_matches[0]["Printed_Page"],
        "record_hash": raw_matches[0]["Record_SHA256"],
    })
    displayed = "\\*sphiyakk-" if parent == "13839-2" else headword
    review.append(f"| {number} | **{r['Form']}** ‘{meaning}’ | **{displayed}**, CDIAL **{parent}** | {evidence} |")

held = []
for form, reason in [
    ("ākkur", "The source compares Early Romani akhor and Turner 48 groups European Romani akhor under akṣōṭa; however, Turner’s addendum derives that etymon from Iranian and also lists Persian Gypsy akūl. The route of Zargari ākkur in Iran is not secure enough to choose reflex versus later borrowing."),
    ("bukʰāló", "The source calls this 'hungry' word genuine Romani, but Turner 9284/9287 do not give the European Romani -ālo form. Its stem and adjective suffix require analysis before assigning a single historical parent."),
    ("cʰal", "The source compares Early Romani učhal 'shadow' (p. 127 §2.3.1), but the exact Turner subsection and historical formation have not been established."),
]:
    matches = [r for r in all_forms.values() if r["Form"] == form and r["ID"] not in linked]
    assert len(matches) == 1, (form, len(matches))
    held.append({"record": {k: matches[0][k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}, "reason": reason})

assert len(accepted) == 6 and len(held) == 3
(D / "pass091-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(D / "pass091-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
review += ["", "Three held records:"]
review += [f"- **{h['record']['Form']}** ‘{h['record']['Gloss']}’: {h['reason']}" for h in held]
review += ["", "Six assignment rows affect six records, all `reflex`. Discovery: homepage `zargari-search017`, full CDIAL `zargari-primary015–016`, original source images and `pass091-source-audit.json`. Save and validation pending."]
(D / "pass091-review.md").write_text("\n".join(review) + "\n")
print("accepted", len(accepted), "held", len(held))
