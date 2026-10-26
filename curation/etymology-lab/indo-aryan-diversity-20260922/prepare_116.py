"""Prepare two Zargari verbs with explicit Turner Romani comparisons."""
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
        "to go", "10452", "yā́ti", ["f_5pdd2w5eus6cq"],
        "Turner 10452 explicitly lists European Romani ǰal 'goes' from yā́ti. The source specifically identifies Zargari džejipej 'to go' as an irregular infinitive from džā- (§3.17.3.2, p. 142), distinguishing this present stem from the gatá-derived irregular past linked in pass100. The infinitive formation is source morphology, not a separate etymon.",
    ),
    (
        "throw!", "5055", "chupáti (crossed with kṣipáti)", ["f_m7ogimhmzhsaw"],
        "Turner 5055 explicitly lists European Romani čiv- 'to throw, put' in its section crossed with kṣipáti. The source prints Zargari čuv [čuw] 'Throw!' in §2.3.20 (p. 128), while giving final /v/ > [w]. The i/u vowel difference is retained as a local form, not explained by an invented sound rule. This short imperative is assigned to the attested Romani branch with Turner's crossing retained in the note, not to a bare kṣipáti node.",
    ),
]

accepted = []
audit = []
review = [
    "# Zargari: go and throw",
    "",
    "Two source-checked reflex records in two historical families. Standing autonomous authorization. Original PDF printed pp. 128–129, 142, glyph audit and imported keys, homepage searches and full CDIAL articles/addenda were checked. No lexical import or rebuild.",
    "",
    "| # | Zargari form and meaning | Proposed etymology | Evidence |",
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
            "parent": parent, "kind": "reflex",
            "citation": f"CDIAL[{parent} and addenda];{r['Source']}",
            "evidence": evidence + " Original PDF, glyph-decoded source audit and imported key checked; no transcription or identity change.",
        })
        audit.append({
            "id": fid, "source_key": key, "raw_unit": raw_matches[0]["Unit_ID"],
            "raw_form": raw_matches[0]["Raw_Form"], "final_form": raw_matches[0]["Final_Forms"],
            "printed_page": raw_matches[0]["Printed_Page"], "record_hash": raw_matches[0]["Record_SHA256"],
        })
        forms.append(r["Form"])
    review.append(f"| {number} | **{' / '.join(forms)}** ‘{meaning}’ | **{headword}**, CDIAL **{parent}** | {evidence} |")

held = []
for form, reason in [
    ("āssayipey", "Turner 14048 gives European Romani asel 'laughs', and source p. 127 prints āssajipej 'to laugh'; the extra -aj- stem formation and whether it reflects a causative or local inflection remain unresolved."),
    ("ayili", "The source prints the irregular past 'came' (p. 140), but the exact āv-/ay- branch and its relationship to the inherited present stem remain unresolved; do not transfer Turner's go etymon to it."),
]:
    matches = [r for r in all_forms.values() if r["Form"] == form and r["ID"] not in linked]
    assert len(matches) == 1, (form, matches)
    held.append({"record": {k: matches[0][k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}, "reason": reason})

assert len(accepted) == 2 and len(held) == 2
(D / "pass116-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(D / "pass116-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
review += ["", "Two held records:"]
review += [f"- **{h['record']['Form']}** ‘{h['record']['Gloss']}’: {h['reason']}" for h in held]
review += ["", "Two assignment rows affect two records, both `reflex`. Discovery: homepage `zargari-search026`, full CDIAL `zargari-primary023`, PDF images and `pass116-source-audit.json`. Save and validation pending."]
(D / "pass116-review.md").write_text("\n".join(review) + "\n")
print("accepted", len(accepted), "held", len(held))
