"""Prepare source-checked Zargari verbs and irregular perfects."""
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
        "say / tell", "9383", "bhánati", ["f_rc4hrjqv7zvxo"],
        "Turner 9383 directly gives European Romani phen- 'to speak' under bhánati. The source prints phen-ipej 'to say; to tell' and defines productive infinitive -ipej after the present stem (p. 142 §3.17.3.1); the suffix is not a separate historical root.",
    ),
    (
        "become", "9416", "bhávati", ["f_reyrxvajcdbli"],
        "Turner 9416 gives Greek/Hungarian Romani uv-/ov- 'become, is' under bhávati. The source prints oj-ipey 'to become' as the copular subjunctive/imperfect stem (p. 142 §3.18.2); its §2.3.13 describes Zargari present-stem v > j, supporting the local oj- form. The productive infinitive is retained.",
    ),
    (
        "went", "4008", "gatá §1", ["f_7lcdghh3dbayy", "f_fgcw7tlimwtvo", "f_e5rjsowoqzu5o"],
        "Turner 4008 §1 explicitly gives European Romani gelo as the past of 'go'. The source lists irregular gender/first-person forms gölu 'he went', geli 'she went', gölum 'I went' (p. 140 §3.10.2). These are ordinary inflections of its inherited perfect stem, preserving ö/e alternation and person/gender endings.",
    ),
    (
        "died", "10278", "mr̥tá", ["f_bgnnh3vddzts2", "f_r5zfmzt3i4a44"],
        "Turner 10278 explicitly gives European Romani mulo under mr̥tá and other Indo-Aryan gendered 'died' outcomes. The source lists mulo/muli as masculine/feminine irregular perfect forms 'he/she died' (p. 140 §3.10.2); no synchronic derivation is asserted between them.",
    ),
    (
        "fell", "7722", "pátati §1", ["f_2puheukxkndjo", "f_3cg3fufsngd7g"],
        "Turner 7722 §1 explicitly includes European Romani per- and its past participle pelo 'fell' under pátati. The source prints Zargari pölu/peli as masculine/feminine irregular perfect forms 'he/she fell' (p. 140 §3.10.2); its ö/e and endings remain source-specific.",
    ),
]

accepted = []
audit = []
review = [
    "# Zargari: saying, becoming and irregular past verbs",
    "",
    "Nine source-checked reflex records across five historical families. Standing autonomous authorization. Original PDF printed pp. 140–142 (PDF pp. 18–20), glyph audit and imported keys, homepage searches and full CDIAL articles/addenda were checked. No lexical import or rebuild.",
    "",
    "| # | Zargari forms and meaning | Proposed etymology | Evidence |",
    "|---|---|---|---|",
]
for number, (meaning, parent, headword, ids, evidence) in enumerate(families, 1):
    forms = []
    for fid in ids:
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
        forms.append(r["Form"])
    review.append(f"| {number} | **{' / '.join(forms)}** ‘{meaning}’ ({len(ids)}) | **{headword}**, CDIAL **{parent}** | {evidence} |")

held = []
for form, reason in [
    ("isipey", "The source defines this auxiliary 'to be' (pp. 141–142), but Turner 9416 lists Romani asom/som as crossing with ásti rather than supplying a direct is- branch. The specific source stem and possible Persian contact require review."),
    ("jeyipey", "The source calls this an irregular infinitive of džā- 'go' (p. 142), but the precise historical present-stem etymon is not established by Turner's separate gatá past entry 4008. It should not be assigned merely from the past forms."),
    ("ayili", "The source gives 'he/she came' as another irregular perfect (p. 140); the matching historical branch and whether this is the same stem as present āv- need verification."),
]:
    matches = [r for r in all_forms.values() if r["Form"] == form and r["ID"] not in linked]
    assert len(matches) == 1, (form, len(matches))
    held.append({"record": {k: matches[0][k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}, "reason": reason})

assert len(accepted) == 9 and len(held) == 3
(D / "pass100-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(D / "pass100-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
review += ["", "Three held records:"]
review += [f"- **{h['record']['Form']}** ‘{h['record']['Gloss']}’: {h['reason']}" for h in held]
review += ["", "Nine assignment rows affect nine records, all `reflex`; gender/person inflection is not synchronic derivation. Discovery: homepage `zargari-search019–020`, full CDIAL `zargari-primary018`, PDF images and `pass100-source-audit.json`. Save and validation pending."]
(D / "pass100-review.md").write_text("\n".join(review) + "\n")
print("accepted", len(accepted), "held", len(held))
