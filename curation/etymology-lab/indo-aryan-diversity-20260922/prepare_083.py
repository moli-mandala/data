"""Prepare source-checked Zargari inherited lexemes and attested plurals."""
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
        "lip", "2563", "ṓṣṭha", ["f_yzvwsp7ineufu"],
        "Turner explicitly records European Romani vušt 'lip' under ṓṣṭha; the source prints vušt in its unchanged-singular/plural noun list (p. 131 §3.1.8).",
    ),
    (
        "day", "6333", "divasá", ["f_izw3vw3vizd6o", "f_yyqcq3glilzc2"],
        "Turner gives European Romani dives and Palula dīs under divasá. The source explicitly contrasts Zargari singular dis and plural dise (p. 131 §3.1.8); the plural is an ordinary inflection of the same inherited stem.",
    ),
    (
        "tongue / language", "5228", "jihvā́", ["f_3a36ln5oiuh5a"],
        "Turner gives southeastern European Romani čhib 'tongue' under jihvā́. The source expressly lists čhiba as the plural of čhib (p. 131 §3.1.8); its singular already has a reviewed 5228 link.",
    ),
    (
        "face / mouth", "10158", "múkha", ["f_v7rxilczmovni"],
        "Turner explicitly gives Greek and Hungarian Romani muy 'mouth, face' under múkha; Zargari muj 'face; mouth' is printed in the irregular-plural list (p. 132 §3.1.9).",
    ),
    (
        "sheep", "9153", "bárkara", ["f_se6hrkoem5adc"],
        "Turner gives European Romani bakro, feminine bakri 'sheep' under bárkara. Rezaibaghbidi prints bakri 'sheep' in §2.3.3 and repeats it in §3.2.1, distinct from the same etymon's widespread later 'goat' sense.",
    ),
    (
        "boy / son", "5026", "\\*chāpa", ["f_le2tucdscb54g", "f_w32jbtfbfht3e"],
        "Turner gives southeastern European Romani čhavo 'boy' under *chāpa. The source prints Zargari čhā and its irregular plural čhāvu (p. 132 §3.1.9), preserving its own truncation and plural vowel; Turner's deeper relationship to śāva remains unclear.",
    ),
    (
        "brother", "9661", "bhrā́tr̥", ["f_it7zwcz5nmg7y"],
        "Turner expressly gives European Romani phral 'brother' under bhrā́tr̥. The source identifies Zargari phorāl as < Early Romani phral and explains its inserted vowel by Zargari CV(CC) syllable structure (p. 129 §2.4.1).",
    ),
]

accepted = []
audit = []
review = [
    "# Zargari: inherited nouns and printed plurals",
    "",
    "Nine source-checked reflex records across seven inherited families. Standing autonomous authorization. The relevant Zargari pages were read in the original PDF (printed pp. 127, 129, 131–133; PDF pp. 5, 7, 9–11), the glyph audit and import rows were matched, and complete CDIAL articles/addenda were checked. No lexical import or rebuild.",
    "",
    "| # | Zargari forms and meaning | Proposed etymology | Evidence |",
    "|---|---|---|---|",
]
for number, (meaning, parent, headword, ids, evidence) in enumerate(families, 1):
    forms = []
    glosses = []
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
            and x["Source"] == r["Source"].split(";")[0]
        ]
        # bakri's installed source field combines its first and repeated attestations.
        if fid == "f_se6hrkoem5adc":
            raw_matches = [
                x for x in raw_rows
                if x["Status"] == "ingest"
                and key in x["Emitted_Keys"].split()
                and x["Final_Gloss"] == r["Gloss"]
            ]
        assert len(raw_matches) >= 1, (fid, raw_matches)
        citation = f"CDIAL[{parent} and addenda];{r['Source']}"
        accepted.append({
            "record": {k: r[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")},
            "parent": parent,
            "kind": "reflex",
            "citation": citation,
            "evidence": evidence + " Original PDF, glyph-decoded source audit and imported key checked; no transcription or identity change.",
        })
        audit.append({
            "id": fid, "source_key": key, "raw_units": [x["Unit_ID"] for x in raw_matches],
            "raw_forms": [x["Raw_Form"] for x in raw_matches],
            "final_forms": [x["Final_Forms"] for x in raw_matches],
            "printed_pages": [x["Printed_Page"] for x in raw_matches],
            "record_hashes": [x["Record_SHA256"] for x in raw_matches],
        })
        forms.append(r["Form"])
        glosses.append(r["Gloss"])
    review.append(f"| {number} | **{' / '.join(forms)}** ‘{meaning}’ ({len(ids)}) | **{headword}**, CDIAL **{parent}** | {evidence} |")

held = []
for form, reason in [
    ("kirmo", "Turner 3438 gives Romani kermo/kirmo under kr̥mi, but both Turner and the imported European Romani comparison admit Persian kirm. Zargari is spoken in Iran, so the immediate route is unresolved."),
    ("gi", "The printed noun list glosses this 'heart', but the exact historical branch has not been established from Turner's article or a diagnostic Romani comparison."),
    ("śoru", "The source prints šoru 'head' and discusses vowel harmony, but a verified Indic etymon and any Persian contact history still need separate review."),
]:
    matches = [r for r in all_forms.values() if r["Form"] == form and r["ID"] not in linked]
    assert len(matches) == 1, (form, len(matches))
    held.append({"record": {k: matches[0][k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}, "reason": reason})

assert len(accepted) == 9 and len(held) == 3
(D / "pass083-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(D / "pass083-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
review += ["", "Three held records:"]
review += [f"- **{h['record']['Form']}** ‘{h['record']['Gloss']}’: {h['reason']}" for h in held]
review += ["", "Nine assignment rows affect nine records, all `reflex`. The irregular plurals are source-defined inflections; no synchronic derivation is asserted. Discovery: homepage search `zargari-search014–015`; full CDIAL prose `zargari-primary012–013`; source images and `pass083-source-audit.json`. Save and validation pending."]
(D / "pass083-review.md").write_text("\n".join(review) + "\n")
print("accepted", len(accepted), "held", len(held))
