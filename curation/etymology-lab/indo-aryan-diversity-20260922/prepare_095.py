"""Prepare source-checked Zargari phonology-example reflexes."""
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
        "hot; warm", "5679", "taptá", "f_dvecdtpt7fbxa",
        "Turner gives European Romani tato 'hot' under taptá, with Pali/Prakrit tatta. The source explicitly prints Zargari tatho for tatto and describes aspiration in place of gemination (p. 127 §2.3.8). Its source-defined tatto variant is already an installed variant, not another assignment row.",
    ),
    (
        "place", "13753", "sthā́na", "f_zak4vrauftnho",
        "Turner §1 gives Welsh Romani than and Greek Romani tan 'place' under sthā́na. The source prints Zargari thān > tān as an aspiration alternation (p. 127 §2.3.1); the installed tān variant already points to this source form.",
    ),
    (
        "testicle", "8381", "pēla", "f_atxwr5jigybp4",
        "Turner gives European Romani pelo 'testicle' under pēla. The source itself compares Zargari bellu with Early Romani pelo while illustrating medial gemination (p. 127 §2.3.7); the Zargari b- and final -u are retained as attested.",
    ),
    (
        "ox; bull", "4313", "gōrūpá", "f_nxj2oi622orci",
        "Turner explicitly gives European Romani guruv m. under gōrūpá, alongside Pali 'bull' and Prakrit 'ox'. The source prints Zargari guruv [guruw] 'ox, bull' and explains its phonetic [w] after /u/ (p. 128 §2.3.20).",
    ),
    (
        "sugar; candy; sweet", "4182", "guḍá²", "f_3ek2f3drextq4",
        "Turner gives European Romani gudlo/gullo/gulo 'sweet' as derivatives of guḍá² 'molasses'. The source prints Zargari gudlo/guldo with 'sugar; candy, sweet' and explicitly calls this liquid metathesis (p. 129 §2.3.24). The installed guldo variant already points to gudlo; the single reviewed record retains all source senses.",
    ),
]

accepted = []
audit = []
review = [
    "# Zargari: five inherited phonology examples",
    "",
    "Five reflex records with direct Romani comparanda in Turner and original source support. Standing autonomous authorization. PDF printed pp. 127–129 (PDF pp. 5–7), source glyph audit, installed keys, homepage search and full CDIAL articles/addenda checked. No lexical import or rebuild.",
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
        "final_forms": raw_matches[0]["Final_Forms"],
        "printed_page": raw_matches[0]["Printed_Page"],
        "record_hash": raw_matches[0]["Record_SHA256"],
    })
    review.append(f"| {number} | **{r['Form']}** ‘{meaning}’ | **{headword}**, CDIAL **{parent}** | {evidence} |")

held = []
for form, reason in [
    ("sujjo", "Turner 12511 gives Greek Romani šuzo under *śucya, while 12525 lists the same Romani forms under *śudhya and sends readers back to *śucya. The source only gives Zargari sujjo 'clean' in a gemination example; the precise older branch remains contested."),
    ("āssayipey", "Turner 14021 hásati supplies regional Indo-Aryan and Nuristani laugh verbs, but no diagnostic European Romani form. The source's Zargari infinitive suffix and stem need separate analysis before selecting a parent."),
    ("cuv", "Turner 5055 places Romani čiv/čhiv 'throw' under chupáti with a kṣipáti crossing. The source's čuv imperative is clear, but the i/u difference and competing ancient root remain unresolved."),
]:
    matches = [r for r in all_forms.values() if r["Form"] == form and r["ID"] not in linked]
    assert len(matches) == 1, (form, len(matches))
    held.append({"record": {k: matches[0][k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}, "reason": reason})

assert len(accepted) == 5 and len(held) == 3
(D / "pass095-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(D / "pass095-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
review += ["", "Three held records:"]
review += [f"- **{h['record']['Form']}** ‘{h['record']['Gloss']}’: {h['reason']}" for h in held]
review += ["", "Five assignment rows affect five records, all `reflex`; source-defined variants already have compiled variant edges. Discovery: homepage `zargari-search018`, full CDIAL `zargari-primary017`, PDF images and `pass095-source-audit.json`. Save and validation pending."]
(D / "pass095-review.md").write_text("\n".join(review) + "\n")
print("accepted", len(accepted), "held", len(held))
