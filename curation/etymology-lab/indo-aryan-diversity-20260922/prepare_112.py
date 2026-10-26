"""Prepare direct Turner Zargari 'front' reflexes."""
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

family = (
    "68-2x", "*agrala- (ágra extended with -la-)",
    ["f_ukrhyfsmef2dg", "f_72gtntxbg3dk2", "f_yx74sw5iqaq5g"],
    "Turner 68, specifically its -la- extension represented by the compiled node 68-2x, explicitly cites European Romani angle 'in front' and anglal 'from in front'. The source prints Zargari ānglāl [ānglār] 'from the front' while documenting liquid dissimilation (§2.3.9, p. 128), then āngl-āl 'from the front' (§3.1.15, p. 132). It lists ānglo 'front; in front' as an adverb (§3.20.1.3, p. 143) and 'in front of' as an adposition (§3.20.2.1, p. 144). These are direct matches to the two Turner Romani comparanda, with the source-defined adverbial and ablative forms retained.",
)

parent, headword, ids, evidence = family
accepted = []
audit = []
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
        "citation": f"CDIAL[68 and addenda; 68-2x];{r['Source']}",
        "evidence": evidence + " Original PDF, glyph-decoded source audit and imported key checked; no transcription or identity change.",
    })
    audit.append({
        "id": fid, "source_key": key, "raw_unit": raw_matches[0]["Unit_ID"],
        "raw_form": raw_matches[0]["Raw_Form"], "final_form": raw_matches[0]["Final_Forms"],
        "printed_page": raw_matches[0]["Printed_Page"], "record_hash": raw_matches[0]["Record_SHA256"],
    })

held = []
for form, reason in [
    ("nāy", "The source prints 'finger' (p. 129), while Turner 6914 directly gives European Romani nai 'fingernail'. A later Romani survey has nay 'nail; finger', but the Zargari semantic extension is not established by Turner itself."),
    ("somlāl", "The source prints 'gold' (p. 128) under liquid dissimilation; Turner 13519 has related Romani sovnakay/somnakay, but the full -lāl versus -nakay correspondence is not explained by the source."),
    ("bārbāyla", "The source prints this 'wind' form (p. 131), while Turner 11491 marks the Romani balval/bārval comparison with question marks and cannot securely distinguish vātá from vāyú. Hold the precise origin."),
]:
    matches = [r for r in all_forms.values() if r["Form"] == form and r["ID"] not in linked]
    assert len(matches) == 1, (form, matches)
    held.append({"record": {k: matches[0][k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}, "reason": reason})

assert len(accepted) == 3 and len(held) == 3
(D / "pass112-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(D / "pass112-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
review = [
    "# Zargari: in front and from the front",
    "",
    "Three source-checked reflex records in one explicit Turner Romani family. Standing autonomous authorization. Original PDF printed pp. 128, 132, 143–144, glyph audit and imported keys, homepage searches and full CDIAL article/addenda were checked. No lexical import or rebuild.",
    "",
    "| Zargari forms and meanings | Proposed etymology | Evidence |",
    "|---|---|---|",
    f"| **{' / '.join(all_forms[x]['Form'] for x in ids)}** ‘from the front; front / in front; in front of’ | **{headword}**, CDIAL **{parent}** | {evidence} |",
    "",
    "Three held records:",
]
review += [f"- **{h['record']['Form']}** ‘{h['record']['Gloss']}’: {h['reason']}" for h in held]
review += ["", "Three assignment rows affect three records, all `reflex`. Discovery: homepage `zargari-search025`, full CDIAL `zargari-primary021`, PDF images and `pass112-source-audit.json`. Save and validation pending."]
(D / "pass112-review.md").write_text("\n".join(review) + "\n")
print("accepted", len(accepted), "held", len(held))
