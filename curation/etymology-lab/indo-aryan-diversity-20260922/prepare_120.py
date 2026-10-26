"""Prepare Zargari one and middle with direct Turner comparanda."""
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
        "one", "2462-2", "*ēkka-", ["f_3mm3to2hqjcky"],
        "Turner 2462 §2 explicitly lists European Romani yek 'one' under *ēkka-. The source prints Zargari jekh [jek] 'one' (p. 127 §2.3.5) and explains that final aspirates may deaspirate. The same lemma recurs in the source numeral paradigm (p. 137 §3.5.1). Its separately imported yek spelling already has a rank-1 variant edge to this jekh record; that existing variant is not duplicated.",
    ),
    (
        "in the middle", "9804", "mádhya", ["f_z7fza2omi3nsk"],
        "Turner 9804 explicitly lists German/Hungarian/Welsh Romani maškər/maškár/maskal 'in, between' under mádhya. The source prints Zargari mośgār 'in the middle' among adverbs of place (p. 143 §3.20.1.3). Source §2.3.1 (p. 127) documents merger of voiceless nonaspirates with voiced consonants; the k/g difference is therefore source-compatible, while the vowel details are retained without a new sound-rule claim.",
    ),
]

accepted = []
audit = []
review = [
    "# Zargari: one and in the middle",
    "",
    "Two source-checked reflex records in two historical families. Standing autonomous authorization. Original PDF printed pp. 127, 143 and the numeral paradigm p. 137, glyph audit and imported keys, homepage searches and full CDIAL articles/addenda were checked. No lexical import or rebuild.",
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
            "parent": parent, "kind": "reflex", "citation": f"CDIAL[{parent} and addenda];{r['Source']}",
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
    ("śel", "The source prints 'one hundred' (pp. 137–138) and other Romani sources attest šel, but Turner’s specific śata entry and treatment of this Romani numeral have not been established; do not infer an etymon from form similarity alone."),
    ("pāng", "The source gives pāng/pāngl 'break' as present/perfect stems (p. 138), and Turner 9361 §2 has European Romani phag- 'to break'; the added nasal and vowel correspondence remain unexplained."),
]:
    matches = [r for r in all_forms.values() if r["Form"] == form and r["ID"] not in linked]
    assert len(matches) == 1, (form, matches)
    held.append({"record": {k: matches[0][k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}, "reason": reason})

assert len(accepted) == 2 and len(held) == 2
(D / "pass120-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(D / "pass120-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
review += ["", "Two held records:"]
review += [f"- **{h['record']['Form']}** ‘{h['record']['Gloss']}’: {h['reason']}" for h in held]
review += ["", "Two assignment rows affect two records, both `reflex`. Discovery: homepage `zargari-search027–028`, full CDIAL `zargari-primary024–025`, PDF images and `pass120-source-audit.json`. Save and validation pending."]
(D / "pass120-review.md").write_text("\n".join(review) + "\n")
print("accepted", len(accepted), "held", len(held))
