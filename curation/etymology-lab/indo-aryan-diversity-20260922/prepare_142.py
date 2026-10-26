"""Prepare source-parsed Zargari say-verb inflections."""
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

assert "f_rc4hrjqv7zvxo" in linked and next(r for r in csv.DictReader(open("cldf/edges.csv")) if r["Child_ID"] == "f_6grdp4jazapl2")["Parent_ID"] == "f_yyjj57acfun4o"
evidence = (
    "Turner 9383 bhánati directly cites European Romani phen- 'to speak'. The source prints the Zargari present stem phen- in productive infinitive phenipej 'to say; to tell' (p. 142 §3.17.3.1), already linked to 9383. It separately gives te-phen-es > ti-phen-is 'you should say' under vowel harmony (p. 129 §2.3.26), mé-phen 'Do not say!' with stressed prohibitive mā- (p. 130 §2.5.3), and ne-phendom 'I did not say' under negative particle nā-/ne- (p. 144 §3.20.4.1). All three are ordinary source-parsed inflections of the same inherited phen- stem, with local mood/negation retained. The te-phen-es spelling already has a compiled rank-1 variant edge to ti-phen-is and is not duplicated."
)
ids = ["f_yyjj57acfun4o", "f_pc45ww5ecbo6k", "f_zfe42xy6rpd4g"]
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
        "parent": "9383", "kind": "reflex", "citation": f"CDIAL[9383 and addenda];{r['Source']}",
        "evidence": evidence + " Original PDF, glyph-decoded source audit and imported key checked; no transcription or identity change.",
    })
    audit.append({
        "id": fid, "source_key": key, "raw_unit": raw_matches[0]["Unit_ID"],
        "raw_form": raw_matches[0]["Raw_Form"], "final_form": raw_matches[0]["Final_Forms"],
        "printed_page": raw_matches[0]["Printed_Page"], "record_hash": raw_matches[0]["Record_SHA256"],
    })

held = []
assert len(accepted) == 3
(D / "pass142-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(D / "pass142-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
review = [
    "# Zargari: say-verb mood and negation",
    "",
    "Three source-checked reflex records from the same inherited phen- stem. Standing autonomous authorization. Original PDF printed pp. 129–130, 142, 144, glyph audit and imported keys, homepage/compiled comparisons and full CDIAL article/addenda were checked. No lexical import or rebuild.",
    "",
    "| Zargari forms and meanings | Proposed etymology | Evidence |",
    "|---|---|---|",
    f"| **{' / '.join(all_forms[x]['Form'] for x in ids)}** ‘you should say; do not say; I did not say’ | **bhánati**, CDIAL **9383** | {evidence} |",
    "",
    "Three assignment rows affect three records, all `reflex`; the existing te-pʰen-es variant edge remains the source spelling relationship. Discovery: `zargari-primary031`, PDF images and `pass142-source-audit.json`. Save and validation pending.",
]
(D / "pass142-review.md").write_text("\n".join(review) + "\n")
print("accepted", len(accepted), "held", len(held))
