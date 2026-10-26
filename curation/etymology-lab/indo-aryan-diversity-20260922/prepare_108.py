"""Prepare high-confidence Zargari ant, walnut and head links."""
import csv
import json
from pathlib import Path

from etymology_assignments import read_assignments


D = Path("curation/etymology-lab/indo-aryan-diversity-20260922")
SOURCE = "rezaibaghbidi2003zargari"
all_forms = {r["ID"]: r for r in csv.DictReader(open("cldf/forms.csv")) if SOURCE in r["Source"]}
linked = {r["Form_ID"] for r in read_assignments() if r["Status"] == "accepted"}
linked.update(r["Child_ID"] for r in csv.DictReader(open("cldf/edges.csv")) if r["Rank"] == "1")
identities = {
    r["Form_ID"]: r["Source_Key"] for r in csv.DictReader(open("data/form-identities.csv"))
    if r["Source_Key"].startswith(SOURCE + ":")
}
source_rows = list(csv.reader(open("data/other/forms/20260825-rezai-baghbidi-zargari.csv")))
raw_rows = list(csv.DictReader(open("data/other/forms/raw_data/20260825-rezai-baghbidi-zargari-audit.csv")))

families = [
    ("ant", "3193", "kīṭá §1", ["f_fvuowf4weessm"],
     "Turner 3193 directly cites European Romani kiri 'ant' under kīṭá §1. The source prints Zargari kiri [khiri], giri 'ant' in §2.3.3 (p. 127) while describing optional aspiration and voicing before front vowels. The installed kiri is this attestation."),
    ("walnut", "48", "akṣōṭa", ["f_i3m57rzqoignk"],
     "Turner 48 directly cites Persian Romani āγūr/akūl 'walnut' and European Romani akhor 'nut'. The source explicitly compares Zargari ākkur 'walnut' with Early Romani akhor and places the doubled kk under its gemination rule (§2.3.7, p. 127). Turner notes an Iranian ultimate source for akṣōṭa in the addendum; this assignment concerns the attested Romani-line reflex, not an assertion of ultimate Sanskrit origin."),
    ("head", "12452", "śíras", ["f_nqw7uuzycwd5i"],
     "Turner 12452 directly cites European Romani šero 'head' under śíras. The source prints Zargari śoru 'head' (p. 132 §3.1.9) and explicitly derives the singular from *šoro while discussing vowel harmony. This explains the source's -u against the Turner Romani comparator."),
]

accepted = []
audit = []
review = [
    "# Zargari: ant, walnut and head",
    "",
    "Three source-checked reflex records across three historical families. Standing autonomous authorization. Original PDF printed pp. 127, 132, 147, glyph audit and imported keys, homepage searches and full CDIAL articles/addenda were checked. No lexical import or rebuild.",
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
        source_matches = [
            x for x in source_rows if x[0] == "Zarg" and x[3] == r["Gloss"]
            and x[7] == r["Source"] and x[10] == key
        ]
        assert len(source_matches) == 1, (fid, source_matches)
        raw_matches = [
            x for x in raw_rows if x["Status"] == "ingest" and key in x["Emitted_Keys"].split()
            and x["Final_Gloss"] == r["Gloss"]
        ]
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
    ("kirmo", "Turner 3438 lists European Romani kirmo 'worm' but explicitly warns that Romani may instead have borrowed Persian kirm; the source p. 128 §2.3.14 only discusses palatalization and does not resolve the immediate donor."),
    ("murś", "Turner 9828 compares European Romani murš to manuṣyà crossed with púruṣa, while the source itself explicitly compares Zargari murš with Sanskrit māriṣa- on p. 147. The conflicting historical branches need editorial adjudication."),
    ("dār", "The source prints dār 'door' on p. 131, but Turner’s door comparanda distinguish inherited European Romani vudar from Indo-Aryan dar; direct Persian dar contact remains possible."),
]:
    matches = [r for r in all_forms.values() if r["Form"] == form and r["ID"] not in linked]
    assert len(matches) == 1, (form, matches)
    r = matches[0]
    held.append({"record": {k: r[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}, "reason": reason})

assert len(accepted) == 3 and len(held) == 3
(D / "pass108-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(D / "pass108-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
review += ["", "Three held records:"]
review += [f"- **{h['record']['Form']}** ‘{h['record']['Gloss']}’: {h['reason']}" for h in held]
review += ["", "Three assignment rows affect three records, all `reflex`. Discovery: homepage `zargari-search022–024`, full CDIAL `zargari-primary020`, PDF images and `pass108-source-audit.json`. Save and validation pending."]
(D / "pass108-review.md").write_text("\n".join(review) + "\n")
print("accepted", len(accepted), "held", len(held))
