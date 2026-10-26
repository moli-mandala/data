"""Prepare Zargari hundred and transparent numeral compounds."""
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

hundred = "f_puhagaeulkuym"
two = "f_tp7o3vogig57i"
five = "f_ahg7g4o5kv5ay"
assert two in linked and five in linked and hundred not in linked

proposals = [
    (
        hundred, "12278", "reflex", None,
        "CDIAL[12278 and addenda]",
        "Turner 12278 śatá '100' explicitly cites European Romani šel. The source prints Zargari śel 'one hundred' in its numeral table (p. 137 §3.5.1) and explains the -i- connector before following numerals (p. 138 §3.5.1.1). This is the simple hundred stem.",
    ),
    (
        "f_cwaorgi7tkz7c", two, "component", [two, hundred],
        "CDIAL[6648];CDIAL[12278]",
        "The source prints duy-śel 'two hundred' (p. 138 §3.5.1) and explicitly states that numerals are compounded (§3.5.1.1). Its two source components are attested duy 'two' (already linked to CDIAL 6648) and śel 'hundred' (newly linked to CDIAL 12278). The component edge records this composition, not a separate whole-word Sanskrit ancestor.",
    ),
    (
        "f_zjmhydvstdca4", hundred, "component", [hundred, five],
        "CDIAL[12278];CDIAL[7655]",
        "The source explicitly prints śel-i-pānj 'one hundred and five' (p. 138 §3.5.1.1) as its example of -i- before the following numeral. Its two source lexical components are śel 'hundred' (newly linked to CDIAL 12278) and pānj 'five' (already linked to CDIAL 7655); -i- remains the source connector. This records components, not a single ancient compound etymon.",
    ),
]

accepted = []
audit = []
review = [
    "# Zargari: hundred and numeral compounds",
    "",
    "Three source-checked records: one reflex and two explicitly printed numeral compositions. Standing autonomous authorization. Original PDF printed pp. 137–138, glyph audit and imported keys, homepage search and full CDIAL article/addenda were checked. No lexical import or rebuild.",
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
    review.append(f"| {number} | **{r['Form']}** ‘{r['Gloss']}’ | **{kind}**, {citation_prefix.replace('CDIAL', 'Turner')} | {evidence} |")

held = []
for form, reason in [
    ("śel-sila", "The printed hundred–thousand composition is clear (p. 138), but the source's sila 'thousand' is not established as Indo-Aryan and may be Greek. Hold a full component analysis pending the immediate origin of sila."),
    ("śel-i-peyindā-sārāndā-śov", "The printed '196' form contains the known śel and śov elements but also mixed tens with unresolved contact histories. Do not assign a whole-word historical origin based on the two secure pieces alone."),
]:
    matches = [r for r in all_forms.values() if r["Form"] == form and r["ID"] not in linked]
    assert len(matches) == 1, (form, matches)
    held.append({"record": {k: matches[0][k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}, "reason": reason})

assert len(accepted) == 3 and len(held) == 2
(D / "pass125-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(D / "pass125-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
review += ["", "Two held records:"]
review += [f"- **{h['record']['Form']}** ‘{h['record']['Gloss']}’: {h['reason']}" for h in held]
review += ["", "Three assignment records yield one reflex and two component edges, preserving source numeral composition. Discovery: homepage `zargari-search029`, full CDIAL `zargari-primary026`, PDF images and `pass125-source-audit.json`. Save and validation pending."]
(D / "pass125-review.md").write_text("\n".join(review) + "\n")
print("accepted", len(accepted), "held", len(held))
