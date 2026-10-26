"""Prepare two source-parsed Zargari negated verbs with exact Romani comparanda."""
import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

D = Path("curation/etymology-lab/indo-aryan-diversity-20260922")
SOURCE = "rezaibaghbidi2003zargari"
targets = [
    (
        "f_hpuh6m5b7cyfs",
        "5193",
        "Turner 5193 jānā́ti 'knows' explicitly gives European and Armenian Romani ǰan-; the printed source has Zargari nā́-džān-āvās 'I did not know' (p. 130 §2.5.3). The source explicitly identifies stressed nā- as its negative prefix, and p. 139 §3.7.1 identifies -āvās as first-person singular imperfect. The inherited know stem is the target; negative and person-tense morphology are retained in the attestation.",
    ),
    (
        "f_y7rxlvrkv3jhs",
        "12598",
        "Turner 12598 śr̥ṇṓti 'hears' explicitly gives European Romani šun-; the printed source has Zargari nā-šundās 'he did not hear' (p. 144 §3.20.4.1), explicitly under its nā- negative prefix. The source describes -d as a perfect-stem marker after n (p. 138 §3.6.2) and -ās as the third-person singular perfect ending (p. 139 §3.7.1). The inherited hear stem matches the diagnostic Romani comparison.",
    ),
]

forms = {r["ID"]: r for r in csv.DictReader(open("cldf/forms.csv")) if SOURCE in r["Source"]}
linked = {r["Form_ID"] for r in read_assignments() if r["Status"] == "accepted"}
linked.update(r["Child_ID"] for r in csv.DictReader(open("cldf/edges.csv")) if r["Rank"] == "1")
identities = {r["Form_ID"]: r["Source_Key"] for r in csv.DictReader(open("data/form-identities.csv")) if r["Source_Key"].startswith(SOURCE + ":")}
source_rows = list(csv.reader(open("data/other/forms/20260825-rezai-baghbidi-zargari.csv")))
raw_rows = list(csv.DictReader(open("data/other/forms/raw_data/20260825-rezai-baghbidi-zargari-audit.csv")))

accepted = []
audit = []
for fid, parent, evidence in targets:
    r = forms[fid]
    assert fid not in linked and r["Language_ID"] == "Zarg"
    key = identities[fid]
    source_matches = [x for x in source_rows if x[0] == "Zarg" and x[3] == r["Gloss"] and x[7] == r["Source"] and x[10] == key]
    assert len(source_matches) == 1, (fid, source_matches)
    raw_matches = [x for x in raw_rows if x["Status"] == "ingest" and key in x["Emitted_Keys"].split() and x["Final_Gloss"] == r["Gloss"]]
    assert len(raw_matches) == 1, (fid, raw_matches)
    accepted.append({
        "record": {k: r[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")},
        "parent": parent,
        "kind": "reflex",
        "citation": f"CDIAL[{parent} and addenda];{r['Source']}",
        "evidence": evidence + " Original PDF, glyph-decoded source audit and imported key checked; no transcription or identity change.",
    })
    audit.append({
        "id": fid, "source_key": key, "raw_unit": raw_matches[0]["Unit_ID"],
        "raw_form": raw_matches[0]["Raw_Form"], "original": r["Original"],
        "compiled_form": r["Form"], "printed_page": raw_matches[0]["Printed_Page"],
        "record_hash": raw_matches[0]["Record_SHA256"],
    })

wanted = {parent for _, parent, _ in targets}
primary = {}
for page, html in enumerate(pickle.load(open("data/cdial/cdial.pickle", "rb")), 1):
    if not any(f"<number>{n}</number>" in html for n in wanted - primary.keys()):
        continue
    for paragraph in BeautifulSoup(html, "html.parser").select("p"):
        number = paragraph.find("number")
        if number and number.get_text(strip=True) in wanted:
            n = number.get_text(strip=True)
            primary[n] = [{
                "page": page,
                "url": f"https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page={page}",
                "html": str(paragraph),
                "text": paragraph.get_text(" ", strip=True),
            }]
    if len(primary) == len(wanted):
        break
assert len(primary) == 2
assert "Gy. eur. arm. ǰan" in primary["5193"][0]["text"]
assert "Gy. eur. šun" in primary["12598"][0]["text"]
(D / "zargari-primary032.json").write_text(json.dumps(primary, ensure_ascii=False, indent=2) + "\n")
(D / "pass147-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
(D / "pass147-decisions.json").write_text(json.dumps({"accepted": accepted, "held": []}, ensure_ascii=False, indent=2) + "\n")
(D / "pass147-review.md").write_text("\n".join([
    "# Zargari: negated know and hear verbs",
    "",
    "Two source-checked reflex records under standing autonomous authorization. Original PDF printed pp. 130, 138–139, 144, glyph audit and imported keys, compiled comparisons and full Turner articles/addenda were checked. No lexical import or rebuild.",
    "",
    "| Zargari forms and meanings | Proposed etymology | Evidence |",
    "|---|---|---|",
    f"| **{forms[targets[0][0]]['Form']}** ‘I did not know’ | **jānā́ti**, CDIAL **5193** | {targets[0][2]} |",
    f"| **{forms[targets[1][0]]['Form']}** ‘he did not hear’ | **śr̥ṇṓti**, CDIAL **12598** | {targets[1][2]} |",
    "",
    "Two proposed assignment rows affect two records, both `reflex`. The earlier held come-preterite forms remain excluded: Turner 1200 explicitly says that preterites are usually from ā́gata-, and the Zargari ay-/yi- branch needs analysis. Discovery: `zargari-primary032`, PDF images and `pass147-source-audit.json`. Save and validation pending.",
]) + "\n")
print("accepted", len(accepted), "held 0")
