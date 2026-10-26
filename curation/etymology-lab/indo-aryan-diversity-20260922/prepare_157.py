"""Prepare Zargari from-below and quickly adverbs."""
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
        "f_jdhck3ch5ycvu", "5731",
        "Turner 5731 tala gives European Romani telal 'from under' alongside tel/tele 'under'. The source explicitly analyses Zargari ter-āl 'from the bottom' as an ablative adverb instead of tel-ār from teli 'bottom' (p. 132 §3.1.15). Three separately installed teli senses are already linked to 5731. The source's r/l alternation and productive -āl morphology are retained; this is the same inherited under/bottom family, not an unattested Sanskrit whole-word compound."),
    (
        "f_olwq5dvqxe5jw", "12484",
        "Turner 12484 śīghrá 'quick' explicitly gives European Romani sig/sigo as both adjective and adverb. The source prints Zargari sige 'quickly' in its manner-adverb list (p. 143 §3.20.1.4). The root and adverb sense match directly; final -e is retained as printed."),
]
forms = {r["ID"]: r for r in csv.DictReader(open("cldf/forms.csv")) if SOURCE in r["Source"]}
linked = {r["Form_ID"] for r in read_assignments() if r["Status"] == "accepted"}
linked.update(r["Child_ID"] for r in csv.DictReader(open("cldf/edges.csv")) if r["Rank"] == "1")
identities = {r["Form_ID"]: r["Source_Key"] for r in csv.DictReader(open("data/form-identities.csv")) if r["Source_Key"].startswith(SOURCE + ":")}
source_rows = list(csv.reader(open("data/other/forms/20260825-rezai-baghbidi-zargari.csv")))
raw_rows = list(csv.DictReader(open("data/other/forms/raw_data/20260825-rezai-baghbidi-zargari-audit.csv")))
for fid in ("f_xp5wgayzk7zlu", "f_z4by54xm7kptw", "f_mnfz27uvtuqik"):
    assert fid in linked

accepted = []
audit = []
for fid, parent, evidence in targets:
    r = forms[fid]
    assert fid not in linked and r["Language_ID"] == "Zarg"
    key = identities[fid]
    sm = [x for x in source_rows if x[0] == "Zarg" and x[3] == r["Gloss"] and x[7] == r["Source"] and x[10] == key]
    assert len(sm) == 1, (fid, sm)
    rm = [x for x in raw_rows if x["Status"] == "ingest" and key in x["Emitted_Keys"].split() and x["Final_Gloss"] == r["Gloss"]]
    assert len(rm) == 1, (fid, rm)
    accepted.append({
        "record": {k: r[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")},
        "parent": parent, "kind": "reflex",
        "citation": f"CDIAL[{parent} and addenda];{r['Source']}",
        "evidence": evidence + " Original PDF, glyph-decoded source audit and imported key checked; no transcription or identity change.",
    })
    audit.append({
        "id": fid, "source_key": key, "raw_unit": rm[0]["Unit_ID"],
        "raw_form": rm[0]["Raw_Form"], "original": r["Original"],
        "compiled_form": r["Form"], "printed_page": rm[0]["Printed_Page"],
        "record_hash": rm[0]["Record_SHA256"],
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
assert "Gy. eur. tel" in primary["5731"][0]["text"] and "telal 'from under'" in primary["5731"][0]["text"]
assert "Gy. eur. sig" in primary["12484"][0]["text"] and "adj. and adv." in primary["12484"][0]["text"]
(D / "zargari-primary034.json").write_text(json.dumps(primary, ensure_ascii=False, indent=2) + "\n")
(D / "pass157-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
(D / "pass157-decisions.json").write_text(json.dumps({"accepted": accepted, "held": []}, ensure_ascii=False, indent=2) + "\n")
(D / "pass157-review.md").write_text("\n".join([
    "# Zargari: from below and quickly",
    "",
    "Two source-checked adverbs under standing autonomous authorization. Original PDF printed pp. 132 and 143, glyph audit and exact installed keys, compiled comparisons and full Turner articles/addenda were checked. No lexical import or rebuild.",
    "",
    "| Zargari forms and meanings | Proposed etymology | Evidence |",
    "|---|---|---|",
    f"| **{forms[targets[0][0]]['Form']}** ‘from the bottom’ | **tala**, CDIAL **5731** | {targets[0][2]} |",
    f"| **{forms[targets[1][0]]['Form']}** ‘quickly’ | **śīghrá**, CDIAL **12484** | {targets[1][2]} |",
    "",
    "Two proposed assignment rows affect two records, both `reflex`. The exact-string `dukʰ-āv` 'I should see' was excluded: Turner 6376 dukhav is a pain causative, while the source makes the Zargari form a vowel-harmony alternant of dikʰ-āv 'see' (p. 129 §2.3.27). Earlier kʰil 'oil' semantic-scope hold remains. Discovery: `zargari-primary034`, PDF images and `pass157-source-audit.json`. Save and validation pending.",
]) + "\n")
print("accepted", len(accepted), "held 0")
