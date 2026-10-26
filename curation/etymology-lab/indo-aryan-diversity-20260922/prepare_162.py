"""Prepare Zargari distributive day/night adverbs."""
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
        "f_racovmazrgcxm", "6333",
        "Turner 6333 divasá 'day' gives European Romani dives and Paluli dīs; the separate source Zargari dis 'day' is already linked to 6333. The grammar explicitly prints dis-este 'every day' as a locative expression of 'every' (p. 132 §3.1.16) and lists diseste in its temporal adverbs (p. 143 §3.20.1.2). The local ending is retained; the inherited day stem is the etymological target."),
    (
        "f_46f37btydv25k", "10702",
        "Turner 10702 rā́trī 'night' gives European Romani rat and locative rati 'at night'. The grammar explicitly prints rājt-ādu 'every night' as another locative expression of 'every' (p. 132 §3.1.16), lists rājtādu among temporal adverbs, and separately prints rāt 'night' in the phrase nāklās rāt 'last night' (p. 143 §3.20.1.2). The source j glide and the locative ending are retained; no separately installed bare-night lemma is assumed."),
]
forms = {r["ID"]: r for r in csv.DictReader(open("cldf/forms.csv")) if SOURCE in r["Source"]}
linked = {r["Form_ID"] for r in read_assignments() if r["Status"] == "accepted"}
edges = list(csv.DictReader(open("cldf/edges.csv")))
linked.update(r["Child_ID"] for r in edges if r["Rank"] == "1")
assert "f_izw3vw3vizd6o" in linked
assert any(e["Child_ID"] == "f_iirlhtpuw5m24" and e["Parent_ID"] == "f_racovmazrgcxm" and e["Rank"] == "1" for e in edges)
assert any(e["Child_ID"] == "f_vc34a7yim2axy" and e["Parent_ID"] == "f_46f37btydv25k" and e["Rank"] == "1" for e in edges)
identities = {r["Form_ID"]: r["Source_Key"] for r in csv.DictReader(open("data/form-identities.csv")) if r["Source_Key"].startswith(SOURCE + ":")}
source_rows = list(csv.reader(open("data/other/forms/20260825-rezai-baghbidi-zargari.csv")))
raw_rows = list(csv.DictReader(open("data/other/forms/raw_data/20260825-rezai-baghbidi-zargari-audit.csv")))

accepted = []
audit = []
for fid, parent, evidence in targets:
    r = forms[fid]
    assert fid not in linked and r["Language_ID"] == "Zarg"
    key = identities[fid]
    sm = [x for x in source_rows if x[0] == "Zarg" and x[3] == r["Gloss"] and x[7] == r["Source"] and x[10] == key]
    assert len(sm) == 1, (fid, sm)
    rm = [x for x in raw_rows if x["Status"] == "ingest" and key in x["Emitted_Keys"].split() and x["Final_Gloss"] == r["Gloss"]]
    dup = [x for x in raw_rows if x["Status"] == "dup" and key in x["Emitted_Keys"].split()]
    assert len(rm) == len(dup) == 1, (fid, rm, dup)
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
        "grammar_duplicate_unit": dup[0]["Unit_ID"],
        "grammar_duplicate_raw": dup[0]["Raw_Form"],
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
assert "Gy. eur. dives" in primary["6333"][0]["text"]
assert "Gy. eur. rat" in primary["10702"][0]["text"]
(D / "zargari-primary035.json").write_text(json.dumps(primary, ensure_ascii=False, indent=2) + "\n")
(D / "pass162-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
(D / "pass162-decisions.json").write_text(json.dumps({"accepted": accepted, "held": []}, ensure_ascii=False, indent=2) + "\n")
(D / "pass162-review.md").write_text("\n".join([
    "# Zargari: every day and every night",
    "",
    "Two source-checked temporal adverbs under standing autonomous authorization. Original PDF printed pp. 132 and 143, source glyph audit (including duplicate grammar occurrences), exact installed keys, compiled variant links and full Turner articles/addenda were checked. No lexical import or rebuild.",
    "",
    "| Zargari forms and meanings | Proposed etymology | Evidence |",
    "|---|---|---|",
    f"| **{forms[targets[0][0]]['Form']}** ‘every day’ | **divasá**, CDIAL **6333** | {targets[0][2]} |",
    f"| **{forms[targets[1][0]]['Form']}** ‘every night’ | **rā́trī**, CDIAL **10702** | {targets[1][2]} |",
    "",
    "Two proposed assignment rows affect two records, both `reflex` of inherited bases with source locative morphology. The separately installed dis-este and rāyt-ādu spellings have existing compiled rank-1 variant edges and are not duplicated. Other phrases with partially unknown deictic or contact members remain excluded. Discovery: `zargari-primary035`, PDF images and `pass162-source-audit.json`. Save and validation pending.",
]) + "\n")
print("accepted", len(accepted), "held 0")
