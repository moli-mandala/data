"""Prepare source-checked Zargari 'few, little' comparison."""
import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

D = Path("curation/etymology-lab/indo-aryan-diversity-20260922")
FID = "f_wyvbq5rb4lv7c"
HEAD = "3792"
SOURCE = "rezaibaghbidi2003zargari"

forms = {r["ID"]: r for r in csv.DictReader(open("cldf/forms.csv")) if SOURCE in r["Source"]}
form = forms[FID]
linked = {r["Form_ID"] for r in read_assignments() if r["Status"] == "accepted"}
linked.update(r["Child_ID"] for r in csv.DictReader(open("cldf/edges.csv")) if r["Rank"] == "1")
assert FID not in linked and form["Language_ID"] == "Zarg"
assert (form["Form"], form["Original"], form["Gloss"]) == ("xaynri", "xajnri", "few; little")

keys = [r["Source_Key"] for r in csv.DictReader(open("data/form-identities.csv")) if r["Form_ID"] == FID]
assert keys == ["rezaibaghbidi2003zargari:p143:s3.20.1.5:i4"]
key = keys[0]
source_rows = list(csv.reader(open("data/other/forms/20260825-rezai-baghbidi-zargari.csv")))
assert len([r for r in source_rows if r[0] == "Zarg" and r[3] == form["Gloss"] and r[7] == form["Source"] and r[10] == key]) == 1
raw_rows = list(csv.DictReader(open("data/other/forms/raw_data/20260825-rezai-baghbidi-zargari-audit.csv")))
raw = [r for r in raw_rows if r["Status"] == "ingest" and key in r["Emitted_Keys"].split() and r["Final_Gloss"] == form["Gloss"]]
assert len(raw) == 1 and raw[0]["Raw_Form"] == "xaj(n)ri"
duplicate = [r for r in raw_rows if r["Status"] == "dup" and key in r["Emitted_Keys"].split() and r["Printed_Page"] == "137"]
assert len(duplicate) == 1 and "xajri" in duplicate[0]["Raw_Form"]

primary = None
for page, html in enumerate(pickle.load(open("data/cdial/cdial.pickle", "rb")), 1):
    if f"<number>{HEAD}</number>" not in html:
        continue
    for paragraph in BeautifulSoup(html, "html.parser").select("p"):
        number = paragraph.find("number")
        if number and number.get_text(strip=True) == HEAD:
            primary = {"page": page, "url": f"https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page={page}", "html": str(paragraph), "text": paragraph.get_text(" ", strip=True)}
    if primary:
        break
assert primary and all(s in primary["text"] for s in ("xai̦i", "xanrík", "a little"))
(D / "zargari-primary037.json").write_text(json.dumps({HEAD: [primary]}, ensure_ascii=False, indent=2) + "\n")

evidence = ("Turner 3792 khaṇḍá¹ 'broken, fragment' explicitly lists Southeast European Romani xai̦i and Greek Romani xanrík 'a little' (also xandí). The survey prints (je) xajri 'few, a few, little, a little' at p. 137 §3.3.7.1 and xaj(n)ri 'few; little' at p. 143 §3.20.1.5. The optional n and xaj-/xanr- frame align with these Romani comparanda; the exact suffixes differ, so this is a stem-level reflex, not an assertion that Zargari preserves the Greek form's final -k. The source's (je) is optional and excluded from the installed headword. Original PDF, glyph-decoded source audit and imported key checked; no transcription or identity change.")
accepted = [{"record": {k: form[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}, "parent": HEAD, "kind": "reflex", "citation": f"CDIAL[{HEAD} and addenda];{form['Source']}", "evidence": evidence}]
(D / "pass172-decisions.json").write_text(json.dumps({"accepted": accepted, "held": []}, ensure_ascii=False, indent=2) + "\n")
(D / "pass172-source-audit.json").write_text(json.dumps([{"id": FID, "source_key": key, "raw_unit": raw[0]["Unit_ID"], "raw_form": raw[0]["Raw_Form"], "original": form["Original"], "compiled_form": form["Form"], "printed_page": raw[0]["Printed_Page"], "record_hash": raw[0]["Record_SHA256"], "duplicate_page": duplicate[0]["Printed_Page"], "duplicate_hash": duplicate[0]["Record_SHA256"]}], ensure_ascii=False, indent=2) + "\n")
(D / "pass172-review.md").write_text("\n".join([
    "# Zargari: few, little",
    "",
    "One source-checked reflex record under standing autonomous authorization. Printed pp. 137 and 143 visually checked; exact raw/installed identity and full Turner article audited. No lexical import or rebuild.",
    "",
    "| Zargari form | Proposed etymology | Evidence |",
    "|---|---|---|",
    f"| **{form['Form']}** ‘{form['Gloss']}’ | **khaṇḍá¹**, CDIAL **{HEAD}** | {evidence} |",
    "",
    "The separately imported xayri spelling is already a compiled rank-1 variant of xaynri and gets no duplicate row. Ašti/n-ašti 'can/cannot' remain held because Turner's addendum raises doubt about a Gypsy asti < OIA ásti derivation; pālo 'back' remains held under competing Turner heads. Discovery: `zargari-primary037.json`, PDF images and `pass172-source-audit.json`. Save and validation pending.",
]) + "\n")
print("accepted", len(accepted), "held 0")
