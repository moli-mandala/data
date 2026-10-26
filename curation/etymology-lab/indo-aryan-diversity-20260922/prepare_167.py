"""Prepare Zargari smell and source-segmented prohibitive verbs."""
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
        "f_n2rtta4bcmpc6", "4014",
        "Turner 4014 gandhá 'smell' explicitly gives Greek Romani khan 'smell' and Welsh Romani khand, with a Romani derived smell verb. The source prints Zargari khān 'smell' (p. 131 §3.1.8). Its long vowel is retained; no Persian gand(a) donor is claimed for this kh-form.",
    ),
    (
        "f_hfqxbqnar4uwi", "1031",
        "Turner 1031 ā́kṣēti explicitly gives European Romani ač- 'to stay, live, stand'. The source prints m-āč 'Do not stay!' < mā-āč (p. 128 §2.3.11) and identifies mā- as its productive prohibitive prefix (p. 144 §3.20.4.3). The link targets the inherited stay stem while retaining source negative and imperative morphology.",
    ),
    (
        "f_qmdr5zfwk3b6a", "1174",
        "Turner 1174 ā́nayati 'fetches' explicitly gives European Romani an- 'bring'. The source prints m-ān 'Do not bring!' < mā-ān (p. 128 §2.3.11) and identifies mā- as its productive prohibitive prefix (p. 144 §3.20.4.3). The inherited bring stem is linked with local negative and imperative morphology retained.",
    ),
    (
        "f_unhq5ujfibk5c", "4205",
        "Turner 4205 §1 gupháti explicitly gives European Romani khuv- 'to plait, weave, spin'; the other numbered branches are distinct. The source prints stressed mā́-khuv 'Do not weave!' and identifies mā- as the prohibitive prefix (p. 130 §2.5.3; cf. p. 144 §3.20.4.3). The link targets the inherited weave stem while retaining source negative and imperative morphology.",
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
assert set(primary) == wanted
assert "gr. khan" in primary["4014"][0]["text"]
assert "eur. ač" in primary["1031"][0]["text"]
assert "eur. an" in primary["1174"][0]["text"]
assert "eur. khuv" in primary["4205"][0]["text"]
(D / "zargari-primary036.json").write_text(json.dumps(primary, ensure_ascii=False, indent=2) + "\n")
(D / "pass167-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
(D / "pass167-decisions.json").write_text(json.dumps({"accepted": accepted, "held": []}, ensure_ascii=False, indent=2) + "\n")
rows = [
    "# Zargari: smell and prohibitive verbs",
    "",
    "Four source-checked records under standing autonomous authorization. Original PDF printed pp. 128, 130–131 and 144, glyph audit and exact installed keys, compiled comparisons and full Turner articles/addenda were checked. No lexical import or rebuild.",
    "",
    "| Zargari forms and meanings | Proposed etymology | Evidence |",
    "|---|---|---|",
]
heads = {"4014": "gandhá", "1031": "ā́kṣēti", "1174": "ā́nayati", "4205": "gupháti §1"}
for fid, parent, evidence in targets:
    rows.append(f"| **{forms[fid]['Form']}** ‘{forms[fid]['Gloss']}’ | **{heads[parent]}**, CDIAL **{parent}** | {evidence} |")
rows += [
    "",
    "Four proposed assignment rows affect four records, all `reflex`. The two prohibitives m-āc and mā́kʰuv were earlier held because an affirmative stem record is not separately installed. This pass applies the same source-segmented negative-verb treatment already used for mépʰen, ne-pʰendom, nā́jānāvās and nā-śundās: the inherited verbal stem receives a reflex link, while the productive prohibitive prefix remains in the original form. No source node is added. Discovery: `zargari-primary036`, PDF images and `pass167-source-audit.json`. Save and validation pending.",
]
(D / "pass167-review.md").write_text("\n".join(rows) + "\n")
print("accepted", len(accepted), "held 0")
