"""Prepare Zargari shadow and transparently compounded eleven–sixteen."""
import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

D = Path("curation/etymology-lab/indo-aryan-diversity-20260922")
SOURCE = "rezaibaghbidi2003zargari"
TEN = "f_opnqw3nlsxviu"
numerals = [
    ("f_pkxaofkhox6o2", "f_iao7cbp7et26g", "2462-2", "eleven"),
    ("f_na4lv2e2uphto", "f_tp7o3vogig57i", "6648", "twelve"),
    ("f_xbivslzucgr3e", "f_g3cnz3td3oxjc", "5994-3", "thirteen"),
    ("f_ofsbm5qgcuyks", "f_7vled5oasq42i", "4655", "fourteen"),
    ("f_67m47rd4qi7wg", "f_ahg7g4o5kv5ay", "7655", "fifteen"),
    ("f_si5h4ye2qs4vu", "f_oqd2hnzthhnfy", "12803-2", "sixteen"),
]
all_forms = {r["ID"]: r for r in csv.DictReader(open("cldf/forms.csv")) if SOURCE in r["Source"]}
linked = {r["Form_ID"] for r in read_assignments() if r["Status"] == "accepted"}
edges = list(csv.DictReader(open("cldf/edges.csv")))
linked.update(r["Child_ID"] for r in edges if r["Rank"] == "1")
identities = {r["Form_ID"]: r["Source_Key"] for r in csv.DictReader(open("data/form-identities.csv")) if r["Source_Key"].startswith(SOURCE + ":")}
source_rows = list(csv.reader(open("data/other/forms/20260825-rezai-baghbidi-zargari.csv")))
raw_rows = list(csv.DictReader(open("data/other/forms/raw_data/20260825-rezai-baghbidi-zargari-audit.csv")))
assert {r["Etymon_ID"] for r in read_assignments() if r["Form_ID"] == TEN and r["Status"] == "accepted"} == {"6227"}
assert any(e["Child_ID"] == "f_iao7cbp7et26g" and e["Parent_ID"] == "f_3mm3to2hqjcky" and e["Rank"] == "1" for e in edges)

accepted = []
audit = []

def add(fid, parent, kind, citation, evidence, components=None):
    r = all_forms[fid]
    assert fid not in linked and r["Language_ID"] == "Zarg"
    key = identities[fid]
    sm = [x for x in source_rows if x[0] == "Zarg" and x[3] == r["Gloss"] and x[7] == r["Source"] and x[10] == key]
    assert len(sm) == 1, (fid, sm)
    rm = [x for x in raw_rows if x["Status"] == "ingest" and key in x["Emitted_Keys"].split() and x["Final_Gloss"] == r["Gloss"]]
    assert len(rm) == 1, (fid, rm)
    x = {
        "record": {k: r[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")},
        "parent": parent, "kind": kind, "citation": citation,
        "evidence": evidence + " Original PDF, glyph-decoded source audit and imported key checked; no transcription or identity change.",
    }
    if components:
        x["components"] = components
    accepted.append(x)
    audit.append({
        "id": fid, "source_key": key, "raw_unit": rm[0]["Unit_ID"],
        "raw_form": rm[0]["Raw_Form"], "original": r["Original"],
        "compiled_form": r["Form"], "printed_page": rm[0]["Printed_Page"],
        "record_hash": rm[0]["Record_SHA256"],
    })

shadow = (
    "The source prints Zargari čhal 'shadow' and explicitly compares Early Romani učhal > čal while discussing merger of aspirates and voiced affricates (p. 127 §2.3.1). Turner 2542 *ōjjhalla 'shade, screen, veil' cites European Romani učhal and Welsh Romani učal 'shade', probably from *avacchāda-. Turner 763 also indexes those same Romani comparanda under the deeper *avacchāda-; 2542 is the nearer formation for this inherited stem. The source's čh is retained."
)
add(
    "f_dsjp74xvq6ca2", "2542", "reflex",
    f"CDIAL[2542];CDIAL[763];{all_forms['f_dsjp74xvq6ca2']['Source']}", shadow,
)

num_evidence = (
    "The source explicitly prints the additive deš- compounds for 11–16 together in its cardinal-number table (p. 137 §3.5.1). Their first member is source deś 'ten', already linked to Turner 6227, which directly gives European Romani deš. The second member is the separately installed source numeral, already linked to its own specific Turner branch. These are synchronic components of a Zargari numeral, not proposed reflexes of the Sanskrit fused teen numerals."
)
for fid, atom, cdial, gloss in numerals:
    assert all_forms[fid]["Gloss"] == gloss
    assert atom in linked and all_forms[atom]["Language_ID"] == "Zarg"
    assert all_forms[fid]["Form"].startswith("deś-")
    note = " For 11, source jek is the installed yek variant with a compiled rank-1 edge to source yekʰ, already linked to Turner 2462 §2." if gloss == "eleven" else ""
    add(
        fid, TEN, "component",
        f"CDIAL[6227];CDIAL[{cdial}];{all_forms[fid]['Source']}",
        num_evidence + note,
        [TEN, atom],
    )

wanted = {"2542", "763", "6227", "2462", "6648", "5994", "4655", "7655", "12803"}
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
assert "Gy. eur. učhal" in primary["2542"][0]["text"]
assert "Gy. eur. (Sofia) učhál" in primary["763"][0]["text"]
assert "Gy. eur. deš" in primary["6227"][0]["text"]
(D / "zargari-primary033.json").write_text(json.dumps(primary, ensure_ascii=False, indent=2) + "\n")
(D / "pass152-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
(D / "pass152-decisions.json").write_text(json.dumps({"accepted": accepted, "held": []}, ensure_ascii=False, indent=2) + "\n")
rows = [
    "# Zargari: shadow and additive teens",
    "",
    "Seven source-defined records under standing autonomous authorization. Original PDF printed pp. 127 and 137, glyph audit and exact installed keys, compiled rank-1 ancestry and full Turner articles/addenda were checked. No lexical import or rebuild.",
    "",
    "| Zargari forms and meanings | Proposed etymology | Evidence |",
    "|---|---|---|",
    f"| **cʰal** ‘shadow’ | **\\*ōjjhalla**, CDIAL **2542** (Turner also indexes the Romani form at **763**) | {shadow} |",
]
for fid, atom, cdial, gloss in numerals:
    rows.append(f"| **{all_forms[fid]['Form']}** ‘{gloss}’ | Components: source **deś** ‘ten’ (CDIAL **6227**) + **{all_forms[atom]['Form']}** (CDIAL **{cdial}**) | Source p. 137 explicitly lists {all_forms[fid]['Original']} as {gloss}; the two source numerals are independently installed and already linked to the cited Turner branches. |")
rows += [
    "",
    "Thirteen proposed assignment rows affect seven records: one reflex and twelve ordered component rows. The source's Greek-derived seven/eight/nine remain excluded from 17–19 until immediate-donor representation is settled. Discovery: `zargari-primary033`, PDF images and `pass152-source-audit.json`. Save and validation pending.",
]
(D / "pass152-review.md").write_text("\n".join(rows) + "\n")
print("accepted", len(accepted), "held 0; proposed rows 13")
