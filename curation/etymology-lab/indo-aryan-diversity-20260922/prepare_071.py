"""Prepare four source-backed Buddruss Shina analyses."""

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OWNER = ROOT / "data/other/forms/etymologies/20260828-buddruss-shina-raetsel.csv"
SOURCE = ROOT / "data/other/forms/20260828-buddruss-shina-raetsel.csv"
OLD = {r["Form_ID"] for r in csv.DictReader(OWNER.open()) if r["Status"] == "accepted"}
source_keys = {(r[2], r[3], r[7]) for r in csv.reader(SOURCE.open())}

by_form = {}
for r in csv.DictReader((ROOT / "cldf/forms.csv").open()):
    if r["Source"].startswith("buddruss-shina1996["):
        by_form[r["Form"]] = r

P = [
    ("alák b", ["f_c5zruehbzg4qw", "f_gbbfrmlcypdas"], "component", "Hindi-Urdu alag + Shina b-", "Buddruss explicitly marks alák b- 'separate' as a loan from Hindi/Urdu alag. The first component is the attested Hindi alag 'separate' (CDIAL 700); the second is Buddruss's separate Shina b- 'become' head on the same page. Two ordered components preserve the light-verb construction rather than presenting the whole phrase as a single inherited stem."),
    ("cilím", "f_234geq2jdbf5i", "borrowed", "from Hindi cilam 'hookah receptacle'", "Buddruss explicitly derives cilím 'water pipe' from Hindi/Persian čilam. Jambu's attested Hindi cilam 'receptacle part of hookah' (Strand) is the immediate documented Indo-Aryan donor; the Persian stage remains in the note, rather than skipping over the Hindi evidence."),
]

decisions = {"accepted": [], "held": []}
table = []
for i, (form, target, kind, label, evidence) in enumerate(P, 1):
    r = by_form[form]
    assert r["ID"] not in OLD
    assert (r["Original"], r["Gloss"], r["Source"]) in source_keys
    record = {k: r[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}
    item = {"record": record, "citation": record["Source"], "evidence": evidence}
    if isinstance(target, list):
        item["components"] = target
        item["kind"] = "component"
    else:
        item["parent"] = target
        item["kind"] = kind
    if form == "alák b":
        item["citation"] += ";CDIAL[700]"
    elif form == "cilím":
        item["citation"] += ";strand[Hindi cilam]"
    decisions["accepted"].append(item)
    table.append(f"| {i} | **{form}** ‘{r['Gloss']}’ | **{label}** ({kind}) | {evidence} |")

(HERE / "pass071-decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2) + "\n")
review = """# Buddruss 1996 Shina riddles: source-identified loans

Two source-backed analyses under standing autonomous authorization. Both glossary heads and their claims were checked against the 300 dpi scan, printed pp. 40–41, alongside the installed source records. The source's many explicitly tentative Burushaski/Iranian comparisons remain unlinked.

| # | Gilgit Shina | Proposed etymology | Evidence |
|---|---|---|
""" + "\n".join(table) + "\n\n" + """Three assignment rows affect two records: two ordered components for alák b (borrowed Hindi adjective plus Shina light verb) and one borrowed link for cilím. The alák b light verb b- remains a separate component. Source spelling, gloss and identity are unchanged.

Primary evidence: Buddruss (1996) *Shina-Rätsel* printed pp. 40–41 (cached scans `tmp/pdfs/shina-raetsel/hi-17.png`, `hi-18.png`); complete Turner CDIAL 700 in `shina-primary002.json`; Jambu homepage candidate search `shina-search001.json`; attested Hindi donor forms in current CLDF. Held: ai 'goat' (Turner 145 or 887); arú's deeper *andaru proposal (tentative); arī́no 'inner' derived from unlinked arú and bā̀i-bā̀ri with unlinked bā̀ri (the graph validator requires linkable bases); the several source words with questioned Turner or Burushaski parallels; loans without an attested immediate donor. The first four-record draft failed the graph's linkable-parent gate; these two analyses were removed before saving, and the final decisions were checked again.

Validation pending. No rebuild.
"""
(HERE / "pass071-review.md").write_text(review)
print(len(P), "records", sum(2 if isinstance(p[1], list) else 1 for p in P), "assignment rows")
