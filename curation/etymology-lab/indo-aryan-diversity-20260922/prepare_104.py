"""Prepare directly supported Zargari pronoun, time and place reflexes."""
import csv
import json
from pathlib import Path

from etymology_assignments import read_assignments


D = Path("curation/etymology-lab/indo-aryan-diversity-20260922")
SOURCE = "rezaibaghbidi2003zargari"
all_forms = {
    r["ID"]: r for r in csv.DictReader(open("cldf/forms.csv"))
    if SOURCE in r["Source"]
}
linked = {r["Form_ID"] for r in read_assignments() if r["Status"] == "accepted"}
linked.update(r["Child_ID"] for r in csv.DictReader(open("cldf/edges.csv")) if r["Rank"] == "1")
identities = {
    r["Form_ID"]: r["Source_Key"]
    for r in csv.DictReader(open("data/form-identities.csv"))
    if r["Source_Key"].startswith(SOURCE + ":")
}
source_rows = list(csv.reader(open("data/other/forms/20260825-rezai-baghbidi-zargari.csv")))
raw_rows = list(csv.DictReader(open("data/other/forms/raw_data/20260825-rezai-baghbidi-zargari-audit.csv")))

families = [
    (
        "interrogative and indefinite 'who'", "2575", "kaḥ punar",
        ["f_kfpuj3smdqne6", "f_kwhwvjge7mcze", "f_nqzpv5xgmlb6c", "f_e2bvuvl7lkqwi", "f_4oqbkgbebpmba", "f_itgr2xhgf5ijc"],
        "Turner 2575 explicitly cites European Romani kon 'who' and, in its addendum, an oblique kɔs in a Pahari paradigm. The source prints Zargari kon/qon (nominative), kos/qos (oblique), kosku (dative/genitive), kostār (ablative/instrumental), and kostu (locative) together in §3.3.5.1, plus indefinite kon/qon 'somebody' in §3.3.6.1 (p. 136). The case endings and indefinite use are source-level morphology of the same inherited pronoun; no distinct etymon is inferred for each case form.",
    ),
    (
        "yesterday", "14108-2", "*hīyas",
        ["f_4vuksmaeykdhm"],
        "Turner 14108 §2 expressly lists Romani Italian yidž and Hungarian īž/ič as 'yesterday' under *hīyas. The source prints ij/iž as the Zargari word for 'yesterday' (§2.3.21 and §3.20.1.2, pp. 128, 143); its pronunciation discussion accounts for the imported glyph variant. The section-2 etymon is the precise parent.",
    ),
    (
        "bottom / below / under", "5731", "tala",
        ["f_xp5wgayzk7zlu", "f_z4by54xm7kptw", "f_mnfz27uvtuqik"],
        "Turner 5731 §1 gives European Romani tel, tele 'under' and other Indo-Aryan bottom/below forms under tala. The source prints Zargari teli as 'bottom' (§3.1.15, p. 132), 'below' (§3.20.1.3, p. 143), and 'beneath; under' (§3.20.2.1, p. 144), supporting all three lexical/syntactic attestations.",
    ),
    (
        "no", "6906", "ná",
        ["f_ohr2inandwid6"],
        "Turner 6906 includes European Romani na 'not' and explicitly notes 'no' in its addendum; the source prints Zargari nā 'no' among negative adverbs in §3.20.1.7 (p. 144). The long vowel is retained from the source transcription, not inferred from the Turner Romani form.",
    ),
]

accepted = []
audit = []
review = [
    "# Zargari: interrogative, yesterday, below and no",
    "",
    "Eleven source-checked reflex records across four historical families. Standing autonomous authorization. Original PDF printed pp. 128, 132, 136, 143–144, glyph audit and imported keys, homepage searches and full CDIAL articles/addenda were checked. No lexical import or rebuild.",
    "",
    "| # | Zargari forms and meaning | Proposed etymology | Evidence |",
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
            x for x in source_rows
            if x[0] == "Zarg" and x[3] == r["Gloss"]
            and x[7] == r["Source"] and x[10] == key
        ]
        assert len(source_matches) == 1, (fid, source_matches)
        raw_matches = [
            x for x in raw_rows
            if x["Status"] == "ingest" and key in x["Emitted_Keys"].split()
            and x["Final_Gloss"] == r["Gloss"]
        ]
        assert len(raw_matches) == 1, (fid, raw_matches)
        citation = f"CDIAL[{parent} and addenda];{r['Source']}"
        accepted.append({
            "record": {k: r[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")},
            "parent": parent,
            "kind": "reflex",
            "citation": citation,
            "evidence": evidence + " Original PDF, glyph-decoded source audit and imported key checked; no transcription or identity change.",
        })
        audit.append({
            "id": fid, "source_key": key, "raw_unit": raw_matches[0]["Unit_ID"],
            "raw_form": raw_matches[0]["Raw_Form"],
            "final_form": raw_matches[0]["Final_Forms"],
            "printed_page": raw_matches[0]["Printed_Page"],
            "record_hash": raw_matches[0]["Record_SHA256"],
        })
        forms.append(r["Form"])
    review.append(f"| {number} | **{' / '.join(forms)}** ‘{meaning}’ ({len(ids)}) | **{headword}**, CDIAL **{parent}** | {evidence} |")

held = []
for form, reason in [
    ("opro", "The source prints 'up; above' and 'above' (pp. 127, 143–144), but Turner 2218 upári points to a separate *uppari entry and does not itself list Romani opro. The precise subentry needs checking before assigning either of these records."),
    ("ānvro", "The source prints 'inside' and 'in; inside' (pp. 143–144). Turner 357 ántara cites Romani andre 'in, within', but the Zargari nv and final -o versus that comparator have not been established from the source's sound rules; hold both records."),
]:
    matches = [r for r in all_forms.values() if r["Form"] == form and r["ID"] not in linked]
    assert matches, form
    for r in matches:
        held.append({"record": {k: r[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}, "reason": reason})

assert len(accepted) == 11 and len(held) == 4
(D / "pass104-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(D / "pass104-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
review += ["", "Four held records:"]
review += [f"- **{h['record']['Form']}** ‘{h['record']['Gloss']}’: {h['reason']}" for h in held]
review += ["", "Eleven assignment rows affect eleven records, all `reflex`. The interrogative case endings remain source-level morphology. Discovery: homepage `zargari-search021`, full CDIAL `zargari-primary019`, PDF images and `pass104-source-audit.json`. Save and validation pending."]
(D / "pass104-review.md").write_text("\n".join(review) + "\n")
print("accepted", len(accepted), "held", len(held))
