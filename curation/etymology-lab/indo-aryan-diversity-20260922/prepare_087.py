"""Prepare second source-checked Zargari noun batch."""
import csv
import json
from pathlib import Path

from etymology_assignments import read_assignments


D = Path("curation/etymology-lab/indo-aryan-diversity-20260922")
SOURCE = "rezaibaghbidi2003zargari"
all_forms = {
    r["ID"]: r
    for r in csv.DictReader(open("cldf/forms.csv"))
    if SOURCE in r["Source"]
}
linked = {r["Form_ID"] for r in read_assignments() if r["Status"] == "accepted"}
linked.update(
    r["Child_ID"]
    for r in csv.DictReader(open("cldf/edges.csv"))
    if r["Rank"] == "1"
)
identities = {
    r["Form_ID"]: r["Source_Key"]
    for r in csv.DictReader(open("data/form-identities.csv"))
    if r["Source_Key"].startswith(SOURCE + ":")
}
source_rows = list(csv.reader(open("data/other/forms/20260825-rezai-baghbidi-zargari.csv")))
raw_rows = list(csv.DictReader(open("data/other/forms/raw_data/20260825-rezai-baghbidi-zargari-audit.csv")))

families = [
    (
        "ash", "3674", "kṣārá¹", ["f_hfilxwk7h3ljc"],
        "Turner explicitly gives southeastern European Romani čhar 'ashes' under the ch- branch of kṣārá¹, contrasting its ash sense with kh- forms mostly denoting alkali. The source prints čhār in its core noun list (p. 131 §3.1.8).",
    ),
    (
        "grass", "4471", "ghāsá", ["f_prjefw5dnqcd2"],
        "Turner lists Armenian Romani khas 'grass' and European Romani khas 'hay' under ghāsá; the source prints Zargari khās 'grass' (p. 131 §3.1.8). Its aspiration and vowel are kept as printed.",
    ),
    (
        "Gypsy man / husband", "5570", "ḍōmba", ["f_can7jp5cb63do", "f_wepl3xik5deue"],
        "Turner explicitly gives European Romani rom 'man, husband', especially 'Gypsy man', under ḍōmba. The source prints Zargari singular rom and plural roma (p. 131 §3.1.8). This lexical ancestry does not equate the modern people with the ancient caste description.",
    ),
    (
        "tree", "3120", "kāṣṭhá", ["f_g45j4gxbgepng"],
        "Turner gives European Romani kašt 'piece of wood, timber' under kāṣṭhá. The source prints Zargari qāšt 'tree' (p. 131 §3.1.8), separately illustrates kāšt/qāšt as free k/q variants before back vowels (p. 128 §2.3.16), and explicitly compares kāšt 'tree' with Sanskrit kāṣṭhá (p. 146 §5.4). The tree/wood sense difference is retained.",
    ),
    (
        "woman", "10504", "yuvatí", ["f_jxifsyxkuqpn4", "f_eu6uetyces642"],
        "Turner directly records European Romani ǰuvel/ǰuvli 'woman, girl' under yuvatí. The source prints Zargari džuvel and its irregular plural džuvla (p. 132 §3.1.9); the installed j represents its transcription profile, not a different stem.",
    ),
    (
        "egg", "1111", "āṇḍá", ["f_lsfpp7ccctkpc"],
        "Turner gives Greek Romani vanro 'egg' under āṇḍá. The source prints Zargari vānvro, including nasalized/denasalized realizations (p. 128 §2.3.10), and repeats the noun in its singular/plural list (p. 132 §3.1.9). This follows the Romani branch despite the older family's deeper origin being debated.",
    ),
]

accepted = []
audit = []
review = [
    "# Zargari: inherited ash, grass, people, tree and egg terms",
    "",
    "Eight source-checked reflex records across six historical families. Standing autonomous authorization. Source PDF printed pp. 128, 131–132, 146 (PDF pp. 6, 9–10, 24), glyph audit, installed import keys and full CDIAL articles/addenda were checked. No lexical import or rebuild.",
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
            if x["Status"] == "ingest"
            and key in x["Emitted_Keys"].split()
            and x["Final_Gloss"] == r["Gloss"]
        ]
        assert len(raw_matches) >= 1, (fid, raw_matches)
        citation = f"CDIAL[{parent} and addenda];{r['Source']}"
        accepted.append({
            "record": {k: r[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")},
            "parent": parent,
            "kind": "reflex",
            "citation": citation,
            "evidence": evidence + " Original PDF, glyph-decoded source audit and imported key checked; no transcription or identity change.",
        })
        audit.append({
            "id": fid, "source_key": key, "raw_units": [x["Unit_ID"] for x in raw_matches],
            "raw_forms": [x["Raw_Form"] for x in raw_matches],
            "final_forms": [x["Final_Forms"] for x in raw_matches],
            "printed_pages": [x["Printed_Page"] for x in raw_matches],
            "record_hashes": [x["Record_SHA256"] for x in raw_matches],
        })
        forms.append(r["Form"])
    review.append(f"| {number} | **{' / '.join(forms)}** ‘{meaning}’ ({len(ids)}) | **{headword}**, CDIAL **{parent}** | {evidence} |")

held = []
for form, reason in [
    ("murś", "Turner 9828 gives European Romani murš under manuṣya but describes crossing with puruṣa; the source instead compares māriṣa (p. 147 §5.4). The conflicting ancient analysis needs adjudication before assigning a single CDIAL parent."),
    ("masik", "The source lists this as 'month', but its relationship to Sanskrit māsa versus a later adjective or contact form has not been checked against a diagnostic Romani comparison."),
    ("jukel", "The source prints džukel 'dog' with irregular plural džukle, but the historical branch is not secure from the current CDIAL search; do not infer one from the plural alone."),
]:
    matches = [r for r in all_forms.values() if r["Form"] == form and r["ID"] not in linked]
    assert len(matches) == 1, (form, len(matches))
    held.append({"record": {k: matches[0][k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}, "reason": reason})

assert len(accepted) == 8 and len(held) == 3
(D / "pass087-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(D / "pass087-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
review += ["", "Three held records:"]
review += [f"- **{h['record']['Form']}** ‘{h['record']['Gloss']}’: {h['reason']}" for h in held]
review += ["", "Eight assignment rows affect eight records, all `reflex`. Printed plurals are ordinary inflections of their inherited stems. Discovery: homepage `zargari-search016`, full CDIAL `zargari-primary014`, primary source images and `pass087-source-audit.json`. Save and validation pending."]
(D / "pass087-review.md").write_text("\n".join(review) + "\n")
print("accepted", len(accepted), "held", len(held))
