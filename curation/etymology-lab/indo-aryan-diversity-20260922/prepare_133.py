"""Prepare direct Turner-backed Zargari above, scissors, grape, month and ours."""
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

families = [
    (
        "above", "2333", "*uppari", ["f_jhhcfgsnqoom4", "f_lqnkry4geo55a"],
        "Turner 2333 directly cites European Romani opre 'on, over' under *uppari. The source prints Zargari opro [ophro] 'up; above' in §2.3.3 (p. 127), opro 'above' as an adverb (§3.20.1.3, p. 143) and as an adposition (§3.20.2.1, p. 144). The source's optional aspiration is preserved; the two installed opro records cover these attestations.",
    ),
    (
        "from above", "2333-2", "*upparil-", ["f_dnz4vptwfb4es"],
        "Turner 2333 explicitly cites European Romani opral 'from above', represented by its compiled inferred -l- extension 2333-2. The source prints Zargari opr-āl 'from above' and identifies -āl as the local ablative place suffix (p. 132 §3.1.15). This is directly parallel to Turner's Romani form, while the source's -āl spelling is retained.",
    ),
    (
        "scissors", "3069", "*kārti-", ["f_7iodahagd42ek"],
        "Turner 14382 explicitly derives European Romani kat 'shears, scissors' from *kārti- rather than karttrī-; 14382 redirects to the substantive *kārti- article 3069. The source prints Zargari kāt/qāt 'scissors' when describing /k/~/q/ free variation before back vowels (§2.3.16, p. 128); the installed kāt is the attested form.",
    ),
    (
        "grape", "6628", "drā́kṣā-", ["f_ibhr7nc2lg6ta"],
        "Turner 6628 lists European Romani drak(h) 'grape' and Persian Romani dérak under drā́kṣā-. The source prints Zargari de.rāk 'grape' with explicit Early Romani drakh comparison in its syllable-structure examples (§2.4.1, p. 129). The source's initial vowel and local syllabification are retained.",
    ),
    (
        "month", "10104", "mā́sa", ["f_g77xntn5szdym"],
        "Turner 10104 explicitly cites European Romani másek 'month' under mā́sa. The source prints Zargari masik 'month' as a consonant-final noun in §3.1.8 (p. 131); its i/e vowel difference remains source-specific.",
    ),
    (
        "ours", "988-2", "*asmākara-", ["f_wgrmxoceucf5y", "f_rmxozlhq3iw3i"],
        "Turner 988 lists European Romani amaro 'ours' under an -ra-/-kāra- extension of asmā́ka, represented by compiled node 988-2. The source prints Zargari āmāros 'ours' (oblique) and āmārostār 'from ours; with ours' together in its possessive paradigm (p. 135 §3.3.3.1). Source case suffixes are retained; both forms have the same inherited possessive stem.",
    ),
]

accepted = []
audit = []
review = [
    "# Zargari: above, scissors, grape, month and ours",
    "",
    "Eight source-checked reflex records across five lexical families, with the local 'above' ablative assigned its precise Turner extension. Standing autonomous authorization. Original PDF printed pp. 127–129, 131–132, 135, 143–144, glyph audit and imported keys, homepage searches and full CDIAL articles/addenda were checked. No lexical import or rebuild.",
    "",
    "| # | Zargari forms and meanings | Proposed etymology | Evidence |",
    "|---|---|---|---|",
]
for number, (meaning, parent, headword, ids, evidence) in enumerate(families, 1):
    forms = []
    for fid in ids:
        r = all_forms[fid]
        assert fid not in linked, fid
        assert r["Language_ID"] == "Zarg", (fid, r["Language_ID"])
        key = identities[fid]
        source_matches = [x for x in source_rows if x[0] == "Zarg" and x[3] == r["Gloss"] and x[7] == r["Source"] and x[10] == key]
        assert len(source_matches) == 1, (fid, source_matches)
        raw_matches = [x for x in raw_rows if x["Status"] == "ingest" and key in x["Emitted_Keys"].split() and x["Final_Gloss"] == r["Gloss"]]
        assert len(raw_matches) == 1, (fid, raw_matches)
        root_citation = parent.split("-")[0]
        citation = (f"CDIAL[3069 and addenda; 14382 cross-reference];{r['Source']}"
                    if parent == "3069" else f"CDIAL[{root_citation} and addenda];{r['Source']}")
        accepted.append({
            "record": {k: r[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")},
            "parent": parent, "kind": "reflex",
            "citation": citation,
            "evidence": evidence + " Original PDF, glyph-decoded source audit and imported key checked; no transcription or identity change.",
        })
        audit.append({
            "id": fid, "source_key": key, "raw_unit": raw_matches[0]["Unit_ID"],
            "raw_form": raw_matches[0]["Raw_Form"], "final_form": raw_matches[0]["Final_Forms"],
            "printed_page": raw_matches[0]["Printed_Page"], "record_hash": raw_matches[0]["Record_SHA256"],
        })
        forms.append(r["Form"])
    review.append(f"| {number} | **{' / '.join(forms)}** ‘{meaning}’ ({len(ids)}) | **{headword}**, CDIAL **{parent}** | {evidence} |")

held = []
for form, reason in [
    ("bāśu", "The source gives 'near' as adverb and adposition (pp. 143–144). A Romani paśe comparison exists outside Turner, but the specific CDIAL entry and b/p, ā/u correspondences are not established."),
    ("pālo", "The source gives 'back' and 'behind' (pp. 143–144); related Romani pala/palal exists in other sources, but an exact Turner etymon and source-level semantic path have not been established."),
]:
    matches = [r for r in all_forms.values() if r["Form"] == form and r["ID"] not in linked]
    assert matches, form
    for r in matches:
        held.append({"record": {k: r[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}, "reason": reason})

assert len(accepted) == 8 and len(held) == 4
(D / "pass133-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(D / "pass133-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
review += ["", "Four held records:"]
review += [f"- **{h['record']['Form']}** ‘{h['record']['Gloss']}’: {h['reason']}" for h in held]
review += ["", "Eight assignment rows affect eight records, all `reflex`. Discovery: homepage `zargari-search031`, full CDIAL `zargari-primary028`, PDF images and `pass133-source-audit.json`. Save and validation pending."]
(D / "pass133-review.md").write_text("\n".join(review) + "\n")
print("accepted", len(accepted), "held", len(held))
