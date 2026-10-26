"""Prepare image-checked Andersen glossary ancestry decisions (pass 065)."""

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# form, target, relation, displayed etymon, evidence
PROPOSALS = [
    ("adʰi", "249", "reflex", "ádhi, CDIAL 249", "Andersen's adhi entry compares RV ádhi and Pali adhi; Turner 249 lists those forms with the sense 'up, above, on'. The attested preposition is the inherited item, not the distinct derivative ádhika."),
    ("amta", "347", "reflex", "ánta, CDIAL 347", "Andersen calls aṁta plural 'borderers; neighbors' and compares RV ánta, Pali anta. Turner 347 has the boundary/proximity noun and Prakrit aṁta; the human plural reflects a straightforward boundary-dweller sense, though Turner does not list this precise Ashokan use."),
    ("amtala", "357", "reflex", "ántara¹, CDIAL 357", "Andersen lists aṁtala 'time interval' beside RV ántara and Pali antara, separating -r- Gu from -l- Bh/Ah/Sa. Turner 357 explicitly gives Ashokan kāl./dh./jau. aṁtala and 'long period' aṁtara under its first ántara section."),
    ("eva", "2524", "reflex", "ēvá¹, CDIAL 2524", "Andersen compares RV ēvá and Pali eva/va for 'so, just, even'. Turner 2524 explicitly lists Ashokan eva, yeva and man. vaṁ as continuations of the emphatic particle."),
    ("na", "6906", "reflex", "ná, CDIAL 6906", "The printed glossary gives RV ná, Pali na for the negative particle. Turner 6906 explicitly lists Ashokan na, with kāl. nā, for this exact function."),
    ("pada", "7747", "reflex", "padá¹, CDIAL 7747", "Andersen gives RV padá and Pali pada for 'quarter; foot, path, place'. Turner 7747 groups the footstep, track and place senses with Pali pada; this is the first padá branch, not *pada 'tendon'."),
    ("pamnā", "7682", "reflex", "pañcāśát, CDIAL 7682", "Andersen prints paṁnā 'fifty' and compares RV pañcāśát with Pali paññāsa/paṇṇāsa. Turner 7682 gives the same Pali assimilated numeral and the Prakrit paṇṇā forms."),
    ("pavata", "7945", "reflex", "párvata¹, CDIAL 7945", "Andersen prints pavata 'mountain, rock' beside RV párvata and Pali pabbata. Turner 7945 explicitly cites Ashokan pavatasi and the rock/mountain sense; 7945-2 is a later -ll- extension and 7946 is a fish."),
    ("pitu", "8179", "reflex", "pitŕ̥, CDIAL 8179", "Andersen's pitu 'father' compares RV pitr̥ and Pali pitar, noting the Brahmagiri piti- variant. Turner 8179 lists pitúḥ in the Old Indo-Aryan paradigm and Ashokan pituna, the same stem's inflection."),
    ("rāja", "10679", "reflex", "rā́jan, CDIAL 10679", "The glossary has rāja 'king' with RV rā́jan and Pali rāja; the printed -ā citation is nominative. Turner 10679 explicitly lists Ashokan gir. rājā/rāño and other Ashokan regional forms under rā́jan."),
    ("vaṁṇa", "11338", "reflex", "várṇa¹, CDIAL 11338", "Andersen prints vaṁṇa 'colour' and compares RV várṇa, Pali vaṇṇa. Turner 11338 is the appearance/colour/class branch and has Pali vaṇṇa; 11339 is the separate 'sound, syllable' branch."),
    ("vasa", "11392", "reflex", "varṣá, CDIAL 11392", "Andersen distinguishes vasa 'year' from vas 'live', gives RV varṣá/Pali vassa, and says this Ashokan use is plural. Turner 11392 section 2 specifically lists Ashokan kāl./dh./jau. vasa 'year'; the rain branch is not the intended sense."),
    ("hevam", "2527", "reflex", "ēvám, CDIAL 2527", "Andersen gives hevam 'so, thus' beside RV ēvám and Pali evaṁ. Turner 2527 explicitly includes Ashokan evaṁ and he- forms; this is the adverb, not emphatic ēvá 2524."),
    ("sāvaṇa", "f_dgnyetyrlbn2m", "variant", "source variant of sāvana", "Andersen's separate sāvaṇa line on printed p. 174 explicitly says 'see sāvana-'. The target is the already saved source head for 'announcement, edict' (CDIAL 12698), not the month-name śrāvaṇa 12699. The source record's blank gloss is preserved."),
]

rows = json.loads((HERE / "andersen-remaining043.json").read_text())
by_form = {r["Form"]: r for r in rows}
sidecar = ROOT / "data/other/forms/etymologies/20260804-andersen-mre.csv"
existing = {r["Form_ID"] for r in csv.DictReader(sidecar.open()) if r["Status"] == "accepted"}
source_rows = list(csv.reader((ROOT / "data/other/forms/20260804-andersen-mre.csv").open()))
source_keys = {(r[2], r[3], r[7]) for r in source_rows}
compiled_ids = set()
for r in csv.DictReader((ROOT / "cldf/forms.csv").open()):
    if r["Source"].startswith("andersen1990["):
        compiled_ids.add(r["ID"])

decisions = {"accepted": [], "held": []}
review_rows = []
for i, (form, parent, kind, label, evidence) in enumerate(PROPOSALS, 1):
    r = by_form[form]
    assert r["ID"] not in existing and (r["Original"], r["Gloss"], r["Source"]) in source_keys and r["ID"] in compiled_ids, form
    record = {k: r[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}
    citation = record["Source"]
    if kind != "variant":
        citation += f";CDIAL[{parent}]"
    decisions["accepted"].append({"record": record, "parent": parent, "kind": kind, "citation": citation, "evidence": evidence})
    review_rows.append(f"| {i} | **{form}** ‘{record['Gloss']}’ | **{label}** ({kind}) | {evidence} |")

(HERE / "pass065-decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2) + "\n")
review = """# Andersen 1990: source-checked glossary heads

Fourteen accepted source-defined records under standing autonomous authorization. The source PDF page number in every record is two less than its printed page number; all listed heads were checked against the scanned page, and the CDIAL article and correct subsection were read in full.

| # | Ashokan Prakrit | Proposed etymology | Evidence |
|---|---|---|---|
""" + "\n".join(review_rows) + "\n\n" + """Fourteen assignment rows affect fourteen records: thirteen inherited reflex links and one explicit source-variant link. No source spelling, locality tags, identity records, or compiled graph rows were edited. Held from this review: **sata** ‘hundred’ because its cached OCR merges the following saṁt/sapaṁnā block and source-boundary metadata; **adʰa** ‘half’ because the stored dental form differs from the printed aḍha; **su** prefix because no sufficiently specific CDIAL entry was established; **as** ‘to be’ because the available CDIAL 977 is a finite ásti entry rather than the glossary's broad root. These require separate review, not guessed links.

Primary evidence: Andersen (1990) scan PDF pp. 135, 143, 153, 156–157, 165, 167, 172, 176 (printed pp. 137, 145, 155, 158–159, 167, 169, 174, 178). Complete Turner articles cached in `andersen-primary006.json` and `andersen-primary007.json`. Candidate discovery used the homepage search implementation, `andersen-search008.json`.

Validation pending until check/save/source gates complete. No rebuild.
"""
(HERE / "pass065-review.md").write_text(review)
print(len(decisions["accepted"]), "accepted", len(existing), "existing Andersen accepted")
