"""Review the remaining SIL Batote Dogri wordlist against CDIAL."""
import csv
import json
from pathlib import Path

from etymology_assignments import read_assignments


BASE = Path("curation/etymology-lab/indo-aryan-diversity-20260922")
SOURCE = "brightbill-turner2007dogri"

linked = {r["Form_ID"] for r in read_assignments() if r["Status"] == "accepted"}
linked.update(
    r["Child_ID"]
    for r in csv.DictReader(open("cldf/edges.csv"))
    if r["Rank"] == "1"
)
remaining = {
    r["ID"]: r
    for r in csv.DictReader(open("cldf/forms.csv"))
    if SOURCE in r["Source"] and r["ID"] not in linked
}
snapshot = {
    r["Item"]: r
    for r in csv.DictReader(
        open("data/other/forms/raw_data/sil_dogri_2007/wordlist_snapshot.tsv"),
        delimiter="\t",
    )
}

# Each form and its exact printed item were checked against rendered pp. 26–28.
# The cited Turner entries were read in full, including their addenda.
selected = [
    (
        "f_iws7t4xxchn4y", "13", "9229", "bāhú", "arm (whole)",
        "Turner 9229 gives Punjabi bā̃h and West Pahari Kɔci bā́ ‘arm’; Batote bã has the same nasalized vowel and body-part sense. The shorter final form is supported by Kɔci bā́, so no extra suffix is posited.",
    ),
    (
        "f_bknht7chooihw", "14", "603", "aratní", "elbow",
        "Turner 603 gives Lahnda and Punjabi arak ‘elbow’ plus West Pahari ɔṛkəṇ/ɔṛkṇi in the addenda. Batote aṛkī shares the distinctive r-k sequence and elbow sense; its final ī is retained in the citation rather than silently normalized to arak.",
    ),
    (
        "f_eojcsdrx3qsye", "128", "8283", "purāṇá", "old (things)",
        "Turner 8283 lists Punjabi purānā and West Pahari Kɔci puraṇo specifically for old things and plants, beside Kɔṭgarh pəraṇɔ. Batote parāṇa has the same lexical sense and regional p-r-ṇ shape.",
    ),
    (
        "f_cawezs5dleuvi", "129", "6983", "náva", "new",
        "Turner 6983 explicitly lists Ḍomaki nΛmá, Torwali nam, and Gawri nam-like forms beside West Pahari and Punjabi nava. Batote nama matches this nasalized regional outcome and the exact ‘new’ sense; no derivational suffix is inferred.",
    ),
    (
        "f_plmf5tr25zlde", "147", "5731", "tala", "below",
        "Turner 5731 explicitly gives Lahnda talē, West Pahari Curahi taḷē and Hindi tale as postpositions ‘below’. Batote tʰalɛ is the same positional word with source aspiration retained; select section 1 tala, not section 2 *talla.",
    ),
    (
        "f_5xasrmksb7xyy", "161", "2485", "ḗkādaśa", "eleven",
        "Turner 2485 gives Punjabi giārā̃, Khetrani žārhā and West Pahari Curahi yāhrā, all ‘eleven’. Batote jara continues this numeral family with the affricated initial and reduced middle syllable; its printed final vowel is preserved.",
    ),
    (
        "f_nvmt7jtovcl6e", "202", "992", "ahám", "I (1 sg.)",
        "Turner 992 explicitly gives Poguli and Ḍoḍi aũ and West Pahari aũ, matching Batote aũ in form and first-person singular meaning. The exact pronominal subsection is ahám, not vayám ‘we’.",
    ),
]

accepted = []
audit = []
lines = [
    "# SIL Batote Dogri: seven further inherited links",
    "",
    "Seven single-record reflexes from Appendix B, printed pp. 26–28. Standing autonomous authorization. The source PDF images and the official-font snapshot were checked against the installed records; all shortlisted CDIAL articles and addenda were read in full.",
    "",
    "| # | Dogri form and meaning | Proposed etymology | Evidence |",
    "|---|---|---|---|",
]
for n, (fid, item, parent, head, meaning, evidence) in enumerate(selected, 1):
    r = remaining[fid]
    s = snapshot[item]
    assert s["Status"] == "response"
    assert s["Gloss"] == meaning, (fid, s["Gloss"], meaning)
    assert r["Gloss"] == meaning, (fid, r["Gloss"], meaning)
    assert r["Original"].strip() in [x.strip() for x in s["Form"].split(",")], (fid, r["Original"], s["Form"])
    assert f"item {item}, Batote" in r["Source"]
    page = s["Printed_Page"]
    citation = f"{SOURCE}[Appendix B, printed p. {page}, item {item}, Batote];CDIAL[{parent} and addenda]"
    accepted.append(
        dict(
            record={k: r[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")},
            parent=parent,
            kind="reflex",
            citation=citation,
            evidence=evidence + " Source image and reviewed import row checked; no transcription or identity changes.",
        )
    )
    audit.append(dict(id=fid, item=item, page=page, snapshot_form=s["Form"], compiled_form=r["Form"], original=r["Original"], snapshot_gloss=s["Gloss"], source=r["Source"], image=f"../tmp/pdfs/dogri/rendered/dogri-{page}.png"))
    lines.append(f"| {n} | **{r['Form']}** ‘{meaning}’ | **{head}**, CDIAL **{parent}** | {evidence} |")

held = []
for form, reason in [
    ("dumb", "Turner 6419 explicitly warns that most *dumbha ‘tail’ forms may be Iranian loans. The immediate donor and route for this Dogri form are not established, so an inherited reflex link would overstate the claim."),
    ("baḍḍa", "One compiled record combines the source's ‘morning’ and ‘big’ responses. Turner 11225 clearly supports big, but assigning the whole mixed-sense record would also cover morning without resolving it."),
    ("manukʰ", "Turner 9828 and 10049 have competing manuṣya/mānuṣa continuations and extensions; the -kʰ outcome needs a closer Dogri comparison before branch choice."),
    ("go", "Turner divides cow/ox gō and gau forms among several branches; the Dogri monosyllable does not choose among 4093, 4147 and 4255."),
    ("suraj", "The Dogri sun form contains final -j, while Turner 13574 sections distinguish sūra, sūrya and sūriya; no exact local derivation has been verified."),
]:
    matches = [r for r in remaining.values() if r["Form"] == form]
    assert len(matches) == 1, (form, len(matches))
    held.append(dict(record={k: matches[0][k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}, reason=reason))

assert len(accepted) == 7 and len(held) == 5
(BASE / "pass064-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(BASE / "pass064-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
lines += ["", "Five candidate records held:"]
lines += [f"- **{h['record']['Form']}** ‘{h['record']['Gloss']}’: {h['reason']}" for h in held]
lines += ["", "Discovery: existing Jambu homepage query implementation (dogri-search001); complete primary entries in dogri-primary001–003. Seven assignment rows affect seven records. No borrowed, derived or component links in this pass. No rebuild.", "", "Save and validation pending."]
(BASE / "pass064-review.md").write_text("\n".join(lines) + "\n")
print("accepted", len(accepted), "held", len(held))
