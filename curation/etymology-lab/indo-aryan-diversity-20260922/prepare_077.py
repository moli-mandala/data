"""Review four verbal responses in the SIL Batote Dogri survey."""
import csv
import json
from pathlib import Path

from etymology_assignments import read_assignments


BASE = Path("curation/etymology-lab/indo-aryan-diversity-20260922")
SOURCE = "brightbill-turner2007dogri"
linked = {r["Form_ID"] for r in read_assignments() if r["Status"] == "accepted"}
linked.update(r["Child_ID"] for r in csv.DictReader(open("cldf/edges.csv")) if r["Rank"] == "1")
remaining = {
    r["ID"]: r
    for r in csv.DictReader(open("cldf/forms.csv"))
    if SOURCE in r["Source"] and r["ID"] not in linked
}
snapshot = {
    r["Item"]: r
    for r in csv.DictReader(open("data/other/forms/raw_data/sil_dogri_2007/wordlist_snapshot.tsv"), delimiter="\t")
}

selected = [
    (
        "f_gabdwan3nu66y", "184", "8209", "píbati", "he is drinking",
        "Turner 8209 explicitly gives Punjabi pīṇā and several West Pahari pī- forms 'drink'. Batote pīya da is an analytic finite response with that same pī- lexical verb; da is retained as part of the source phrase rather than assigned separately.",
    ),
    (
        "f_2n5zedl4hn7ua", "185", "13479", "suptá", "he is sleeping",
        "Turner 13479 gives Pali/Prakrit sutta- 'asleep', nearby Ḍoḍi suttōrō 'asleep', and West Pahari Jaunsari sūtṇõ 'sleep'. Batote suttaḍa contains the same geminate t sleep stem in a finite response; the verbal ending does not require a separate etymon.",
    ),
    (
        "f_yvgu5jqrmgaco", "188", "12598", "śr̥ṇṓti", "he is listening",
        "Turner 12598 gives Ḍoḍi śuṇṇō, Punjabi sunanā, and West Pahari śuṇṇū 'hear', along with related Kotgarhi addenda. Batote sūnada 'is listening' continues this regional hearing stem with an ordinary finite ending; its plain n transcription is preserved.",
    ),
    (
        "f_r3r2sxi37cday", "193", "6507-2", "*dēkṣati", "he saw",
        "Turner 6507 section 2 gives Punjabi dekhṇā, West Pahari Chamba dikkhṇā and Kotgarhi dékhṇõ 'see'; it notes regional transitive forms in i. Batote dikʰiya 'he saw' matches that northern dikh-/dekh- stem as an inflected response, not the separate dr̥śyátē branch.",
    ),
]

accepted = []
audit = []
lines = [
    "# SIL Batote Dogri: four verbal responses",
    "",
    "Four source-backed reflex records. Standing autonomous authorization. Appendix B printed p. 28 was visually checked; the frozen official-font snapshot and installed source rows match the forms. Full Turner articles and addenda were read.",
    "",
    "| # | Dogri form and survey prompt | Proposed etymology | Evidence |",
    "|---|---|---|---|",
]
for n, (fid, item, parent, head, meaning, evidence) in enumerate(selected, 1):
    r = remaining[fid]
    s = snapshot[item]
    assert s["Status"] == "response" and s["Gloss"] == meaning
    assert r["Gloss"] == meaning
    assert r["Original"].strip() in [x.strip() for x in s["Form"].split(",")], (fid, r["Original"], s["Form"])
    assert f"item {item}, Batote" in r["Source"]
    page = s["Printed_Page"]
    citation = f"{SOURCE}[Appendix B, printed p. {page}, item {item}, Batote];CDIAL[{parent.split('-')[0]} and addenda]"
    accepted.append(dict(
        record={k: r[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")},
        parent=parent,
        kind="reflex",
        citation=citation,
        evidence=evidence + " Source image and reviewed import row checked; no transcription or identity changes.",
    ))
    audit.append(dict(id=fid, item=item, page=page, snapshot_form=s["Form"], compiled_form=r["Form"], original=r["Original"], snapshot_gloss=s["Gloss"], source=r["Source"], image=f"../tmp/pdfs/dogri/rendered/dogri-{page}.png"))
    lines.append(f"| {n} | **{r['Form']}** ‘{meaning}’ | **{head}**, CDIAL **{parent}** | {evidence} |")

assert len(accepted) == 4
(BASE / "pass077-decisions.json").write_text(json.dumps({"accepted": accepted, "held": []}, ensure_ascii=False, indent=2) + "\n")
(BASE / "pass077-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
lines += [
    "",
    "Four assignment rows affect four survey records. The source gives complete inflected answers; these links concern the lexical verb stems and preserve the exact response forms. Jakt ‘child’ remains held because the precise Iranian-versus-Indic history of the jātak family is disputed. The prior pass064 holds remain excluded.",
    "",
    "Discovery: existing Jambu homepage search (dogri-search004); primary prose in dogri-primary007. No rebuild.",
    "",
    "Save and validation pending.",
]
(BASE / "pass077-review.md").write_text("\n".join(lines) + "\n")
print("accepted", len(accepted))
