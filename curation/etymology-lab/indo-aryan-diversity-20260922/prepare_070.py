"""Review five further SIL Batote Dogri lexical records."""
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
        "f_irur7knfuenwg", "37", "6225", "davara", "thread",
        "Turner 6225 gives Prakrit ḍōra ‘thread’, Punjabi ḍor and Nepali ḍoro, with West Pahari Kɔṭgarh ḍōr in the addenda. Batote doṛ has the same thread sense and dental/retroflex series; the source transcribes final ɽ, retained in its normalized form.",
    ),
    (
        "f_sm6j6udgj7ski", "90", "9964", "mahiṣá", "buffalo",
        "Turner 9964 gives adjacent Ḍoḍi maĩ ‘buffalo’, Churahi mhai and Kɔci maī, continuing mahiṣa. Batote mɛ has the same reduced m-vowel stem and animal sense; the source has no nasalization mark, so one is not supplied.",
    ),
    (
        "f_dy746fzp6x2gg", "119", "12115", "vḗlā", "morning",
        "Turner 12115 gives Punjabi belā ‘time’, Kashmiri vēla ‘time, hour’ and Odia beḷa ‘daytime’. Batote bɛla is the morning response beside a second morning word; it selects a particular time of day from this lexical family. Turner considers a much older Dravidian input possible, which does not identify a modern donor.",
    ),
    (
        "f_idqhzeogz7cvm", "198", "6624-2", "*dravaḍati", "run!",
        "Turner 6624 explicitly gives a -ḍ- extension with nearby Ḍoḍi dauṛṇō, Lahnda dɔṛaṇ and West Pahari Kɔṭgarh dɔṛnõ ‘run’. Jambu's 6624-2 is that extension node, and Batote imperative dɔṛ retains the diagnostic retroflex; it is ordinary inflection, not a new derivation.",
    ),
    (
        "f_edh3ajuieks2q", "198", "6624-2", "*dravaḍati", "run!",
        "The source gives dɔṛiyɛ as a second response to the same ‘run!’ prompt. Turner 6624's regional dauṛ-/dɔṛ- comparanda support the same -ḍ- extension as the co-occurring dɔṛ; the -iyɛ imperative ending is retained in the record and is not treated as a separate derivation.",
    ),
]

accepted = []
audit = []
lines = [
    "# SIL Batote Dogri: thread, buffalo, time, and running",
    "",
    "Five reflex records across four etymological families. Standing autonomous authorization. Appendix B printed pp. 26–28 was visually checked; the frozen official-font snapshot and installed source rows match the selected forms. Full Turner articles and addenda were read.",
    "",
    "| # | Dogri form and meaning | Proposed etymology | Evidence |",
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

assert len(accepted) == 5
(BASE / "pass070-decisions.json").write_text(json.dumps({"accepted": accepted, "held": []}, ensure_ascii=False, indent=2) + "\n")
(BASE / "pass070-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
lines += [
    "",
    "Five assignment rows affect five records. All are reflexes; the two ‘run!’ forms are alternative source responses and ordinary inflected forms of one stem. The prior five pass064 holds remain unresolved and are excluded.",
    "",
    "Discovery: existing Jambu homepage query implementation (dogri-search002); complete primary prose in dogri-primary004. No rebuild.",
    "",
    "Save and validation pending.",
]
(BASE / "pass070-review.md").write_text("\n".join(lines) + "\n")
print("accepted", len(accepted))
