"""Review three further Zargari present/perfect verb families."""
import csv
import json
from pathlib import Path

from etymology_assignments import read_assignments


BASE = Path("curation/etymology-lab/indo-aryan-diversity-20260922")
SOURCE = "rezaibaghbidi2003zargari"
linked = {r["Form_ID"] for r in read_assignments() if r["Status"] == "accepted"}
linked.update(r["Child_ID"] for r in csv.DictReader(open("cldf/edges.csv")) if r["Rank"] == "1")
remaining = {
    r["ID"]: r
    for r in csv.DictReader(open("cldf/forms.csv"))
    if SOURCE in r["Source"] and r["ID"] not in linked
}
source_rows = list(csv.reader(open("data/other/forms/20260825-rezai-baghbidi-zargari.csv")))
raw_rows = list(csv.DictReader(open("data/other/forms/raw_data/20260825-rezai-baghbidi-zargari-audit.csv")))
identity_keys = {
    r["Form_ID"]: r["Source_Key"]
    for r in csv.DictReader(open("data/form-identities.csv"))
    if r["Source_Key"].startswith(SOURCE + ":")
}

families = [
    (
        "sell", "11640", "\\*vikriṇāti / víkrīṇītē",
        ["f_hoi2krgubr6lo", "f_cwid6ifr7y5d2"],
        "Turner 11640 explicitly lists European Romani bikin- ‘sell’ and Middle Indo-Aryan vikkiṇāti. The source gives biken- as the present stem and biken-d- as its perfect stem on p.138 §3.6.2; -d is the source's productive perfect marker, not a separate historical root.",
    ),
    (
        "steal", "4933", "cōráyati",
        ["f_qlssvb3wbvsvk", "f_tkvwfqu4feakw"],
        "Turner 4933 directly gives European Romani čor- ‘to steal’ under cōráyati. The source prints čor-/čor-d- on p.138 §3.6.2; installed cor/cord follow the Zargari transcription profile, and -d marks the local perfect stem.",
    ),
    (
        "want", "10074", "mā́rgati / mr̥gyáti",
        ["f_sbt6blkkfbfgo", "f_s2rkja5ozykgy"],
        "Turner 10074 gives European and Armenian Romani mang- ‘ask for’, and Palestinian Romani mángăr ‘wants’. The addendum questions the Sanskritized headword mā́rgati in favour of older mr̥gyáti; keep that uncertainty inside the CDIAL family. The source prints māng-/māng-l- as present/perfect stems on p.138 §3.6.2, with productive -l.",
    ),
]

accepted = []
audit = []
lines = [
    "# Zargari: sell, steal, and want",
    "",
    "Six reflex records across three verb families. Standing autonomous authorization. Rezaibaghbidi (2003) printed p.138 §3.6.2 was checked against PDF page 16 and the existing decoded primary text. Full Turner articles and addenda were read.",
    "",
    "| # | Zargari forms and meaning | Proposed etymology | Evidence |",
    "|---|---|---|---|",
]
for n, (meaning, parent, head, ids, evidence) in enumerate(families, 1):
    assert len(ids) == 2
    forms = []
    for fid in ids:
        r = remaining[fid]
        assert r["Gloss"] == "to " + meaning
        assert "p. 138, section 3.6.2" in r["Source"]
        key = identity_keys[fid]
        matches = [
            x for x in source_rows
            if x[0] == "Zarg" and x[3] == r["Gloss"] and x[7] == r["Source"] and x[10] == key
        ]
        assert len(matches) == 1, (fid, matches)
        raw = [
            x for x in raw_rows
            if x["Status"] == "ingest" and key in x["Emitted_Keys"].split()
            and x["Final_Gloss"] == r["Gloss"] and x["Source"] == r["Source"]
        ]
        assert len(raw) == 1, (fid, raw)
        citation = f"{SOURCE}[p. 138, section 3.6.2];CDIAL[{parent} and addenda]"
        accepted.append(dict(
            record={k: r[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")},
            parent=parent,
            kind="reflex",
            citation=citation,
            evidence=evidence + " Source image, glyph-decoded audit and reviewed import row checked; no transcription or identity changes.",
        ))
        audit.append(dict(id=fid, source_unit=raw[0]["Unit_ID"], raw_form=raw[0]["Raw_Form"], final_form=raw[0]["Final_Forms"], import_key=matches[0][10], page=138, image="../tmp/pdfs/zargari/etymology/page-16.png"))
        forms.append(r["Form"])
    lines.append(f"| {n} | **{' / '.join(forms)}** ‘to {meaning}’ (2) | **{head}**, CDIAL **{parent}** | {evidence} |")

held = []
for forms, reason in [
    ("pāng / pāngl", "Turner 9361 section 2 explicitly gives Romani phag- ‘break’, but the source has pāng- with a nasal and no aspiration. This may still be related, but the precise Zargari nasal/aspiration development has not been verified."),
    ("nāngāv / nāngāvd", "The source defines these as a present/perfect pair ‘hit, strike’, but no exact Turner Romani comparandum or diagnostic historical root has been established."),
]:
    for form in forms.split(" / "):
        ms = [r for r in remaining.values() if r["Form"] == form]
        assert len(ms) == 1, (form, len(ms))
        held.append(dict(record={k: ms[0][k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}, reason=reason))

assert len(accepted) == 6 and len(held) == 4
(BASE / "pass074-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(BASE / "pass074-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
lines += ["", "Four records held:"]
lines += [f"- **{h['record']['Form']}** ‘{h['record']['Gloss']}’: {h['reason']}" for h in held]
lines += ["", "Six assignment rows affect six records; all are reflexes. The perfect forms are source-defined stems of their present verbs, not new synchronic derivations. Discovery: Jambu homepage zargari-search013, complete CDIAL prose zargari-primary011, source glyph decoding zargari-primary-pages. No rebuild.", "", "Save and validation pending."]
(BASE / "pass074-review.md").write_text("\n".join(lines) + "\n")
print("accepted", len(accepted), "held", len(held))
