"""Conservative exact-record Rama Gondi first-500 proposals and explicit holds."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / "survey-inventory.json").read_text())
assert len(records) == 500
proposals = {}


def add(gloss, forms, parent, evidence):
    hits = []
    for i, rec in enumerate(records, 1):
        if rec["Gloss"] != gloss or rec["Form"] not in forms:
            continue
        assert i not in proposals
        proposals[i] = (parent, evidence)
        hits.append(i)
    assert hits, (gloss, forms)


add("arm", {"hai", "hai̯"}, 2023,
    "DEDR 2023 gives Gondi kay/kai ‘hand, arm’ and Tamil/Malayalam kai; the two Rama Gondi hai/hai̯ responses are close voiceless-fricative dialect variants of this arm noun.")

accepted = []
for i, (number, evidence) in sorted(proposals.items()):
    rec = records[i - 1]
    accepted.append({"record": rec, "parent": f"d{number}", "kind": "reflex",
                     "citation": rec["Source"] + f";dedr[entry {number}]",
                     "evidence": evidence, "status": "pending-review", "surveyIndex": i})

held = []
for i, rec in enumerate(records, 1):
    if i in proposals:
        continue
    form, gloss = rec["Form"], rec["Gloss"]
    if gloss == "axe" and form == "goḍēl":
        reason = "The full DEDR addendum da32 directly gives Gondi goḍel ‘axe’, but da32 has no linkable parent node in the current graph; no new node is added in this pass."
    elif gloss == "blood" and form.startswith(("nakt", "nat")):
        reason = f"{form!r} ‘blood’ resembles Gondi nattur/nettur in DEDR 3748, but the source's /kt/ or /ktar/ cluster is not explained by the attested comparisons."
    elif gloss == "bone" and form.startswith(("haḍ", "hada", "haṛ", "hard")):
        reason = f"{form!r} ‘bone’ is close to regional haḍḍī/haddi vocabulary; an immediate donor or an inherited Gondi correspondence is not established."
    elif gloss == "breast" and form.startswith(("cat", "cʰat", "sat", "satʰ", "ts", "śi", "dud")):
        reason = f"{form!r} ‘breast’ resembles regional chātī/dūdh forms; the immediate donor and sense development in this Gondi site are unresolved."
    elif gloss in {"cabbage", "cauliflower"}:
        reason = f"{form!r} ‘{gloss}’ contains regional gobi/kobi vocabulary, often compounded with another plant term; its immediate donor and complete component analysis are not attested."
    elif gloss == "chili" and form.startswith(("mir", "mɛr", "mid")):
        reason = f"{form!r} ‘chili’ resembles regional mirc/mirci vocabulary; the immediate donor into this Gondi site is not attested."
    elif gloss == "bad" and form.startswith(("kar", "kʰar", "kⁿar", "bek")):
        reason = f"{form!r} ‘bad’ resembles regional kharāb/bekār vocabulary; the immediate donor into this site is unresolved."
    elif gloss == "all" and form.startswith(("sab", "sub", "sar", "sam")):
        reason = f"{form!r} ‘all’ resembles regional sab/sarv vocabulary; its immediate donor or inherited analysis in this site is unresolved."
    elif gloss == "burn":
        reason = f"{form!r} ‘burn’ is a finite or derived response; its stem and relation to a particular DEDR article are not securely analysed."
    elif gloss in {"bite", "broken"}:
        reason = f"{form!r} ‘{gloss}’ is an inflected response whose stem and sound correspondence are not secure."
    elif "(" in form or " " in form:
        reason = f"{form!r} ‘{gloss}’ contains source parentheses or multiple components; a full secure analysis is lacking."
    else:
        reason = f"{form!r} ‘{gloss}’ lacks a secure dialect-specific comparison or an attested immediate donor."
    held.append({"record": rec, "reason": reason, "status": "research-hold", "surveyIndex": i})

parents = {x["parent"] for x in accepted}
articles = {}
with gzip.open(P.parents[2] / "data/dedr/entry-texts-audit.csv.gz", "rt") as stream:
    for row in csv.DictReader(stream):
        if row["Entry_ID"] in parents:
            articles[row["Entry_ID"]] = {k: row[k] for k in ["Page", "Citation", "Raw_Article"]}
            articles[row["Entry_ID"]]["URL"] = "https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page=" + row["Page"]
assert set(articles) == parents
assert len(accepted) + len(held) == len(records)
(P / "proposed-decisions.json").write_text(json.dumps({"status": "pending-review", "accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(P / "research-holds.json").write_text(json.dumps(held, ensure_ascii=False, indent=2) + "\n")
(P / "primary-evidence.json").write_text(json.dumps(articles, ensure_ascii=False, indent=2) + "\n")
(P / "source-record-evidence.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
(P / "unexamined-inventory.json").write_text(json.dumps(json.loads((P / "deferred-inventory.json").read_text()), ensure_ascii=False, indent=2) + "\n")
print(len(accepted), "proposed records,", len(held), "holds,", len(articles), "parent articles")
