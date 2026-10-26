"""Prepare directly attested Lehr/Turner Darrai Pashai comparisons."""
import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments


base = Path("curation/etymology-lab/indo-aryan-diversity-20260922")
targets = [
    ("f_idgphbbjz6axg", "9348", "Turner 9348 expressly lists Pashai bai 'good' under bhagín, alongside Shumashti and Gawri bäi. Lehr's Darrai bai has the same form and sense."),
    ("f_us5soa4n3f3wk", "1179", "Turner 1179 gives Pashai alenā/alinā 'blue, green' under ānīla. Lehr's Darrai alina 'green' is the same adjective; its unmarked final vowel length is preserved."),
    ("f_f6uvzwygrjhuk", "1179", "Turner 1179 gives Pashai alinā 'blue, green' exactly; Lehr's Darrai alinā 'green' matches form and sense."),
    ("f_spzo6yz6x222q", "4407", "Turner 4407 §2 expressly lists Pashai gaṛ- 'to send' under ghaṭayati. Lehr's Darrai gaṛ- 'send, mail' is the exact stem, and the source's extension to mail is retained. The graph has one unsplit 4407 parent node; §2 is documented here."),
    ("f_gip3h562nuxdq", "8140", "Turner 8140 gives Pashai paṣī́ 'wall' from pāṣī́ 'stone, rock'. Lehr's Darrai paśi 'wall' has a closely matching palatal sibilant and the exact local sense; the source transcription is preserved."),
]
ids = {x[0] for x in targets}
records = {r["ID"]: r for r in csv.DictReader(open("cldf/forms.csv")) if r["ID"] in ids}
assert len(records) == len(ids)
raw = list(csv.reader(open("data/other/forms/20230705-pashai.csv")))
linked = {r["Form_ID"] for r in read_assignments() if r["Status"] == "accepted"}
linked.update(r["Child_ID"] for r in csv.DictReader(open("cldf/edges.csv")) if r["Rank"] == "1")
accepted = []
for fid, parent, evidence in targets:
    r = records[fid]
    assert fid not in linked and r["Source"] == "lehr" and r["Tags"] == "dialect:Pas:dar:Darrai-i%20Nur"
    assert any(len(x) > 3 and x[0] == "dar" and x[2:4] == [r["Form"], r["Gloss"]] for x in raw)
    accepted.append({
        "record": {k: r[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")},
        "parent": parent,
        "kind": "reflex",
        "citation": f"lehr;CDIAL[{parent}]",
        "evidence": evidence,
    })

primary = {}
wanted = {x[1] for x in targets}
for page_num, html in enumerate(pickle.load(open("data/cdial/cdial.pickle", "rb")), 1):
    if not any(f"<number>{n}</number>" in html for n in wanted - primary.keys()):
        continue
    soup = BeautifulSoup(html, "html.parser")
    for paragraph in soup.select("p"):
        number = paragraph.find("number")
        if number and number.get_text(strip=True) in wanted:
            n = number.get_text(strip=True)
            primary[n] = {"page": page_num, "url": f"https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page={page_num}", "html": str(paragraph), "text": paragraph.get_text(" ", strip=True)}
    if len(primary) == len(wanted):
        break
assert len(primary) == len(wanted)
for n, phrase in {"9348": "Paš. bai", "1179": "Paš. alenā", "4407": "Paš. gaṛ", "8140": "Paš. paṣī"}.items():
    assert phrase in primary[n]["text"], (n, phrase)
(base / "pashai-primary020.json").write_text(json.dumps(primary, ensure_ascii=False, indent=2) + "\n")
held = [
    {"form": "dār-", "gloss": "have, hold", "reason": "Turner 6791 gives this exact Pashai form but explicitly flags possible Persian borrowing; the immediate origin is unresolved."},
    {"form": "bā̌s", "gloss": "steam", "reason": "Turner 9223 §4 says the cited Pashai form may be borrowed from Kafiri; an inherited reflex cannot yet be selected."},
    {"form": "uncaṛek", "gloss": "wellspring", "reason": "Turner 14309 identifies it as a diminutive of unč < útsa; decide whether an attested Pashai base supports a derived edge before linking."},
]
(base / "pass107-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
table = [
    "| 1 | **bai** ‘good’ | **bhagín**, CDIAL **9348** | Turner explicitly cites Pashai *bai* ‘good’, exactly matching Lehr. |",
    "| 2 | **alina** ‘green’ | **ānīla**, CDIAL **1179** | Turner gives Pashai *alenā/alinā* ‘blue, green’; Lehr's unmarked final length is preserved. |",
    "| 3 | **alinā** ‘green’ | **ānīla**, CDIAL **1179** | Turner gives this Pashai form exactly. |",
    "| 4 | **gaṛ-** ‘send, mail’ | **ghaṭayati §2**, CDIAL **4407** | Turner explicitly gives Pashai *gaṛ-* ‘to send’ in §2; the compiled parent node is unsplit. |",
    "| 5 | **paśi** ‘wall’ | **pāṣī́**, CDIAL **8140** | Turner gives Pashai *paṣī́* ‘wall’; the source's palatal-sibilant notation is retained. |",
]
(base / "pass107-review.md").write_text("\n".join([
    "# Lehr Darrai Pashai: five direct Turner comparisons",
    "",
    "Five source-backed reflex records across four historical families, prepared under standing autonomous authorization. Exact raw Lehr rows and compiled persistent IDs checked; complete Turner articles and addenda extracted in `pashai-primary020.json`.",
    "",
    "| # | Lehr Darrai | Proposed etymology | Evidence |",
    "|---|---|---|---|",
    *table,
    "",
    "Five proposed assignment rows affect five compiled records. **dār-** ‘have, hold’ is held because Turner flags possible Persian borrowing; **bā̌s** ‘steam’ because Turner flags possible Kafiri borrowing; **uncaṛek** ‘wellspring’ because Turner identifies a diminutive whose direct base has not been established in the graph. No source page locator is available for the Lehr CSV, so none is invented.",
    "",
    "The Jambu homepage service was unavailable locally; exact compiled CDIAL comparanda were used for candidate discovery, followed by full cached primary entries. Validation pending. No rebuild.",
]) + "\n")
print("accepted", len(accepted), "held", len(held))
