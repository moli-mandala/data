"""Prepare the exact Lehr Darrai Pashai read stem for etymology review."""
import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments


base = Path("curation/etymology-lab/indo-aryan-diversity-20260922")
form_id = "f_q7p2owqzz3vec"
record = next(r for r in csv.DictReader(open("cldf/forms.csv")) if r["ID"] == form_id)
assert (record["Form"], record["Gloss"], record["Source"]) == ("paṛ-", "read", "lehr")
assert record["Tags"] == "dialect:Pas:dar:Darrai-i%20Nur"
raw = [r for r in csv.reader(open("data/other/forms/20230705-pashai.csv")) if len(r) > 3 and r[2:4] == ["paṛ-", "read"]]
assert len(raw) == 1 and raw[0][0] == "dar"
linked = {r["Form_ID"] for r in read_assignments() if r["Status"] == "accepted"}
linked.update(r["Child_ID"] for r in csv.DictReader(open("cldf/edges.csv")) if r["Rank"] == "1")
assert form_id not in linked

# Lookup follows exact CDIAL 7712 after candidate discovery in compiled forms;
# it does not scan the HTML corpus for new candidates.
entry = None
for page_num, html in enumerate(pickle.load(open("data/cdial/cdial.pickle", "rb")), 1):
    if "<number>7712</number>" not in html:
        continue
    soup = BeautifulSoup(html, "html.parser")
    for paragraph in soup.select("p"):
        if paragraph.find("number") and paragraph.find("number").get_text(strip=True) == "7712":
            entry = {"page": page_num, "url": f"https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page={page_num}", "html": str(paragraph), "text": paragraph.get_text(" ", strip=True)}
            break
    break
assert entry and "Paš. paṛ" in entry["text"] and "Addenda" in entry["text"]
(base / "pashai-primary019.json").write_text(json.dumps({"7712": entry}, ensure_ascii=False, indent=2) + "\n")

evidence = "Turner 7712 explicitly lists Pashai paṛ- 'reads' immediately after Prakrit paḍhaï; Lehr's Darrai paṛ- 'read' is an exact lexical and semantic match. The causative 'teach' forms in the same article are distinct and are not assigned here."
accepted = [{
    "record": {k: record[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")},
    "parent": "7712",
    "kind": "reflex",
    "citation": "lehr;CDIAL[7712]",
    "evidence": evidence,
}]
held = [
    {"form": "koṭ-", "gloss": "grind, crush, pound, hit", "reason": "Turner 3241 groups Paš. kuṭ- under kuṭṭáyati but also has a separate *kōṭṭayati branch; Darrai o does not yet select between them."},
    {"form": "tost-", "gloss": "stand", "reason": "Turner 5745 gives Paš. testewā́ 'standing' with restoration of s; the relation of Lehr's finite/infinitive stem to that participial form needs morphology review."},
]
(base / "pass103-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(base / "pass103-review.md").write_text("\n".join([
    "# Lehr Darrai Pashai: read",
    "",
    "One direct source-backed verbal reflex prepared under standing autonomous authorization. The exact Lehr raw record and persistent compiled identity were checked; the complete Turner 7712 article and addendum are saved in `pashai-primary019.json`.",
    "",
    "| # | Lehr Darrai | Proposed etymology | Evidence |",
    "|---|---|---|---|",
    "| 1 | **paṛ-** ‘read’ | **páṭhati**, CDIAL **7712** | Turner explicitly gives Pashai *paṛ-* ‘reads’ in the non-causative series; Lehr's Darrai stem matches both form and meaning. |",
    "",
    "One proposed `reflex` row affects one compiled record. **koṭ-** ‘grind, crush, pound, hit’ is held because Turner 3241 separates *kuṭṭáyati* and *kōṭṭayati* branches; **tost-** ‘stand’ is held pending comparison of its stem with Turner’s Pashai participial *testewā́*. No source page locator is available for the Lehr CSV, so none is invented.",
    "",
    "The Jambu homepage service was unavailable locally; exact compiled CDIAL comparanda were used for candidate discovery, followed by the complete cached primary article. Validation pending. No rebuild.",
]) + "\n")
print("accepted", len(accepted), "held", len(held))
