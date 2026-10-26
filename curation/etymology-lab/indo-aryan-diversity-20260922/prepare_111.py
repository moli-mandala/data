"""Prepare three direct Turner comparisons in Lehr Darrai Pashai."""
import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments


base = Path("curation/etymology-lab/indo-aryan-diversity-20260922")
targets = [
    ("f_b3wo3jdgf2co6", "6495", "Turner 6495 explicitly gives Pashai dūr in the distant/far series. Lehr's Darrai dur 'far' matches the form and meaning while retaining its source vowel notation."),
    ("f_mpl27hhrgqesy", "2037", "Turner 2037 explicitly cites Pashai ōbarā 'high' under *udbhara 'rising'. Lehr's Darrai obārā has the same consonants, vowels, and meaning; its accent/length placement remains as printed."),
    ("f_6tssxga6mbh7g", "10951", "Turner 10951 explicitly cites Pashai lam 'slowly' under lamba 'pendent/long'. Lehr's Darrai lam 'slow' is the corresponding adjectival/adverbial local use, with no extra suffix posited."),
]
ids = {x[0] for x in targets}
records = {r["ID"]: r for r in csv.DictReader(open("cldf/forms.csv")) if r["ID"] in ids}
assert len(records) == 3
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
for n, phrase in {"6495": "Paš. dūr", "2037": "Paš. ōbarā", "10951": "Paš. lam"}.items():
    assert phrase in primary[n]["text"]
(base / "pashai-primary021.json").write_text(json.dumps(primary, ensure_ascii=False, indent=2) + "\n")
held = [{"form": "aśpāṛ", "gloss": "rice", "reason": "Turner 14193 gives this Pashai form only conditionally under akṣavāṭa, pointing instead to a hypothesized *ākṣapāṭa; the exact parent is unsettled."}]
(base / "pass111-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(base / "pass111-review.md").write_text("\n".join([
    "# Lehr Darrai Pashai: far, high and slow",
    "",
    "Three direct source-backed reflexes prepared under standing autonomous authorization. Exact raw Lehr rows and persistent compiled identities checked; complete Turner articles and addenda are in `pashai-primary021.json`.",
    "",
    "| # | Lehr Darrai | Proposed etymology | Evidence |",
    "|---|---|---|---|",
    "| 1 | **dur** ‘far’ | **dūrá**, CDIAL **6495** | Turner explicitly gives Pashai *dūr* ‘distant’; Lehr's source vowel notation is preserved. |",
    "| 2 | **obārā** ‘high’ | **\\*udbhara**, CDIAL **2037** | Turner explicitly gives Pashai *ōbarā* ‘high’ under this reconstructed rising stem. |",
    "| 3 | **lam** ‘slow’ | **lamba**, CDIAL **10951** | Turner explicitly gives Pashai *lam* ‘slowly’; the local adjective/adverb distinction is retained. |",
    "",
    "Three proposed `reflex` rows affect three compiled records. **aśpāṛ** ‘rice’ is held because Turner 14193 assigns the Pashai form only conditionally between *akṣavāṭa* and a hypothetical *ākṣapāṭa*. No source page locator is available for the Lehr CSV, so none is invented.",
    "",
    "The Jambu homepage service was unavailable locally; exact compiled CDIAL comparanda were used for candidate discovery, followed by full cached primary entries. Validation pending. No rebuild.",
]) + "\n")
print("accepted", len(accepted), "held", len(held))
