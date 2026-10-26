"""Prepare Khetrani family and nature words from the LSI comparative vocabulary."""
import csv
import json
import pickle
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments


base = Path("curation/etymology-lab/indo-aryan-diversity-20260922")
targets = [
    ("f_dumvnk4yyyb2q", "8179", "Turner 8179 explicitly lists Khetrani piū́ 'father' beside Lahnda peo and Punjabi piu/peo. Grierson peū has the same p-vowel-u stem and exact kin sense."),
    ("f_lsaxptpehtoas", "9349", "Turner 9349 gives Khetrani bhε̃ṇ 'sister' beside Lahnda bhêṇ. Grierson's bʰēn preserves the same bʰ-n sister stem; its source nasal notation is not silently changed."),
    ("f_s47sjt3acx6ss", "8265", "Turner 8265 explicitly lists Khetrani putr̥ 'son' and Lahnda puttur. Grierson's putur has the shared put-r consonants and exact family sense."),
    ("f_yrpcb3sfdfypw", "5009", "Turner 5009 explicitly cites Khetrani cālī f. 'goat' under chāgala. Grierson's cālī 'female goat' matches form, gender, and sense."),
    ("f_4jenuxng2anu4", "6152", "Turner 6152 cites Khetrani dant and Lahnda dand 'tooth'. Grierson's dand matches the latter local assimilation; the source's voiced final consonant is retained."),
    ("f_lbt57bvxyyy2a", "7636", "Turner 7636 explicitly lists Khetrani pakhī 'bird' under pakṣín. Grierson's pakʰī is the same bird word in the source transcription."),
    ("f_vf447fbm4dshm", "9480", "Turner 9480 explicitly cites Khetrani bah 'fire' under bhāsá 'light', beside Lahnda bhāh. Grierson's bah has the exact Khetrani form and local fire sense."),
    ("f_scn3ua5d6m6zu", "10104", "Turner 10104 explicitly lists Khetrani māh 'moon, month' under mā́sa. Grierson's māh 'Moon' is the moon sense of that same form."),
]
ids = {x[0] for x in targets}
records = {r["ID"]: r for r in csv.DictReader(open("cldf/forms.csv")) if r["ID"] in ids}
assert len(records) == len(ids)
source_rows = list(csv.reader(open("data/other/forms/20260813-grierson-lsi.csv")))
linked = {r["Form_ID"] for r in read_assignments() if r["Status"] == "accepted"}
linked.update(r["Child_ID"] for r in csv.DictReader(open("cldf/edges.csv")) if r["Rank"] == "1")
accepted = []
audit = []
for fid, parent, evidence in targets:
    r = records[fid]
    assert fid not in linked and r["Language_ID"] == "khet" and "LSI-KHETRANI" in r["Tags"]
    matches = [x for x in source_rows if len(x) > 10 and x[0] == "LSI-KHETRANI" and x[7] == r["Source"]]
    assert len(matches) == 1, (fid, len(matches))
    raw = matches[0]
    assert raw[3] == r["Gloss"] and unicodedata.normalize("NFC", raw[2]) == unicodedata.normalize("NFC", r["Original"])
    accepted.append({
        "record": {k: r[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")},
        "parent": parent,
        "kind": "reflex",
        "citation": f"{r['Source']};CDIAL[{parent}]",
        "evidence": evidence + " Exact digital-edition key and original spelling checked; no import or transcription change.",
    })
    audit.append({"id": fid, "raw_source_row": raw, "compiled_original": r["Original"], "compiled_form": r["Form"], "source": r["Source"]})

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
for n, phrase in {"8179": "khet. piū́", "9349": "khet. bhε", "8265": "khet. putr̥", "5009": "khet. (LSI) cālī", "6152": "khet. dant", "7636": "khet. pakhī", "9480": "khet. bah", "10104": "khet. māh"}.items():
    assert unicodedata.normalize("NFD", phrase) in unicodedata.normalize("NFD", primary[n]["text"]), (n, phrase)
(base / "khetrani-primary002.json").write_text(json.dumps(primary, ensure_ascii=False, indent=2) + "\n")
(base / "pass117-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
held = [
    {"form": "abbā", "gloss": "Father", "reason": "Likely nursery/contact word; the immediate history should not be flattened to pitŕ̥."},
    {"form": "candr", "gloss": "Moon", "reason": "A separate survey synonym that may be learned or borrowed; assess the immediate source before linking to candrá."},
    {"form": "gāũ", "gloss": "Cow", "reason": "The northern go/gāũ family occurs under two Turner ancestral branches; this source form needs branch review."},
]
(base / "pass117-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
table = [
    "| 1 | **peū** ‘father’ | **pitŕ̥**, CDIAL **8179** | Turner lists Khetrani *piū́* and neighbouring Lahnda *peo*. |",
    "| 2 | **bʰēn** ‘sister’ | **bhaginī**, CDIAL **9349** | Turner lists Khetrani *bhε̃ṇ* and Lahnda *bhêṇ*. |",
    "| 3 | **putur** ‘son’ | **putrá**, CDIAL **8265** | Turner lists Khetrani *putr̥* and Lahnda *puttur*. |",
    "| 4 | **cālī** ‘female goat’ | **chāgala**, CDIAL **5009** | Turner cites Khetrani *cālī* f. ‘goat’ exactly. |",
    "| 5 | **dand** ‘tooth’ | **dánta**, CDIAL **6152** | Turner gives Khetrani *dant* and Lahnda *dand* ‘tooth’. |",
    "| 6 | **pakʰī** ‘bird’ | **pakṣín**, CDIAL **7636** | Turner gives Khetrani *pakhī* ‘bird’ directly. |",
    "| 7 | **bah** ‘fire’ | **bhāsá**, CDIAL **9480** | Turner gives Khetrani *bah* ‘fire’ directly. |",
    "| 8 | **māh** ‘moon’ | **mā́sa**, CDIAL **10104** | Turner gives Khetrani *māh* ‘moon, month’ directly. |",
]
(base / "pass117-review.md").write_text("\n".join([
    "# Grierson Khetrani: family, animal and nature words",
    "",
    "Eight source-backed reflex records in the digital LSI comparative vocabulary, prepared under standing autonomous authorization. Exact original-spelling rows and source locators were checked, and complete Turner articles are in `khetrani-primary002.json`. Original physical table images have not yet been independently checked.",
    "",
    "| # | Khetrani | Proposed etymology | Evidence |",
    "|---|---|---|---|",
    *table,
    "",
    "Eight proposed `reflex` rows affect eight Khetrani records. **abbā** ‘father’, **candr** ‘moon’, and **gāũ** ‘cow’ remain held for immediate donor or historical branch. Six earlier numeral records are already saved in pass114; no duplicate target is proposed here. No source ingestion or rebuild.",
    "",
    "The Jambu homepage service was unavailable locally; exact compiled Turner comparanda led to the full cached primary articles. Validation pending.",
]) + "\n")
print("accepted", len(accepted), "held", len(held))
