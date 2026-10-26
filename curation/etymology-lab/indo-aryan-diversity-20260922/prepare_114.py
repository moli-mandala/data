"""Prepare six conservative Khetrani numerals from the LSI comparative vocabulary."""
import csv
import json
import pickle
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments


base = Path("curation/etymology-lab/indo-aryan-diversity-20260922")
targets = [
    ("f_5prcthsmizunq", "6648", "Turner 6648 directly lists Khetrani ba 'two' in the Sindhic/Lahnda b- series from dva. Grierson's bā retains the same b- numeral with source-marked length."),
    ("f_bpm6iwjiwitrs", "7655", "Turner 7655 directly lists Khetrani pāñj 'five' alongside Lahnda pañj and Sindhi pañja. Grierson's pañ omits the final affricate in this printed response; the transcription is retained."),
    ("f_jbp24hp2nompe", "12803-3", "Turner 12803 §3 directly lists Khetrani chē 'six' alongside Lahnda chī and Punjabi chī. Grierson's cʰī fits this ch-initial series; the different source vowel and aspiration notation are preserved."),
    ("f_3j2poupamt7oo", "13139", "Turner 13139 gives adjacent Lahnda/Punjabi satt and Dardic sat forms under saptá 'seven'. Grierson's Khetrani sat preserves the diagnostic s-t numeral frame with a single t as printed."),
    ("f_lmnhht5u7dmpy", "941", "Turner 941 directly lists Khetrani aṭh 'eight' under aṣṭā́. Grierson's aṭʰ has the same consonants and gloss, with the source's aspiration placement retained."),
    ("f_q3wo4la7omofg", "6984", "Turner 6984 directly lists Khetrani naũ 'nine' under náva². Grierson's nō̃ is a nasalized rounded-vowel continuation with its source-specific length notation retained."),
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
        "citation": f"{r['Source']};CDIAL[{parent.split('-')[0]}]",
        "evidence": evidence + " The installed source key and original spelling match the digital Grierson edition; no import or profile change is made.",
    })
    audit.append({"id": fid, "raw_source_row": raw, "compiled_original": r["Original"], "compiled_form": r["Form"], "source": r["Source"]})

primary = {}
wanted = {x[1].split("-")[0] for x in targets}
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
for n, phrase in {"6648": "khet. ba", "7655": "khet. pāñj", "12803": "khet. chē", "941": "khet. aṭh", "6984": "khet. naũ"}.items():
    assert unicodedata.normalize("NFD", phrase) in unicodedata.normalize("NFD", primary[n]["text"]), (n, phrase)
(base / "khetrani-primary001.json").write_text(json.dumps(primary, ensure_ascii=False, indent=2) + "\n")
(base / "pass114-source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
held = [
    {"form": "hik", "gloss": "One", "reason": "The h-initial shape requires specific Khetrani branch evidence before selecting plain ēka or reinforced *ēkka."},
    {"form": "trāē", "gloss": "Three", "reason": "Turner 5994 puts Khetrani trē in §1 but Lahnda trāe in §2; the printed Khetrani trāē may indicate either branch."},
    {"form": "cār", "gloss": "Four", "reason": "The relevant catvāri-family branch and precise Khetrani comparison have not been checked in this pass."},
    {"form": "ḍḍāh", "gloss": "Ten", "reason": "Turner 6227 gives Khetrani dā, unlike the printed doubled-retroflex ḍḍāh; source or dialect difference requires review."},
]
(base / "pass114-decisions.json").write_text(json.dumps({"accepted": accepted, "held": held}, ensure_ascii=False, indent=2) + "\n")
(base / "pass114-review.md").write_text("\n".join([
    "# Grierson Khetrani: six basic numerals",
    "",
    "The first six etymology links proposed for the Khetrani column of the 1928 *Linguistic Survey of India* comparative vocabulary, under standing autonomous authorization. The installed forms were checked against the exact original-spelling rows in the digital edition and full Turner articles/addenda. A scan-image check of Grierson's physical table is still pending and is recorded as a limitation.",
    "",
    "| # | Khetrani | Proposed etymology | Evidence |",
    "|---|---|---|---|",
    "| 1 | **bā** ‘two’ | **dva**, CDIAL **6648** | Turner directly lists Khetrani *ba* in the Sindhic/Lahnda b-initial series. |",
    "| 2 | **pañ** ‘five’ | **páñca**, CDIAL **7655** | Turner gives Khetrani *pāñj* beside Lahnda *pañj*; the printed shorter source form is retained. |",
    "| 3 | **cʰī** ‘six’ | **\\*kṣaṭ §3**, CDIAL **12803-3** | Turner gives Khetrani *chē* and Lahnda *chī* in this branch; Grierson's vowel and affricate notation are preserved. |",
    "| 4 | **sat** ‘seven’ | **saptá**, CDIAL **13139** | Turner gives regional *sat/satt* continuations; the source has single *t*. |",
    "| 5 | **aṭʰ** ‘eight’ | **aṣṭā́**, CDIAL **941** | Turner gives Khetrani *aṭh* directly. |",
    "| 6 | **nō̃** ‘nine’ | **náva²**, CDIAL **6984** | Turner gives Khetrani *naũ* directly, a nasalized rounded-vowel numeral. |",
    "",
    "Six proposed `reflex` rows affect six Khetrani records. **hik** ‘one’, **trāē** ‘three’, **cār** ‘four’, and **ḍḍāh** ‘ten’ are held for branch or source-form review. Exact source keys, original spellings and Turner prose are in the audit and primary evidence files. No source ingestion or rebuild.",
    "",
    "The Jambu homepage service was unavailable locally; compiled CDIAL candidates led to full primary entries. Validation pending.",
]) + "\n")
print("accepted", len(accepted), "held", len(held))
