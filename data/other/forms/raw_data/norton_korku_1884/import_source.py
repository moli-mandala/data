"""Prepare the reviewed English–Kor forward vocabulary of Cust/Norton 1884."""

import argparse
import csv
import json
import shutil
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
STEM = "20260925-cust-norton-korku"
SOURCE = "cust1884korku"
DIALECT = "dialect:ko:norton-1884:Korku%20%28Norton%201884%29"
EXPECTED_COLUMNS = {165: (32, 33), 166: (33, 34), 167: (27, 33),
                    168: (34, 29), 169: (37, 33), 170: (33, 33),
                    171: (36, 36), 172: (8, 6)}


def build():
    with (ROOT / "reviewed_inventory.tsv").open(encoding="utf-8", newline="") as handle:
        pilot = list(csv.DictReader(handle, delimiter="\t"))
    with (ROOT / "remaining_inventory.tsv").open(encoding="utf-8", newline="") as handle:
        remaining = list(csv.DictReader(handle, delimiter="\t"))
    inventory = [dict(item, page="165") for item in pilot] + remaining
    assert Counter((int(item["page"]), item["column"]) for item in inventory) == {
        (page, column): count for page, counts in EXPECTED_COLUMNS.items()
        for column, count in zip(("left", "right"), counts)
    }
    assert Counter(item["status"] for item in inventory) == {"selected": 370, "held": 107}
    rows, audit = [], []
    seen = set()
    for item in inventory:
        page = int(item["page"])
        column, number = item["column"], int(item["item"])
        assert 1 <= number <= EXPECTED_COLUMNS[page][0 if column == "left" else 1]
        key = f"{SOURCE}:english-kor:p{page}:{column}:item{number:02}"
        assert key not in seen
        seen.add(key)
        forms = [x.strip() for x in item["forms"].split(";") if x.strip()]
        assert all(x == unicodedata.normalize("NFC", x) for x in forms)
        status, note = item["status"], item["note"].strip()
        assert status in {"selected", "held"} and note and item["printed_response"].strip()
        assert bool(forms) == (status == "selected")
        audit.append({
            "entry_key": key,
            "printed_page": page,
            "pdf_page": page + 19,
            "column": column,
            "item": number,
            "gloss": item["gloss"].strip(),
            "printed_response": item["printed_response"].strip(),
            "selected_forms": forms,
            "status": status,
            "note": note,
        })
        if status != "selected":
            continue
        for alternate, form in enumerate(forms, 1):
            form_key = key if len(forms) == 1 else f"{key}:alt{alternate}"
            rows.append([
                "ko", "", form.casefold(), item["gloss"].strip(), "", "", form,
                f"{SOURCE}[p. {page}, English–Kor, {column} column, item {number}]",
                "", "", form_key, "", "", "", DIALECT,
            ])
    assert len(audit) == 477 and len(rows) == 408
    with (ROOT / "numbers_inventory.tsv").open(encoding="utf-8", newline="") as handle:
        numbers = list(csv.DictReader(handle, delimiter="\t"))
    assert len(numbers) == 12
    for number in numbers:
        ordinal = int(number["item"])
        key = f"{SOURCE}:numbers:p177:item{ordinal:02}"
        forms = [form.strip() for form in number["forms"].split(";") if form.strip()]
        assert number["page"] == "177" and number["status"] in {"selected", "reused"}
        assert bool(forms) == (number["status"] == "selected")
        assert all(form == unicodedata.normalize("NFC", form) for form in forms)
        audit.append({
            "entry_key": key, "printed_page": 177, "pdf_page": 196,
            "section": "Numbers", "column": number["column"], "item": ordinal,
            "gloss": number["gloss"], "printed_response": number["printed_response"],
            "selected_forms": forms, "status": number["status"], "note": number["note"],
        })
        for alternative, form in enumerate(forms, 1):
            form_key = key if len(forms) == 1 else f"{key}:alt{alternative}"
            rows.append([
                "ko", "", form.casefold(), number["gloss"], "", "", form,
                f"{SOURCE}[p. 177, Numbers, item {ordinal}]", "", "", form_key,
                "", "", "", DIALECT + " num",
            ])
    assert len(audit) == 489 and len(rows) == 421
    with (ROOT / "verb_forms_candidates.tsv").open(encoding="utf-8", newline="") as handle:
        verbs = list(csv.DictReader(handle, delimiter="\t"))
    assert len(verbs) == 288
    audit_by_key = {item["entry_key"]: item for item in audit}
    for item in verbs:
        page, column, number = int(item["page"]), item["column"], int(item["item"])
        slot, alternate = int(item["slot"]), int(item["alternative"])
        key = f"{SOURCE}:english-kor:p{page}:{column}:item{number:02}"
        form = item["form"].strip()
        assert item["review_status"] == "selected" and form
        assert form == unicodedata.normalize("NFC", form)
        assert audit_by_key[key]["status"] in {"held", "selected_grammatical"}
        if audit_by_key[key]["status"] == "held":
            audit_by_key[key]["note"] += "; inflected slots parsed according to printed p. 178"
        audit_by_key[key]["status"] = "selected_grammatical"
        audit_by_key[key]["selected_forms"].append(form)
        rows.append([
            "ko", "", form.casefold(), item["gloss"], "", "", form,
            f"{SOURCE}[p. {page}, English–Kor, {column} column, item {number}, verb slot {slot}]",
            "", "", f"{key}:slot{slot}:alt{alternate}", "", "", "",
            DIALECT + " " + item["tags"],
        ])
    assert len(audit) == 489 and len(rows) == 709
    with (ROOT / "held_resolution.tsv").open(encoding="utf-8", newline="") as handle:
        resolutions = list(csv.DictReader(handle, delimiter="\t"))
    resolution_counts = Counter()
    held_keys = {key for key, entry in audit_by_key.items() if entry["status"] == "held"}
    for item in resolutions:
        page, column, number = int(item["page"]), item["column"], int(item["item"])
        key = f"{SOURCE}:english-kor:p{page}:{column}:item{number:02}"
        assert key in held_keys, key
        assert item["decision"] in {"selected", "reused"}
        assert item["note"].strip()
        form = item["form"].strip()
        if item["decision"] == "reused":
            assert not form
            audit_by_key[key]["status"] = "reused"
            audit_by_key[key]["note"] += "; " + item["note"]
            continue
        assert form and form == unicodedata.normalize("NFC", form)
        resolution_counts[key] += 1
        audit_by_key[key]["status"] = "selected_grammatical"
        audit_by_key[key]["selected_forms"].append(form)
        audit_by_key[key]["note"] += "; " + item["note"]
        rows.append([
            "ko", "", form.casefold(), item["gloss"], "", "", form,
            f"{SOURCE}[p. {page}, English–Kor, {column} column, item {number}]",
            "", "", f"{key}:resolution{resolution_counts[key]}", "", "", "",
            DIALECT + " " + item["tags"],
        ])
    assert len(audit) == 489 and len(rows) == 709 + sum(resolution_counts.values())
    assert len({row[10] for row in rows}) == len(rows)
    from reconcile_reverse import extend
    from grammar import extend as grammar_extend
    rows, audit = grammar_extend(*extend(rows, audit))
    conflicts = {
        "cust1884korku:english-kor:p169:left:item13": "gloss: forward mouth conflicts with reverse month for mahina; printed sense retained",
        "cust1884korku:kor-english:p173:left:item12": "gloss: reverse gum conflicts with forward gun",
        "cust1884korku:kor-english:p173:left:item33": "gloss: reverse therefore conflicts with forward wherefore",
        "cust1884korku:kor-english:p173:right:item15": "gloss: reverse fate conflicts with forward late",
        "cust1884korku:kor-english:p173:right:item34": "gloss: reverse son conflicts with forward sow",
    }
    for entry in audit:
        if entry["entry_key"] in conflicts:
            entry["review_reason"] = conflicts[entry["entry_key"]]
    audit_lookup = {x['entry_key']: x for x in audit}
    for row in rows:
        origin = next((audit_lookup[k] for k in (row[10], re.sub(r':(?:alt|slot|resolution).*$', '', row[10])) if k in audit_lookup), None)
        if origin and ':english-kor:' in row[10]:
            note = origin['note'].lower()
            for word, tag in (('pronoun','pron'),('noun','noun'),('adjectiv','adj'),('adverb','adv'),('postposition','postp'),('preposition','prep'),('conjunction','conj')):
                if word in note and tag not in row[14].split():
                    row[14] += ' ' + tag
                    break
            if 'only imperative' in note: row[14] += ' verb impv'
        if row[6] == row[2] or row[6].casefold() == row[2]:
            row[6] = ""
        for label, tag in (("noun", "noun"), ("verb", "verb"), ("adjective", "adj")):
            if " (" + label + ")" in row[3]:
                row[3] = row[3].replace(" (" + label + ")", "")
                row[14] += " " + tag
        comparison = re.search(r"\s*\((?:Hind\.|Hin\.)[^)]*\)", row[3])
        if comparison:
            row[6] = comparison.group().strip()
            row[3] = row[3][:comparison.start()] + row[3][comparison.end():]
        if any(row[10] == key or row[10].startswith(key + ":") for key in conflicts):
            row[14] += " uncertain"
    return rows, audit


def write(install=False):
    rows, audit = build()
    with (ROOT / f"{STEM}.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    with (ROOT / "audit.jsonl").open("w", encoding="utf-8") as handle:
        for item in audit:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    if install:
        shutil.copyfile(ROOT / f"{STEM}.csv", ROOT.parents[1] / f"{STEM}.csv")
    print(f"{len(rows)} Korku forms from {len(audit)} forward, reverse, and numeral cells; no database built")


if __name__ == "__main__":
    # The legacy build above remains reproducible evidence for the old snapshots.
    # All current installation goes through the whole-source approval guard.
    from install_full_recovery import main
    main()
