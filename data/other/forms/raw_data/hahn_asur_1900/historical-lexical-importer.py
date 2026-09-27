"""Prepare the lexical lists and paradigms in Hahn's complete Asur primer."""

import argparse
import csv
import json
import shutil
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STEM = "20260925-hahn-asur"
SOURCE = "hahn1900asur"
DIALECT = "dialect:Asuri:hahn-1900-asur-dukma:Asur%20Dukma%20%28Hahn%201900%29"
EXPECTED_GROUPS = {"h_to_v": 2, "prefixed_y": 3, "other_differences": 26}
EXPECTED_STATUSES = {"selected": 29, "held": 2}


def build_p170():
    with (ROOT / "reviewed_inventory.tsv").open(encoding="utf-8", newline="") as handle:
        inventory = list(csv.DictReader(handle, delimiter="\t"))
    assert len(inventory) == 31
    assert Counter(item["section"] for item in inventory) == EXPECTED_GROUPS
    assert Counter(item["status"] for item in inventory) == EXPECTED_STATUSES
    rows, audit = [], []
    for index, item in enumerate(inventory, 1):
        assert int(item["item"]) == index
        key = f"{SOURCE}:p170:comparison:{index:02}"
        mundari = item["mundari_control"].strip()
        asur = item["asur_printed"].strip()
        gloss = item["gloss"].strip()
        status = item["status"]
        note = item["note"].strip()
        assert all((mundari, asur, gloss, note))
        assert all(x == unicodedata.normalize("NFC", x) for x in (mundari, asur, gloss))
        audit.append({
            "entry_key": key,
            "printed_page": 170,
            "pdf_page": 182,
            "section": item["section"],
            "item": index,
            "mundari_control": mundari,
            "asur_printed": asur,
            "printed_english_gloss": gloss,
            "status": status,
            "note": note,
        })
        if status == "held":
            continue
        assert status == "selected" and "," not in asur
        rows.append([
            "Asuri", "", asur.casefold(), gloss, "", "", asur,
            f"{SOURCE}[p. 170, Asur–Mundari comparison, item {index}]",
            "", f"Mundari comparison control: {mundari}", key, "", "", "", DIALECT,
        ])
    assert len(rows) == 29 and len({row[10] for row in rows}) == 29
    return rows, audit


def build():
    rows, audit = build_p170()
    with (ROOT / "full_inventory.tsv").open(encoding="utf-8", newline="") as handle:
        inventory = list(csv.DictReader(handle, delimiter="\t"))
    for item in inventory:
        page = int(item["printed_page"])
        key = f"{SOURCE}:p{page}:s{item['section']}:{int(item['item']):02}"
        record = dict(item, entry_key=key, pdf_page=page + 12,
                      raw_page_ocr="article-ocr.txt", uncertainty_type="")
        audit.append(record)
        if item["status"] != "selected":
            if item["status"] == "held":
                record["uncertainty_type"] = "transcription_or_gloss"
            continue
        forms = item["source_form"].split("|")
        for number, form in enumerate(forms, 1):
            form = unicodedata.normalize("NFC", form.strip())
            child = key if len(forms) == 1 else f"{key}:variant:{number}"
            tags = item["tags"].split()
            if not tags:
                if item["section"].split("-")[0] in {"39", "40", "41"}:
                    tags = ["adv"]
                elif item["section"] == "18":
                    tags = ["noun", "kinship", "poss"]
                elif item["section"] == "43":
                    tags = ["num"]
                elif item["section"] == "44":
                    tags = ["num", "ord"]
                elif item["section"] == "45":
                    tags = ["postp"]
                elif item["section"] == "46":
                    tags = ["conj"] if int(item["item"]) < 13 else ["interj"]
            if "ú" in form:
                tags.append("uncertain")
                record["uncertainty_type"] = "source_diacritic"
                record["note"] += " Printed acute ú retained: source orthography explains ū but not ú; no linguistic emendation."
            etymology = ""
            if item["section"] == "49":
                etymology = "Hahn tentatively compares this Asur form with Kurukh; he explicitly leaves the Dravidian versus Kolarian origin of some items disputed. No donor or cognacy link inferred."
            if page == 171 and item["section"] == "50-right" and item["item"] == "4":
                etymology = "Hahn: perhaps the seter in Mundari."
            if page == 171 and item["section"] == "50-right" and item["item"] == "5":
                etymology = "Hahn: perhaps connected with the Kurukh ērnā, see."
            rows.append(["Asuri", "", form, item["gloss"], "", "", "",
                         f"{SOURCE}[p. {page}, section {item['section']}, item {item['item']}]",
                         "", etymology, child, f"{key}:variant:1" if number > 1 else "", "", "", " ".join([DIALECT] + tags)])
        record["emitted_rows"] = len(forms)
    assert len({row[10] for row in rows}) == len(rows)
    assert len({item["entry_key"] for item in audit}) == len(audit)
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
    print(f"{len(rows)} Asur forms from {len(audit)} accounted source records; no database built")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    write(parser.parse_args().install)
