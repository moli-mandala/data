"""Prepare the directly glossed simplex entries in Samuells's 1856 Juang glossary."""

import argparse
import csv
import json
import shutil
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STEM = "20260925-samuells-juang"
SOURCE = "samuells1856juang"
DIALECT = "dialect:ju:samuells-1856-juang:Juang%20%28Samuells%201856%29"
EXPECTED_STATUSES = {"selected": 19, "selected_alternates": 1, "held": 11}


def build():
    with (ROOT / "reviewed_inventory.tsv").open(encoding="utf-8", newline="") as handle:
        inventory = list(csv.DictReader(handle, delimiter="\t"))
    assert len(inventory) == 31
    assert Counter(item["status"] for item in inventory) == EXPECTED_STATUSES
    assert Counter(int(item["printed_page"]) for item in inventory) == {302: 24, 303: 7}
    rows, audit = [], []
    for index, item in enumerate(inventory, 1):
        assert int(item["item"]) == index
        page = int(item["printed_page"])
        assert page in (302, 303)
        key = f"{SOURCE}:p{page}:vocabulary:{index:02}"
        prompt = item["english_prompt"].strip()
        printed = item["printed_juang"].strip()
        status = item["status"]
        forms = [form.strip() for form in item["selected_forms"].split("|") if form.strip()]
        note = item["note"].strip()
        assert all((prompt, printed, note))
        assert all(s == unicodedata.normalize("NFC", s) for s in (prompt, printed, *forms))
        assert (status == "held") == (not forms)
        if status == "selected":
            assert forms == [printed]
        if status == "selected_alternates":
            assert len(forms) == 2 and " or ".join(forms) == printed
        audit.append({
            "entry_key": key,
            "item": index,
            "printed_page": page,
            "pdf_page": page + 32,
            "english_prompt": prompt,
            "printed_juang": printed,
            "status": status,
            "selected_forms": forms,
            "note": note,
        })
        for variant, form in enumerate(forms, 1):
            row_key = f"{key}:v{variant}" if len(forms) > 1 else key
            comment = f"Printed alternate {variant}/2 for water" if len(forms) > 1 else ""
            rows.append([
                "ju", "", form.casefold(), prompt.casefold(), "", "", form,
                f"{SOURCE}[p. {page}, vocabulary item {index}]", "", comment,
                row_key, "", "", "", DIALECT,
            ])
    assert len(rows) == 21 and len({row[10] for row in rows}) == 21
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
    print(f"{len(rows)} forms from {len(audit)} printed Juang prompts; no database built")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    write(parser.parse_args().install)
