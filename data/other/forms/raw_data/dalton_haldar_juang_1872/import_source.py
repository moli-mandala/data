"""Import Haldar's complete Dalton 1872 Juanga comparison and separate list."""

import argparse
import csv
import json
import shutil
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STEM = "20260925-dalton-haldar-juang"
SOURCE = "dalton1872haldarjuang"
DIALECT = "dialect:ju:haldar-1872-juanga:Juanga%20%28Haldar%20compiled%201872%29"
EXPECTED_STATUSES = {"selected": 27, "selected_alternates": 2, "held": 10, "blank": 3}
CONTROL_NAMES = ("Mundari", "Santal", "Korwa", "Kharia", "Ho", "Kuri/Muasi", "Talain/Mon", "Khasi")


def build_p236():
    with (ROOT / "reviewed_inventory.tsv").open(encoding="utf-8", newline="") as handle:
        inventory = list(csv.DictReader(handle, delimiter="\t"))
    assert len(inventory) == 42
    assert Counter(item["status"] for item in inventory) == EXPECTED_STATUSES
    rows, audit = [], []
    for index, item in enumerate(inventory, 1):
        assert int(item["row"]) == index
        key = f"{SOURCE}:p236:row:{index:02}"
        prompt = item["english_prompt"].strip()
        printed = item["juanga_printed"].strip()
        status = item["status"]
        forms = [form.strip() for form in item["selected_forms"].split("|") if form.strip()]
        mask = item["control_presence_mask"].strip()
        note = item["note"].strip()
        assert all((prompt, printed, note))
        assert len(mask) == 8 and set(mask) <= {"0", "1"}
        assert all(s == unicodedata.normalize("NFC", s) for s in (prompt, printed, *forms))
        assert (status in ("held", "blank")) == (not forms)
        if status == "blank":
            assert printed == "..."
        if status == "selected_alternates":
            assert len(forms) == 2
        if status == "selected":
            assert len(forms) == 1
        controls = dict(zip(CONTROL_NAMES, [c == "1" for c in mask]))
        audit.append({
            "entry_key": key,
            "row": index,
            "printed_page": 236,
            "pdf_page": 251,
            "english_prompt": prompt,
            "juanga_printed": printed,
            "status": status,
            "selected_forms": forms,
            "control_columns_present": controls,
            "note": note,
        })
        for variant, form in enumerate(forms, 1):
            row_key = f"{key}:v{variant}" if len(forms) > 1 else key
            comment = ("Source annotation: (s.); no donor edge inferred" if "(s.)" in printed else "")
            if len(forms) > 1:
                comment = f"Printed alternate {variant}/2" + (f"; {comment}" if comment else "")
            rows.append([
                "ju", "", form.casefold(), prompt.casefold(), "", "", form,
                f"{SOURCE}[p. 236, Haldar table, Juanga column, row {index}]", "", comment,
                row_key, "", "", "", DIALECT,
            ])
    assert len(rows) == 31 and len({row[10] for row in rows}) == 31
    return rows, audit


def build():
    rows, audit = build_p236()
    with (ROOT / "comparative-extension.tsv").open(encoding="utf-8", newline="") as handle:
        extension = list(csv.DictReader(handle, delimiter="\t"))
    assert len(extension) == 185
    for page, count in {235: 35, 237: 42, 238: 37, 239: 37, 240: 25, 241: 9}.items():
        assert [int(item["row"]) for item in extension if item["printed_page"] == str(page)] == list(range(1, count + 1))
    for item in extension:
        page = int(item["printed_page"])
        number = int(item["row"])
        assert int(item["pdf_page"]) == page + 15
        assert page in {235, 237, 238, 239, 240, 241}
        assert item["status"] in {"selected", "selected_alternates", "held", "blank"}
        assert item["english_prompt"] and item["juanga_printed"] and item["note"]
        forms = [unicodedata.normalize("NFC", part.strip()) for part in item["reviewed_forms"].split("|") if part.strip()]
        assert bool(forms) == item["status"].startswith("selected")
        assert (item["juanga_printed"] == "...") == (item["status"] == "blank")
        if item["status"] == "selected_alternates":
            assert len(forms) > 1
        key = f"{SOURCE}:p{page}:row:{number:02}"
        audit.append({
            "entry_key": key,
            "row": number,
            "printed_page": page,
            "pdf_page": page + 15,
            "section": "comparison table",
            "english_prompt": item["english_prompt"],
            "juanga_printed": item["juanga_printed"],
            "status": item["status"],
            "selected_forms": forms,
            "note": item["note"],
            "control_columns_present": dict.fromkeys(CONTROL_NAMES, None),
        })
        for variant, form in enumerate(forms, 1):
            row_key = f"{key}:v{variant}" if len(forms) > 1 else key
            comment = "Source annotation: (s.); no donor edge inferred" if "(s.)" in item["juanga_printed"] else ""
            rows.append([
                "ju", "", form.casefold(), item["english_prompt"].casefold(), "", "", form,
                f"{SOURCE}[p. {page}, Haldar table, Juanga column, row {number}]", "", comment,
                row_key, "", "", "", DIALECT,
            ])

    with (ROOT / "separate-list-inventory.tsv").open(encoding="utf-8", newline="") as handle:
        separate = list(csv.DictReader(handle, delimiter="\t"))
    assert len(separate) == 134
    assert len({item["inventory_key"] for item in separate}) == 134
    for item in separate:
        page = int(item["printed_page"])
        assert page in {241, 242} and int(item["pdf_page"]) == page + 15
        assert item["review_status"] in {"reviewed", "held"}
        assert item["english_gloss"] and item["source_reading"]
        forms = [unicodedata.normalize("NFC", part.strip()) for part in item["selected_forms"].split("|") if part.strip()]
        assert bool(forms) == (item["review_status"] == "reviewed")
        key = f"{SOURCE}:{item['inventory_key']}"
        audit.append({
            "entry_key": key,
            "row": int(item["row_index"]),
            "printed_page": page,
            "pdf_page": page + 15,
            "section": f"separate Juang list, {item['section']}",
            "english_prompt": item["english_gloss"],
            "juanga_printed": item["source_reading"],
            "status": "selected" if forms else "held",
            "selected_forms": forms,
            "note": item["review_note"],
            "control_columns_present": {},
        })
        for variant, form in enumerate(forms, 1):
            row_key = f"{key}:v{variant}" if len(forms) > 1 else key
            rows.append([
                "ju", "", form.casefold(), item["english_gloss"].casefold(), "", "", form,
                f"{SOURCE}[p. {page}, separate Juang list, {item['section']}, row {item['row_index']}]",
                "", item["review_note"] if item["review_note"].startswith("Printed note:") else "",
                row_key, "", "", "", DIALECT,
            ])
    assert len(audit) == 42 + 185 + 134
    assert len({row[10] for row in rows}) == len(rows)
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
    print(f"{len(rows)} Juanga forms from {len(audit)} audited comparison/list cells; no database built")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    write(parser.parse_args().install)
