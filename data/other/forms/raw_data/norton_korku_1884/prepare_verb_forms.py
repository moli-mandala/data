"""Stage grammatically labelled Norton verb forms for source-local review.

The printed notes on p. 178 explain the three forward-vocabulary slots as
present imperative, present indicative/future, and past indicative. This
script never infers a stem. It does not install generated rows.
"""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SKIP = {
    (165, "left", 1),  # negative fourth slot and abbreviated slash notation
    (166, "left", 12),  # source explicitly calls the word Hindustani
    (169, "left", 3),  # source punctuation confounds slot splitting
    (169, "right", 27),  # cross-reference to lift, not full distinct paradigm
    (170, "left", 29),  # F.-marked supplement embedded inside the paradigm
    (170, "right", 18),  # lone form without identified slot
    (170, "right", 22),  # strike slots appear in different order in print
    (170, "right", 24),  # F.-marked alternative and only two slots
    (171, "left", 4),  # unlabeled three forms with no secure slot order
}
SINGLE_IMPERATIVES = {
    (166, "left", 8), (167, "right", 10),
    (168, "right", 15), (168, "right", 27),
    (169, "right", 14), (169, "right", 16),
    (170, "left", 2), (170, "left", 13),
}
TAGS = ("verb impv", "verb pres ind", "verb pret ind")


def alternatives(slot):
    slot = slot.strip()
    if slot in {"", "—"}:
        return []
    return [part.strip() for part in slot.replace("/", " or ").split(" or ") if part.strip()]


def main():
    with (ROOT / "verb_paradigm_screen.tsv").open(encoding="utf-8", newline="") as handle:
        source = list(csv.DictReader(handle, delimiter="\t"))
    rows = []
    decisions = []
    for item in source:
        key = (int(item["page"]), item["column"], int(item["item"]))
        if key in SKIP or item["extra"]:
            decisions.append((key, "held", "Irregular/ambiguous printed slot structure"))
            continue
        slots = [item[f"candidate_slot{i}"] for i in range(1, 4)]
        if not slots[1] and not slots[2] and key not in SINGLE_IMPERATIVES:
            decisions.append((key, "held", "Single form not explicitly identified as imperative"))
            continue
        if not slots[2] and slots[1] and not slots[0]:
            decisions.append((key, "held", "Insufficient printed slot labels"))
            continue
        for i, slot in enumerate(slots):
            for n, form in enumerate(alternatives(slot), 1):
                if "(F.)" in form or "(See " in form or form == "etc.":
                    continue
                rows.append({
                    "page": key[0], "column": key[1], "item": key[2],
                    "gloss": item["gloss"], "slot": i + 1, "alternative": n,
                    "form": form, "tags": TAGS[i],
                    "review_status": "candidate",
                    "note": "Printed slot interpreted according to Norton p. 178; exact form requires visual review",
                })
        decisions.append((key, "candidate", "Printed slot sequence parsed without stem inference"))
    with (ROOT / "verb_forms_candidates.tsv").open("w", encoding="utf-8", newline="") as handle:
        fields = ("page", "column", "item", "gloss", "slot", "alternative", "form", "tags", "review_status", "note")
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} candidate inflected forms, {sum(d[1] == 'held' for d in decisions)} held verb cells")


if __name__ == "__main__":
    main()
