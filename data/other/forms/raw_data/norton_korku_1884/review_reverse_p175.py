"""Record visual reverse-index comparison for printed p. 175."""

import csv
from pathlib import Path


PATH = Path(__file__).with_name("reverse_inventory_candidates.tsv")
ADDITIONS = {
    ("left", 5): ("kātlānkan", "fan", "Reverse-only direct noun"),
    ("left", 7): ("kātrē", "skin", "Printed variant beside forward kalrē"),
    ("left", 12): ("khabardao", "be careful", "Printed variant beside forward khabadār"),
    ("left", 13): ("kattā", "sour", "Distinct from forward khatta"),
    ("left", 14): ("kattābā", "is sour", "Reverse-only inflected predicate"),
    ("left", 19): ("kiding", "scorpion", "Printed variant beside forward kidung"),
    ("left", 24): ("kolā kē", "take out", "Distinct from forward olēt-kē"),
    ("left", 30): ("kor", "child", "Distinct from forward seni"),
    ("right", 7): ("lāj", "belly", "Printed variant beside forward bāj"),
    ("right", 11): ("lawa,en", "tired", "Printed variant beside forward luwa,en"),
    ("right", 23): ("mahina", "month", "Forward p169 incorrectly gives same form as mouth; source conflict recorded"),
    ("right", 25): ("māndē", "word", "Reverse head differs from forward māndi word"),
}
HOLDS = {
    ("left", 3): "Forward p167 prints kātān gram, but reverse p175 prints grain; possible compositor error, sense held",
}


def main():
    with PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    selected = [row for row in rows if row["page"] == "175"]
    assert len(selected) == 80
    for row in selected:
        key = (row["column"], int(row["ordinal"]))
        assert row["review_status"] == "pending"
        row["review_status"] = "same-record"
        row["review_note"] = "Visually compared with printed p. 175 and forward vocabulary"
        if key in ADDITIONS:
            row["reviewed_form"], row["reviewed_gloss"], row["review_note"] = ADDITIONS[key]
            row["review_status"] = "reverse-addition"
        if key in HOLDS:
            row["review_status"] = "held"
            row["review_note"] = HOLDS[key]
    with PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Reviewed {len(selected)} p. 175 heads: {len(ADDITIONS)} additions, {len(HOLDS)} holds")


if __name__ == "__main__":
    main()
