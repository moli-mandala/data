"""Record visual reverse-index comparison for both printed p. 174 columns."""

import csv
from pathlib import Path


PATH = Path(__file__).with_name("reverse_inventory_candidates.tsv")
ADDITIONS = {
    ("left", 6): ("gehu", "wheat", "Reverse head differs from forward printed gchū"),
    ("left", 8): ("gili", "long", "Direct reverse head; forward gill is F.-qualified"),
    ("left", 28): ("ī", "who", "Reverse-only interrogative pronoun"),
    ("left", 29): ("īa", "whose", "Reverse-only possessive interrogative pronoun"),
    ("left", 31): ("iñ", "I", "Reverse-only personal pronoun"),
    ("left", 32): ("iñya", "mine", "Distinct from forward īñye"),
    ("left", 34): ("inīja", "his; her; its", "Reverse-only possessive pronoun"),
    ("right", 3): ("jahāzo", "boat", "Printed variant beside forward jahājo"),
    ("right", 4): ("jāisā", "as", "Distinct from forward hindar"),
    ("right", 10): ("jilya", "peacock", "Printed variant beside forward jelya"),
    ("right", 25): ("kakar", "worry", "Printed variant beside forward kakai"),
    ("right", 29): ("kandā", "piece", "Reverse-only lexeme"),
}
HOLDS = {
    ("right", 23): "kabdūr has no printed gloss; cannot assign a sense",
    ("right", 34): "Reverse prints kapār, he; forward prints kapār head—likely truncated typo, not secure pronoun",
}


def main():
    with PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    selected = [row for row in rows if row["page"] == "174"]
    assert len(selected) == 77
    for row in selected:
        key = (row["column"], int(row["ordinal"]))
        assert row["review_status"] == "pending"
        row["review_status"] = "same-record"
        row["review_note"] = "Visually compared with printed p. 174 and forward vocabulary"
        if key in ADDITIONS:
            row["reviewed_form"], row["reviewed_gloss"], row["review_note"] = ADDITIONS[key]
            row["review_status"] = "reverse-addition"
        if key in HOLDS:
            row["review_status"] = "held"
            row["review_note"] = HOLDS[key]
    assert set(ADDITIONS) | set(HOLDS) <= {
        (row["column"], int(row["ordinal"])) for row in selected
    }
    with PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Reviewed {len(selected)} p. 174 heads: {len(ADDITIONS)} additions, {len(HOLDS)} holds")


if __name__ == "__main__":
    main()
