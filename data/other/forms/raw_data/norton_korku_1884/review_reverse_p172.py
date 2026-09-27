"""Record visual comparison of all printed p. 172 Kor–English heads."""

import csv
from pathlib import Path


PATH = Path(__file__).with_name("reverse_inventory_candidates.tsv")
LEFT = (
    "ābā", "ādiren", "āḍē", "ai", "aiang", "ājī", "ākī", "ālē", "am", "ama",
    "āmbi", "ānglukī", "ānjūmē", "āpārānge", "ārā", "ārrākī", "ārūkī", "ārsā",
    "āsarū", "āsī", "āsīkī", "ātkom", "ātōdin", "awal sagīrā", "awal",
    "bābā chaulī", "bābūlī", "badako",
)
RIGHT = (
    "badsā", "bāgā-bāgā", "bāgi,ā", "bākī", "bamre", "bānā", "bāng", "barē",
    "barsādo", "barūn", "batkil or bitkil", "bauī", "baurā", "bautā", "bēdai",
    "bēri", "bēriā", "bhātā", "bhagwān or bhagwat", "biānē", "bidē", "bilā",
    "bin", "bindil", "bing", "bitil", "Bitwār", "bo",
)
NEW = {
    ("left", 2): ("ādiren", "arrived", "Reverse-only arrival form beside forward walen"),
    ("left", 15): ("ārā", "herbs", "Distinct from forward āvā"),
    ("left", 21): ("āsīkī", "beg", "Printed vowel differs from forward āsīke"),
    ("right", 1): ("badsā", "cloud; heaven", "Distinct from forward badrā/badarā cloud"),
    ("right", 5): ("bamre", "cold", "Direct reverse head; forward bramre is F.-qualified and held"),
    ("right", 8): ("barē", "about", "Independent sense from forward dī barē therefore"),
    ("right", 12): ("bauī", "back", "Printed variant beside forward baurī"),
    ("right", 14): ("bautā", "bracelet; bangle", "Printed variant beside forward bantā"),
    ("right", 19): ("bhagwat", "God", "Second printed reverse alternative absent forward"),
    ("right", 26): ("bitil", "gravel", "Printed variant beside forward būtil"),
}


def main():
    with PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    reviewed = [row for row in rows if row["page"] == "172"]
    assert len(reviewed) == 56
    for row in reviewed:
        column, ordinal = row["column"], int(row["ordinal"])
        row["reviewed_form"] = (LEFT if column == "left" else RIGHT)[ordinal - 1]
        row["reviewed_gloss"] = row["ocr_gloss"].rstrip(" .")
        row["review_status"] = "same-record"
        row["review_note"] = "Visually checked against printed p. 172; forward form/sense already staged"
        if (column, ordinal) in NEW:
            form, gloss, note = NEW[column, ordinal]
            row["reviewed_form"] = form
            row["reviewed_gloss"] = gloss
            row["review_status"] = "reverse-addition"
            row["review_note"] = note
    by_key = {(row["column"], int(row["ordinal"])) for row in reviewed}
    assert set(NEW) <= by_key
    with PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Reviewed {len(reviewed)} printed p. 172 heads; {len(NEW)} reverse additions")


if __name__ == "__main__":
    main()
