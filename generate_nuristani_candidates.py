"""Generate evidence-based CDIAL candidates for stored Proto-Nuristani heads.

The current Strand lexicon has the same ordered sequence of Proto-Nuristani
heads as our 2022 parse, but now exposes a PIE → Proto-Aryan → OIA hierarchy.
This script aligns those sequences exactly and reports only structural matches;
it does not edit the curated cognate mapping.
"""

import csv
from collections import defaultdict
from pathlib import Path


HIERARCHY = Path("data/strand_hierarchy.csv")
PARAMS = Path("data/other/params/strand3.csv")
CANDIDATES = Path("data/nuristani_cognate_candidates.csv")
IA_LANGUAGES = {"OIA", "PAr"}


def numeric_id(value):
    return value and value[0].isdigit()


def main():
    with PARAMS.open(encoding="utf-8") as file:
        stored_rows = list(csv.reader(file))
    stored = [row for row in stored_rows if row[1] == "PNur"]
    stored_ia = defaultdict(list)
    for row in stored_rows:
        if row[1] == "Indo-Aryan":
            stored_ia[(row[2], row[3])].append(row)
    with HIERARCHY.open(encoding="utf-8") as file:
        hierarchy = list(csv.DictReader(file))
    live = [row for row in hierarchy if row["language_id"] == "PNur"]
    assert len(stored) == len(live)
    for old, new in zip(stored, live):
        assert (old[2], old[3]) == (new["form"], new["gloss"]), (old, new)

    by_index = {row["source_index"]: row for row in hierarchy}
    children = defaultdict(list)
    for row in hierarchy:
        children[row["parent_index"]].append(row)

    descendants = {}

    def subtree(row):
        index = row["source_index"]
        if index not in descendants:
            result = []
            todo = list(children[index])
            while todo:
                child = todo.pop()
                result.append(child)
                todo.extend(children[child["source_index"]])
            descendants[index] = result
        return descendants[index]

    output = []
    for old, node in zip(stored, live):
        ancestors = []
        parent_index = node["parent_index"]
        while parent_index:
            parent = by_index[parent_index]
            ancestors.append(parent)
            parent_index = parent["parent_index"]

        direct = None
        for row in ancestors:
            if row["language_id"] in IA_LANGUAGES and numeric_id(row["id"]):
                direct = row
                break
            matches = stored_ia[(row["form"], row["gloss"])]
            if row["language_id"] == "OIA" and len(matches) == 1:
                direct = dict(row)
                direct["id"] = matches[0][0]
                break
        candidates = []
        evidence = ""
        anchor = None
        if direct:
            candidates = [direct]
            evidence = (
                "numeric IA ancestor" if numeric_id(direct["id"])
                else "parsed Strand OIA ancestor"
            )
            anchor = direct
        else:
            for ancestor in ancestors:
                options = {
                    row["id"]: row
                    for row in subtree(ancestor)
                    if row["language_id"] in IA_LANGUAGES and numeric_id(row["id"])
                }
                if len(options) == 1:
                    candidates = list(options.values())
                    evidence = "sole numeric IA node under common ancestor"
                    anchor = ancestor
                    break
                if options:
                    candidates = list(options.values())
                    evidence = "multiple numeric IA nodes under common ancestor"
                    anchor = ancestor
                    break

        if not candidates:
            output.append({
                "Proto_Nuristani_ID": old[0],
                "Proto_Nuristani_Form": old[2],
                "Proto_Nuristani_Gloss": old[3],
                "Indo_Aryan_ID": "",
                "Indo_Aryan_Form": "",
                "Indo_Aryan_Gloss": "",
                "Evidence": "no structural IA candidate",
                "Candidate_Count": 0,
                "Anchor": "",
            })
            continue

        for candidate in candidates:
            output.append({
                "Proto_Nuristani_ID": old[0],
                "Proto_Nuristani_Form": old[2],
                "Proto_Nuristani_Gloss": old[3],
                "Indo_Aryan_ID": candidate["id"],
                "Indo_Aryan_Form": candidate["form"],
                "Indo_Aryan_Gloss": candidate["gloss"],
                "Evidence": evidence,
                "Candidate_Count": len(candidates),
                "Anchor": (
                    f'{anchor["language_id"]} {anchor["form"]} “{anchor["gloss"]}”'
                    if anchor
                    else ""
                ),
            })

    with CANDIDATES.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output[0])
        writer.writeheader()
        writer.writerows(output)

    clear = {row["Proto_Nuristani_ID"] for row in output if row["Candidate_Count"] == 1}
    ambiguous = {row["Proto_Nuristani_ID"] for row in output if row["Candidate_Count"] > 1}
    none = {row["Proto_Nuristani_ID"] for row in output if row["Candidate_Count"] == 0}
    print(
        f"wrote {len(output)} candidate rows: {len(clear)} clear PNur heads, "
        f"{len(ambiguous)} ambiguous, {len(none)} without structural candidates"
    )


if __name__ == "__main__":
    main()
