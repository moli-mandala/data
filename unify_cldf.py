"""
unify_cldf.py — fold the etyma (parameters.csv) and the attested reflexes (forms.csv) into ONE
table: cldf/forms.csv, one row per node in the etymon graph. Reflexes point at their etymon via a
self-referential `Origin_ID`; etyma have an empty `Origin_ID`. parameters.csv is then removed.

Column mapping (uniform Gloss/Description for both kinds):
    etymon : Form=headword, Gloss=full CDIAL entry HTML, Description=Etyma,  Origin_ID=""
    reflex : Form=form,     Gloss=meaning,               Description=notes, Origin_ID=<etymon id>

Run LAST in the data pipeline (after make_cldf.py, align.py, link_refs.py), since those read the
split files.
"""

import csv
import os
import sys

UNIFIED = [
    "ID", "Language_ID", "Form", "Gloss", "Native", "Phonemic", "Original", "Cognateset",
    "Description", "Tags", "Source", "Origin_ID",
]


def main():
    with open("cldf/parameters.csv", encoding="utf-8") as f:
        params = list(csv.DictReader(f))
    with open("cldf/forms.csv", encoding="utf-8") as f:
        forms = list(csv.DictReader(f))

    with open("cldf/forms.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(UNIFIED)
        for p in params:  # etyma
            w.writerow([
                p["ID"], p["Language_ID"], p["Name"], p["Description"],
                "", "", "", "", p.get("Etyma", ""), "", "", "",
            ])
        for r in forms:  # reflexes
            w.writerow([
                r["ID"], r["Language_ID"], r["Form"], r["Gloss"], r["Native"], r["Phonemic"],
                r["Original"], r["Cognateset"], r["Description"], r.get("Tags", ""), r["Source"],
                r["Parameter_ID"],
            ])

    os.remove("cldf/parameters.csv")
    print(
        f"unified cldf/forms.csv: {len(params)} etyma + {len(forms)} reflexes "
        f"= {len(params) + len(forms)} nodes; removed parameters.csv",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
