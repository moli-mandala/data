"""Review unlinked Schmidt forms against linked, non-Schmidt Shinaic forms.

Run after building the unified CLDF.  The default action regenerates an
editable review CSV while preserving its Decision and Notes columns.  Mark
accepted rows ``yes`` and rerun with ``--apply`` to update the Schmidt source
CSV.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from link_northern_shina_database import distance, normalized_form, root_id, senses


HERE = Path(__file__).resolve().parent
DATA = HERE.parents[3]
RAW_FORMS = HERE.parent / "20230621-shina.csv"
FORMS = DATA / "cldf/forms.csv"
LANGUAGES = DATA / "cldf/languages.csv"
REVIEW = HERE / "schmidt_shina_database_etymologies.csv"

# Schmidt c̣/c represents the Shina affricate written ʦ̣ in CDIAL.  Without
# that source-specific equivalence, the broad edit metric incorrectly prefers
# Savi pito 'bitter' under 8182 to the direct Shina ʦ̣iṭu under 5806.
PARAMETER_OVERRIDES = {986: "5806", 987: "5806", 989: "5806"}


def generate_review() -> None:
    with FORMS.open(encoding="utf-8") as handle:
        form_rows = list(csv.DictReader(handle))
    forms = {row["ID"]: row for row in form_rows}
    with LANGUAGES.open(encoding="utf-8") as handle:
        clades = {row["ID"]: row["Clade"] for row in csv.DictReader(handle)}

    evidence_by_sense: dict[str, list[tuple[str, str, str, str, str]]] = defaultdict(list)
    for row in form_rows:
        if clades.get(row["Language_ID"]) != "Shinaic" or not row["Origin_ID"]:
            continue
        sources = {source for source in row["Source"].split(";") if source}
        if sources == {"schmidt"}:
            continue
        rid = root_id(row, forms)
        spellings = {normalized_form(row["Form"])}
        if row["Phonemic"]:
            spellings.add(normalized_form(row["Phonemic"]))
        for sense in senses(";".join((row["Gloss"], row["Description"]))):
            for spelling in spellings:
                evidence_by_sense[sense].append(
                    (rid, spelling, row["Language_ID"], row["Form"], ";".join(sorted(sources)))
                )

    previous = {}
    if REVIEW.exists():
        with REVIEW.open(encoding="utf-8") as handle:
            previous = {row["Row"]: row for row in csv.DictReader(handle)}

    with RAW_FORMS.open(encoding="utf-8") as handle:
        raw_rows = list(csv.reader(handle))

    output = []
    for row_number, row in enumerate(raw_rows, 1):
        if (row[1] and str(row_number) not in previous) or clades.get(row[0]) != "Shinaic":
            continue
        target = normalized_form(row[5] or row[2])
        candidates = [
            item
            for sense in senses(row[3])
            for item in evidence_by_sense.get(sense, [])
        ]
        if not target or not candidates:
            continue

        best_by_root = {}
        for rid, candidate, evidence_language, evidence_form, source in candidates:
            score = distance(target, candidate) / max(len(target), len(candidate), 1)
            if rid not in best_by_root or score < best_by_root[rid][0]:
                best_by_root[rid] = (score, evidence_language, evidence_form, source)
        ranked = sorted(
            (score, rid, language, form, source)
            for rid, (score, language, form, source) in best_by_root.items()
        )
        best_score, rid, evidence_language, evidence_form, source = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 1.0
        if best_score > 0.34 or second_score - best_score < 0.12:
            continue
        if row_number in PARAMETER_OVERRIDES:
            rid = PARAMETER_OVERRIDES[row_number]
            best_score, _, evidence_language, evidence_form, source = next(
                candidate for candidate in ranked if candidate[1] == rid
            )

        old = previous.get(str(row_number), {})
        output.append(
            {
                "Row": row_number,
                "Language_ID": row[0],
                "Form": row[2],
                "Gloss": row[3],
                "Phonemic": row[5],
                "Parameter_ID": rid,
                "Distance": f"{best_score:.3f}",
                "Evidence_Language_ID": evidence_language,
                "Evidence_Form": evidence_form,
                "Evidence_Source": source,
                "Decision": old.get("Decision", ""),
                "Notes": old.get("Notes", ""),
            }
        )

    fields = [
        "Row", "Language_ID", "Form", "Gloss", "Phonemic", "Parameter_ID",
        "Distance", "Evidence_Language_ID", "Evidence_Form", "Evidence_Source",
        "Decision", "Notes",
    ]
    with REVIEW.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    print(f"Wrote {len(output)} Schmidt candidates to {REVIEW}")


def apply_review() -> None:
    with REVIEW.open(encoding="utf-8") as handle:
        accepted = {
            int(row["Row"]): row["Parameter_ID"]
            for row in csv.DictReader(handle)
            if row["Decision"].strip().lower() in {"yes", "y", "accept", "accepted"}
        }
    with RAW_FORMS.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    changed = 0
    for row_number, parameter_id in accepted.items():
        row = rows[row_number - 1]
        if not row[1]:
            row[1] = parameter_id
            changed += 1
        elif row[1] != parameter_id:
            raise ValueError(f"Row {row_number} already links to {row[1]}, not {parameter_id}")
    with RAW_FORMS.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    print(f"Applied {changed} accepted Schmidt assignments to {RAW_FORMS}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    apply_review() if args.apply else generate_review()


if __name__ == "__main__":
    main()
