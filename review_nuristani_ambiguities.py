"""Generate a compact report for manual Proto-Nuristani cognate review."""

import csv
import sys
from difflib import SequenceMatcher

from analyze_nuristani_overlaps import comparable_form, gloss_tokens


def read(path):
    with open(path, encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    ambiguous = {
        row["Proto_Nuristani_ID"]
        for row in read("data/nuristani_cognates_uncertain.csv")
        if row["Reason"].startswith("Strand shows inherited ancestry")
    }
    candidates = [
        row
        for row in read("data/nuristani_head_candidates.csv")
        if row["Proto_Nuristani_ID"] in ambiguous
    ]
    candidates.sort(key=lambda row: int(row["Proto_Nuristani_ID"][1:]))
    requested = set(sys.argv[1:])
    if requested:
        entries = [
            row
            for row in read("cldf/forms.csv")
            if row["Language_ID"] == "Indo-Aryan"
            and not row["Origin_ID"]
            and row["ID"][0].isdigit()
        ]
        for row in candidates:
            if row["Proto_Nuristani_ID"] not in requested:
                continue
            target_form = comparable_form(row["Target_Form"])
            target_gloss = gloss_tokens(row["Target_Gloss"])
            ranked = []
            for entry in entries:
                entry_form = comparable_form(entry["Form"])
                similarity = SequenceMatcher(None, target_form, entry_form).ratio()
                overlap = target_gloss & gloss_tokens(entry["Gloss"])
                score = similarity + min(0.6, 0.2 * len(overlap))
                if similarity >= 0.45 or overlap:
                    ranked.append((score, similarity, len(overlap), entry))
            ranked.sort(key=lambda item: (-item[0], -item[2], -item[1], item[3]["ID"]))
            print(
                f'\n{row["Proto_Nuristani_ID"]} {row["Proto_Nuristani_Form"]}'
                f' < {row["Target_Form"]} [{row["Target_Gloss"]}]'
            )
            for score, similarity, _, entry in ranked[:12]:
                print(
                    f'  {entry["ID"]}\t{entry["Form"]}\t{entry["Gloss"][:100]}'
                    f'\tform={similarity:.3f}\tscore={score:.3f}'
                )
        return
    for row in candidates:
        print(
            f'{row["Proto_Nuristani_ID"]}\t{row["Proto_Nuristani_Form"]}'
            f'\t{row["Target_Form"]}\t{row["Target_Gloss"]}'
            f'\t{row["Indo_Aryan_ID"]}\t{row["Indo_Aryan_Form"]}'
            f'\t{row["Indo_Aryan_Gloss"][:100]}'
            f'\t{row["Form_Similarity"]}\t{row["Gloss_Overlap"]}'
        )


if __name__ == "__main__":
    main()
