"""Match Strand's OIA/Proto-Aryan ancestor heads against parsed CDIAL heads."""

import csv
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from analyze_nuristani_overlaps import comparable_form, gloss_tokens


def main():
    with open("cldf/forms.csv", encoding="utf-8") as file:
        entries = {
            row["ID"]: row
            for row in csv.DictReader(file)
            if not row["Origin_ID"]
        }
    cdial = [
        row for row in entries.values()
        if row["Language_ID"] == "Indo-Aryan" and row["ID"][0].isdigit()
    ]
    cdial_by_initial = defaultdict(list)
    for row in cdial:
        form = comparable_form(row["Form"])
        if form:
            cdial_by_initial[form[0]].append((row, form))

    with open("data/other/params/strand3.csv", encoding="utf-8") as file:
        stored = [row for row in csv.reader(file) if row[1] == "PNur"]
    with open("data/strand_hierarchy.csv", encoding="utf-8") as file:
        hierarchy = list(csv.DictReader(file))
    live = [row for row in hierarchy if row["language_id"] == "PNur"]
    assert len(stored) == len(live)
    by_index = {row["source_index"]: row for row in hierarchy}

    with open("data/nuristani_cognate_candidates.csv", encoding="utf-8") as file:
        structural = {
            row["Proto_Nuristani_ID"]: row["Indo_Aryan_ID"]
            for row in csv.DictReader(file)
            if row["Candidate_Count"] == "1"
        }

    output = []
    gold = Counter()
    for old, node in zip(stored, live):
        assert (old[2], old[3]) == (node["form"], node["gloss"])
        target = None
        parent_index = node["parent_index"]
        while parent_index:
            parent = by_index[parent_index]
            if parent["language_id"] in {"OIA", "PAr"}:
                target = parent
                break
            parent_index = parent["parent_index"]
        if not target:
            continue

        target_form = comparable_form(target["form"])
        target_gloss = gloss_tokens(target["gloss"])
        ranked = []
        for candidate, candidate_form in cdial_by_initial.get(target_form[:1], []):
            if abs(len(candidate_form) - len(target_form)) > max(4, len(target_form) // 2):
                continue
            similarity = SequenceMatcher(None, target_form, candidate_form).ratio()
            if similarity < 0.6:
                continue
            overlap = target_gloss & gloss_tokens(candidate["Gloss"])
            score = similarity + (0.20 if overlap else 0) + (0.05 if target_gloss == gloss_tokens(candidate["Gloss"]) else 0)
            ranked.append((score, similarity, len(overlap), candidate, overlap))
        ranked.sort(key=lambda item: (-item[0], -item[1], item[3]["ID"]))
        if not ranked:
            continue
        expected = structural.get(old[0], "")
        top = ranked[0]
        runner_score = ranked[1][0] if len(ranked) > 1 else 0
        if expected and expected[0].isdigit():
            if top[3]["ID"] == expected:
                gold["top agrees"] += 1
            elif any(item[3]["ID"] == expected for item in ranked[:5]):
                gold["expected in top 5"] += 1
            else:
                gold["expected below top 5"] += 1
        output.append({
            "Proto_Nuristani_ID": old[0],
            "Proto_Nuristani_Form": old[2],
            "Indo_Aryan_ID": top[3]["ID"],
            "Indo_Aryan_Form": top[3]["Form"],
            "Indo_Aryan_Gloss": top[3]["Gloss"],
            "Target_Language": target["language_id"],
            "Target_Form": target["form"],
            "Target_Gloss": target["gloss"],
            "Form_Similarity": f"{top[1]:.3f}",
            "Gloss_Overlap": ";".join(sorted(top[4])),
            "Score": f"{top[0]:.3f}",
            "Runner_Up_Score": f"{runner_score:.3f}",
            "Structural_ID": expected,
        })

    with open("data/nuristani_head_candidates.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output[0])
        writer.writeheader()
        writer.writerows(output)
    print(f"wrote {len(output)} head-match candidates")
    print(gold)


if __name__ == "__main__":
    main()
