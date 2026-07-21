"""Rank Indo-Aryan etyma by exact overlap with Proto-Nuristani reflexes."""

import csv
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher


def normalized_form(value):
    return unicodedata.normalize("NFC", value.strip().strip("-"))


def comparable_form(value):
    value = unicodedata.normalize("NFD", value.lower())
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.translate(str.maketrans({
        "ʦ": "ts", "ʣ": "dz", "č": "c", "ǰ": "j", "š": "s", "ž": "z",
        "ṣ": "s", "ṭ": "t", "ḍ": "d", "ṇ": "n", "ṅ": "n", "ñ": "n",
        "ṛ": "r", "ṝ": "r", "ḷ": "l", "ʹ": "", "′": "", "˜": "",
    }))
    return re.sub(r"[^a-z]", "", value)


def gloss_tokens(value):
    return {
        token
        for token in re.findall(r"[a-z]+", value.lower())
        if token not in {"a", "an", "the", "of", "to", "be", "and", "or", "one", "with"}
    }


def main():
    with open("cldf/forms.csv", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    entries = {row["ID"]: row for row in rows if not row["Origin_ID"]}
    signatures = defaultdict(set)
    for row in rows:
        origin = row["Origin_ID"].lstrip(">~")
        if not origin or row["Relation"] not in {"reflex", ""}:
            continue
        form = normalized_form(row["Form"])
        if form:
            signatures[origin].add((row["Language_ID"], form))

    ia_by_signature = defaultdict(set)
    for origin, values in signatures.items():
        entry = entries.get(origin)
        if entry and entry["Language_ID"] == "Indo-Aryan" and origin[0].isdigit():
            for value in values:
                ia_by_signature[value].add(origin)

    with open("data/nuristani_cognate_candidates.csv", encoding="utf-8") as file:
        hierarchy_rows = list(csv.DictReader(file))
    hierarchy = {
        row["Proto_Nuristani_ID"]: row["Indo_Aryan_ID"]
        for row in hierarchy_rows
        if row["Candidate_Count"] == "1"
    }
    with open("data/other/params/strand3.csv", encoding="utf-8") as file:
        stored_pnur = [row for row in csv.reader(file) if row[1] == "PNur"]
    with open("data/strand_hierarchy.csv", encoding="utf-8") as file:
        source_rows = list(csv.DictReader(file))
    live_pnur = [row for row in source_rows if row["language_id"] == "PNur"]
    assert len(stored_pnur) == len(live_pnur)
    source_by_index = {row["source_index"]: row for row in source_rows}
    source_target = {}
    for old, node in zip(stored_pnur, live_pnur):
        assert (old[2], old[3]) == (node["form"], node["gloss"])
        parent_index = node["parent_index"]
        while parent_index:
            parent = source_by_index[parent_index]
            if parent["language_id"] in {"OIA", "PAr"}:
                source_target[old[0]] = parent
                break
            parent_index = parent["parent_index"]

    output = []
    agreement = Counter()
    for entry_id, entry in entries.items():
        if entry["Language_ID"] != "PNur":
            continue
        scores = Counter()
        evidence = defaultdict(list)
        for signature in signatures[entry_id]:
            for candidate in ia_by_signature[signature]:
                scores[candidate] += 1
                evidence[candidate].append(f"{signature[0]}:{signature[1]}")
        ranked = scores.most_common()
        expected = hierarchy.get(entry_id)
        if expected:
            if ranked and ranked[0][0] == expected and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
                agreement["hierarchy agrees with unique top overlap"] += 1
            elif expected in scores:
                agreement["hierarchy candidate appears but is not unique top"] += 1
            elif ranked:
                agreement["hierarchy disagrees with overlap"] += 1
            else:
                agreement["hierarchy has no overlap evidence"] += 1
        for rank, (candidate, score) in enumerate(ranked, 1):
            target = source_target.get(entry_id, {})
            target_form = target.get("form", "")
            candidate_form = entries[candidate]["Form"]
            left, right = comparable_form(target_form), comparable_form(candidate_form)
            form_similarity = SequenceMatcher(None, left, right).ratio() if left and right else 0
            target_gloss = target.get("gloss", "")
            candidate_gloss = entries[candidate]["Gloss"]
            gloss_overlap = gloss_tokens(target_gloss) & gloss_tokens(candidate_gloss)
            output.append({
                "Proto_Nuristani_ID": entry_id,
                "Proto_Nuristani_Form": entry["Form"],
                "Proto_Nuristani_Gloss": entry["Gloss"],
                "Indo_Aryan_ID": candidate,
                "Indo_Aryan_Form": entries[candidate]["Form"],
                "Indo_Aryan_Gloss": entries[candidate]["Gloss"],
                "Overlap": score,
                "Rank": rank,
                "Runner_Up_Overlap": ranked[1][1] if len(ranked) > 1 else 0,
                "Shared_Reflexes": ";".join(sorted(evidence[candidate])),
                "Hierarchy_ID": expected or "",
                "Target_Language": target.get("language_id", ""),
                "Target_Form": target_form,
                "Target_Gloss": target_gloss,
                "Form_Similarity": f"{form_similarity:.3f}",
                "Gloss_Overlap": ";".join(sorted(gloss_overlap)),
            })

    with open("data/nuristani_overlap_candidates.csv", "w", newline="", encoding="utf-8") as file:
        fields = list(output[0])
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)

    unique_top = {
        row["Proto_Nuristani_ID"]
        for row in output
        if row["Rank"] == 1 and int(row["Overlap"]) > int(row["Runner_Up_Overlap"])
    }
    strong = {
        row["Proto_Nuristani_ID"]
        for row in output
        if row["Rank"] == 1 and int(row["Overlap"]) >= 2 and int(row["Overlap"]) > int(row["Runner_Up_Overlap"])
    }
    print(f"candidate rows: {len(output)}; unique top PNur: {len(unique_top)}; overlap >=2: {len(strong)}")
    print(agreement)


if __name__ == "__main__":
    main()
