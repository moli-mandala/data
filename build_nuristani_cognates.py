"""Materialize the reviewed high-confidence Proto-Nuristani cognate set."""

import csv


MANUAL_OVERRIDES = {
    "n70": "8218",
    "n74": "8236",
    "n171": "8056",
    "n325": "9235",
    "n326": "9235",
    "n450": "9529",
    "n705": "558",
    "n854": "11616",
    "n1375": "6216",
    "n1390": "12897",
    "n1474": "14621",
    "n1514": "6724",
    "n1562": "6640",
    "n1678": "13139",
    "n1735": "12812",
    "n1736": "12812",
    "n1803": "5041",
    "n1806": "5039",
    "n1876": "12655",
    "n1948": "9028-3",
    "n2147": "249",
    "n2337": "11107",
    "n2488": "2485",
    "n2489": "2485",
    "n2490": "2485",
    "n2723": "12762",
    "n2724": "12762",
    "n2776": "14152",
    "n2777": "14152",
    "n2780": "14152",
    "n2859": "12703",
    "n2871": "12729",
    "n2929": "13978",
    "n2935": "14107",
    "n2940": "14024",
    "n2949": "13981",
    "n2952": "13992",
    "n2956": "13982",
    "n3085": "3539",
    "n3088": "3539",
    "n3330": "14185",
    "n3331": "14185",
    "n3381": "11589",
    "n3438": "2457",
    "n3606": "14360",
    "n3607": "14360",
    "n3608": "14360",
    "n3617": "4605",
    "n3618": "4605",
    "n3619": "4605",
    "n3626": "4600",
    "n3630": "4715",
    "n3633": "4714",
    "n3639": "2905",
    "n3701": "5227",
    "n3705": "5241",
    "n3739": "1044",
    "n3744": "1045",
}

MANUAL_REJECTED = {
    "n104", "n541", "n544", "n547", "n550", "n659", "n846", "n850",
    "n996", "n1074", "n1077", "n1108", "n1133", "n1136", "n1148",
    "n1519", "n1567", "n1568", "n1571", "n1572", "n1595", "n1600",
    "n1621", "n1691", "n1705", "n1800", "n1967", "n2111", "n2116",
    "n2117", "n2118", "n2123", "n2128", "n2131", "n2132", "n2134",
    "n2137", "n2140", "n2154", "n2157", "n2305", "n2306", "n2329",
    "n2422", "n2485", "n2497", "n2500", "n2503", "n2508", "n2513",
    "n2580", "n2581", "n2669", "n2772", "n2851", "n2875", "n2927",
    "n2967", "n3117", "n3227", "n3304", "n3320", "n3321", "n3512",
    "n3513", "n3516", "n3519", "n3665", "n3741", "n3818",
}


def read(path):
    with open(path, encoding="utf-8") as file:
        return list(csv.DictReader(file))


def strand_parent_languages():
    with open("data/other/params/strand3.csv", encoding="utf-8") as file:
        stored = [row for row in csv.reader(file) if row[1] == "PNur"]
    hierarchy = read("data/strand_hierarchy.csv")
    live = [row for row in hierarchy if row["language_id"] == "PNur"]
    by_index = {row["source_index"]: row for row in hierarchy}
    assert len(stored) == len(live)
    result = {}
    for old, node in zip(stored, live):
        assert (old[2], old[3]) == (node["form"], node["gloss"])
        parent_index = node["parent_index"]
        while parent_index:
            parent = by_index[parent_index]
            if parent["language_id"] != "PNur":
                result[old[0]] = parent["language_id"]
                break
            parent_index = parent["parent_index"]
    return result


def main():
    selected = {}

    for row in read("data/nuristani_cognate_candidates.csv"):
        if row["Candidate_Count"] == "1":
            selected[row["Proto_Nuristani_ID"]] = (
                row["Indo_Aryan_ID"],
                row["Evidence"],
            )

    for row in read("data/nuristani_overlap_candidates.csv"):
        if row["Proto_Nuristani_ID"] in selected or row["Rank"] != "1":
            continue
        unique_top = int(row["Overlap"]) > int(row["Runner_Up_Overlap"])
        supported = (
            row["Target_Form"]
            and (
                (row["Gloss_Overlap"] and float(row["Form_Similarity"]) >= 0.5)
                or float(row["Form_Similarity"]) >= 0.8
            )
        )
        if unique_top and supported:
            selected[row["Proto_Nuristani_ID"]] = (
                row["Indo_Aryan_ID"],
                "exact duplicated reflex + matching Strand ancestor",
            )

    for row in read("data/nuristani_head_candidates.csv"):
        if row["Proto_Nuristani_ID"] in selected:
            continue
        exact_head = float(row["Form_Similarity"]) == 1
        clear_margin = float(row["Score"]) - float(row["Runner_Up_Score"]) >= 0.1
        if exact_head and row["Gloss_Overlap"] and clear_margin:
            selected[row["Proto_Nuristani_ID"]] = (
                row["Indo_Aryan_ID"],
                "exact Strand ancestor headword + matching gloss",
            )

    for row in read("data/nuristani_head_candidates.csv"):
        pnur = row["Proto_Nuristani_ID"]
        if pnur in selected or pnur in MANUAL_REJECTED:
            continue
        selected[pnur] = (
            MANUAL_OVERRIDES.get(pnur, row["Indo_Aryan_ID"]),
            "manually reviewed Strand ancestor and CDIAL head",
        )

    with open("cldf/forms.csv", encoding="utf-8") as file:
        entries = {row["ID"]: row for row in csv.DictReader(file)}
    with open("cldf/merges.csv", encoding="utf-8") as file:
        merges = {row["Addendum_ID"]: row["Main_ID"] for row in csv.DictReader(file)}

    def canonical_ia(entry_id):
        entry_id = merges.get(entry_id, entry_id)
        if entry_id in entries:
            return entry_id
        section_id = entry_id.replace(".", "-")
        if section_id in entries:
            return merges.get(section_id, section_id)
        base_id = entry_id.split(".")[0]
        if base_id in entries:
            return merges.get(base_id, base_id)
        return entry_id

    selected = {
        pnur: (canonical_ia(ia), evidence)
        for pnur, (ia, evidence) in selected.items()
    }
    missing = sorted((pnur, ia) for pnur, (ia, _) in selected.items() if pnur not in entries or ia not in entries)
    if missing:
        raise ValueError(f"selected unknown entries: {missing}")
    wrong_languages = sorted(
        (pnur, entries[pnur]["Language_ID"], ia, entries[ia]["Language_ID"])
        for pnur, (ia, _) in selected.items()
        if entries[pnur]["Language_ID"] != "PNur" or entries[ia]["Language_ID"] != "Indo-Aryan"
    )
    if wrong_languages:
        raise ValueError(f"selected wrong-language entries: {wrong_languages}")

    parent_languages = strand_parent_languages()
    inherited = {}
    strand_borrowings = {}
    rejected_borrowings = []
    for pnur, (ia, evidence) in selected.items():
        if parent_languages.get(pnur) == "OIA":
            strand_borrowings[pnur] = (ia, evidence)
            continue
        if parent_languages.get(pnur) != "PAr":
            continue
        base = ia.split("-")[0]
        ia_row = entries[ia]
        base_row = entries.get(base, ia_row)
        if ia_row["Relation"] == "borrowed" or base_row["Relation"] == "borrowed":
            rejected_borrowings.append((pnur, ia))
        else:
            inherited[pnur] = (ia, evidence)
    selected = inherited

    rows = [
        {
            "Ancestor_ID": f"pii-{ia}",
            "Proto_Nuristani_ID": pnur,
            "Indo_Aryan_ID": ia,
            "Evidence": evidence,
        }
        for pnur, (ia, evidence) in sorted(
            selected.items(), key=lambda item: int(item[0][1:])
        )
    ]
    with open("data/nuristani_cognates.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    borrowing_rows = [
        {
            "Proto_Nuristani_ID": pnur,
            "Indo_Aryan_ID": ia,
            "Evidence": f"Strand places PNur beneath OIA; {evidence}",
        }
        for pnur, (ia, evidence) in sorted(
            strand_borrowings.items(), key=lambda item: int(item[0][1:])
        )
    ]
    with open("data/nuristani_borrowings.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=borrowing_rows[0])
        writer.writeheader()
        writer.writerows(borrowing_rows)

    mapped = {row["Proto_Nuristani_ID"] for row in rows} | set(strand_borrowings)
    rejected = dict(rejected_borrowings)
    head_candidates = {row["Proto_Nuristani_ID"]: row for row in read("data/nuristani_head_candidates.csv")}
    overlap_candidates = {
        row["Proto_Nuristani_ID"]: row
        for row in read("data/nuristani_overlap_candidates.csv")
        if row["Rank"] == "1"
    }
    with open("data/other/params/strand3.csv", encoding="utf-8") as file:
        pnur_rows = [row for row in csv.reader(file) if row[1] == "PNur"]
    uncertain = []
    for pnur in pnur_rows:
        if pnur[0] in mapped:
            continue
        head = head_candidates.get(pnur[0], {})
        overlap = overlap_candidates.get(pnur[0], {})
        candidate = overlap or head
        if pnur[0] in rejected:
            reason = "IA candidate is marked borrowed, so no inherited Proto-II link was added"
        elif pnur[0] in MANUAL_REJECTED:
            reason = "manual review found no sufficiently clear CDIAL match"
        elif head:
            reason = "Strand shows inherited ancestry, but the parsed IA match is not clear enough"
        else:
            reason = "no OIA or Proto-Aryan ancestor is given in Strand's hierarchy"
        uncertain.append({
            "Proto_Nuristani_ID": pnur[0],
            "Proto_Nuristani_Form": pnur[2],
            "Proto_Nuristani_Gloss": pnur[3],
            "Strand_Ancestor": " ".join(
                value for value in (head.get("Target_Form", ""), head.get("Target_Gloss", "")) if value
            ),
            "Best_IA_ID": rejected.get(pnur[0], candidate.get("Indo_Aryan_ID", "")),
            "Best_IA_Form": candidate.get("Indo_Aryan_Form", ""),
            "Reason": reason,
        })
    with open("data/nuristani_cognates_uncertain.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=uncertain[0])
        writer.writeheader()
        writer.writerows(uncertain)

    evidence_counts = {}
    for row in rows:
        evidence_counts[row["Evidence"]] = evidence_counts.get(row["Evidence"], 0) + 1
    print(
        f"wrote {len(rows)} cognate mappings across "
        f"{len({row['Ancestor_ID'] for row in rows})} Proto-II ancestors: {evidence_counts}; "
        f"wrote {len(borrowing_rows)} Strand OIA borrowings; "
        f"rejected {len(rejected_borrowings)} IA borrowings; {len(uncertain)} uncertain/unmatched"
    )


if __name__ == "__main__":
    main()
