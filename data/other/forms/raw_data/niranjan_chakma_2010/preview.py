"""Pair OCR cells for review only. Never emits installed forms or stable IDs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = [json.loads(line) for line in (ROOT / "raw-lines.jsonl").read_text().splitlines()]
    pages = defaultdict(lambda: defaultdict(list))
    for line in raw:
        pages[line["pdf_page"]][line["column"]].append(line)
    review = json.loads((ROOT / "structure-review.json").read_text())
    candidates, unpaired = [], []
    for plan in review["pages"]:
        number = plan["pdf_page"]
        columns = pages[number]
        joins = {group[0]: group[1:] for group in plan["join_english_lines"]}
        continuations = {index for group in plan["join_english_lines"] for index in group[1:]}
        excluded = set(plan["exclude_english_lines"])
        used = {column: set() for column in ("chakma", "bengali")}
        local = []
        groups = []
        for index, english in enumerate(columns["english"], 1):
            if index in excluded or index in continuations:
                continue
            anchors = [english] + [columns["english"][i - 1] for i in joins.get(index, [])]
            groups.append((anchors, [index] + joins.get(index, []), None))
        for extra in plan.get("recovered_english_cells", []):
            groups.append(([extra], [], extra["recovery_id"]))
        groups.sort(key=lambda group: group[0][0]["bbox"][1])
        for anchors, line_indices, recovery_id in groups:
            english = anchors[0]
            center = (english["bbox"][1] + english["bbox"][3]) / 2
            record = {"pdf_page": number, "printed_page": plan["printed_page"],
                      "candidate_row": len(local) + 1,
                      "english_line_indices": line_indices,
                      "english_raw": " ".join(line["raw"] for line in anchors),
                      "english_bboxes": [line["bbox"] for line in anchors],
                      "status": "unreviewed", "review_type": ["ocr", "cell-alignment"],
                      "cells": {}}
            if recovery_id is not None:
                record["english_recovery_id"] = recovery_id
            for column in ("chakma", "bengali"):
                matches = []
                for j, line in enumerate(columns[column]):
                    other_center = (line["bbox"][1] + line["bbox"][3]) / 2
                    if abs(center - other_center) <= 24:
                        matches.append({"ocr_line_index": j + 1, "raw": line["raw"],
                                        "bbox": line["bbox"]})
                        used[column].add(j)
                record["cells"][column] = matches
                if len(matches) != 1:
                    record["review_type"].append(f"{column}-cell-count:{len(matches)}")
            local.append(record)
        if len(local) != plan["expected_candidate_records"]:
            raise ValueError(f"PDF {number}: candidate count changed")
        candidates.extend(local)
        for column in ("chakma", "bengali"):
            for index, line in enumerate(columns[column]):
                if index not in used[column]:
                    unpaired.append({"pdf_page": number, "column": column,
                                     "ocr_line_index": index + 1, **line})
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("candidate-cells.jsonl", candidates), ("unpaired-lines.jsonl", unpaired)):
        (args.output / filename).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(json.dumps({"candidate_records": len(candidates), "unpaired_native_lines": len(unpaired),
                      "irregular_cell_counts": sum(len(r["review_type"]) > 2 for r in candidates),
                      "installed_rows": 0}))


if __name__ == "__main__":
    main()
