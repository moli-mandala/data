"""Separate opposite-column OCR by original word coordinates, without accepting text."""
import json
from pathlib import Path
from record_brokskat_first_reading import sha

ROOT = Path(__file__).resolve().parent
# Vertical dividers read from every original page; coordinates at the pinned 3x OCR render.
DIVIDERS = {79: 535, 80: 403, 81: 498, 82: 508, 83: 515, 84: 498,
            85: 479, 86: 538, 87: 528, 88: 503, 89: 460}


def main():
    raw_path = ROOT / "whole-remaining-raw-lines.jsonl"
    pages = [r for r in map(json.loads, raw_path.read_text().splitlines()) if r["pdf_page"] in DIVIDERS]
    assert [r["pdf_page"] for r in pages] == list(range(79, 90))
    output, headers, excluded = [], [], []
    assigned_words = 0
    for page in pages:
        number = page["pdf_page"]
        cols = {"left": [], "right": []}
        for line_number, line in enumerate(page["lines"], 1):
            if line["bbox"][1] >= 1585 or (number == 79 and line["bbox"][1] < 250):
                excluded.append({"pdf_page": number, "raw_line": line_number,
                                 "reason": "printed footer" if line["bbox"][1] >= 1585 else "chapter heading",
                                 "evidence": line})
                continue
            parts = {"left": [], "right": []}
            for word_number, word in enumerate(line["words"], 1):
                center = word["left"] + word["width"] / 2
                side = "left" if center < DIVIDERS[number] else "right"
                parts[side].append(dict(word, raw_word_number=word_number))
                assigned_words += 1
            for side, words in parts.items():
                if not words:
                    continue
                words.sort(key=lambda w: w["left"])
                bbox = [min(w["left"] for w in words), min(w["top"] for w in words),
                        max(w["left"] + w["width"] for w in words),
                        max(w["top"] + w["height"] for w in words)]
                cols[side].append({"raw_line_number": line_number, "raw_ocr_line_id": line["ocr_line"],
                                   "raw": " ".join(w["text"] for w in words), "bbox": bbox,
                                   "words": words, "original_unsplit_raw": line["raw"],
                                   "opposite_column_split": bool(parts["left"] and parts["right"])})
        for side, lines in cols.items():
            lines.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
            left = min(r["bbox"][0] for r in lines)
            for n, line in enumerate(lines, 1):
                line["column_line_number"] = n
                if line["bbox"][0] < left + 40:
                    headers.append({"page": number, "column": side, "line": n,
                                    "y": line["bbox"][1], "raw": line["raw"]})
            output.append({"pdf_page": number, "printed_page": number-2, "column": side,
                           "divider_x": DIVIDERS[number], "lines": lines,
                           "raw_tsv_sha256": page["raw_tsv_sha256"],
                           "status": "coordinate-separated raw OCR; no transcription accepted"})
    assert len(output) == 22
    assert sum(len(line["words"]) for col in output for line in col["lines"]) == assigned_words
    target = ROOT / "numbered-column-raw-evidence.jsonl"
    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output))
    (ROOT / "numbered-column-layout-review.json").write_text(json.dumps({
        "status": "column separation complete; group-boundary review pending",
        "input_sha256": sha(raw_path), "output_sha256": sha(target),
        "columns": 22, "assigned_words": assigned_words,
        "lines_split_between_columns": sum(line["opposite_column_split"] for col in output for line in col["lines"]) // 2,
        "excluded_nonlexical_lines": excluded, "candidate_boundaries": headers,
    }, ensure_ascii=False, indent=2) + "\n")
    for record in headers:
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
