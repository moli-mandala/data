"""Reproducible OCR evidence, not an installed wordlist or a database build.

Run with an explicit cached PDF and scratch output directory. Each column is
recognized independently; word coordinates retain their original page frame.
No OCR line is silently classified as a lexical record.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

import pypdfium2 as pdfium

PDF_SHA256 = "f022a2a448445cdc9840891478bd5864e2ded1ae98985332b51f0938c075842d"
PAGES = range(21, 49)
SCALE = 4
# Normalized page coordinates; preserve headings and footers in the raw ledger.
COLUMNS = {"chakma": (0.08, 0.44, "ben"),
           "bengali": (0.44, 0.68, "ben"),
           "english": (0.68, 0.98, "eng")}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_lines(tsv, x_offset):
    groups = {}
    for word in csv.DictReader(io.StringIO(tsv), delimiter="\t"):
        if word["level"] != "5" or not word["text"].strip():
            continue
        key = tuple(int(word[k]) for k in ("block_num", "par_num", "line_num"))
        groups.setdefault(key, []).append({
            "text": word["text"], "confidence": float(word["conf"]),
            "left": int(word["left"]) + x_offset, "top": int(word["top"]),
            "width": int(word["width"]), "height": int(word["height"]),
        })
    result = []
    for key, words in groups.items():
        left = min(w["left"] for w in words)
        top = min(w["top"] for w in words)
        right = max(w["left"] + w["width"] for w in words)
        bottom = max(w["top"] + w["height"] for w in words)
        result.append({"ocr_line": list(key), "raw": " ".join(w["text"] for w in words),
                       "bbox": [left, top, right, bottom], "words": words,
                       "status": "unreviewed", "review_type": "ocr-and-structure"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path,
                        help="JSON mapping PDF pages to two column boundaries")
    parser.add_argument("--bengali-model-dir", type=Path,
                        help="Optional local tessdata directory for the Bengali passes")
    args = parser.parse_args()
    boundaries = json.loads(args.boundaries.read_text()) if args.boundaries else {}
    if sha256(args.pdf) != PDF_SHA256:
        raise SystemExit("Source PDF differs from the pinned scan")
    args.output.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(args.pdf)
    if len(document) != 110:
        raise SystemExit("Expected 110 PDF pages")
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    version = subprocess.check_output(["tesseract", "--version"], text=True).splitlines()[0]
    ledger = []
    inventory = []
    for number in PAGES:
        page = document[number - 1]
        bitmap = page.render(scale=SCALE)
        image = bitmap.to_pil().copy()
        bitmap.close()
        page.close()
        image.save(args.output / f"p{number:03}.png")
        counts = {}
        columns = COLUMNS
        if str(number) in boundaries:
            first, second = boundaries[str(number)]
            if not 0.08 < first < second < 0.98:
                raise ValueError(f"Invalid boundaries on PDF page {number}")
            columns = {"chakma": (0.08, first, "ben"),
                       "bengali": (first, second, "ben"),
                       "english": (second, 0.98, "eng")}
        for column, (start, end, language) in columns.items():
            left, right = int(image.width * start), int(image.width * end)
            stem = args.output / f"p{number:03}-{column}"
            image.crop((left, 0, right, image.height)).save(stem.with_suffix(".png"))
            model_options = (["--tessdata-dir", str(args.bengali_model_dir), "--oem", "1"]
                             if language == "ben" and args.bengali_model_dir else [])
            if language == "ben" and args.bengali_model_dir:
                # This model embeds an automatic English sublanguage; disable
                # it for explicitly Bengali-script cells, or it emits Latin noise.
                model_options += ["-c", "tessedit_load_sublangs="]
            subprocess.run(["tesseract", str(stem.with_suffix(".png")), str(stem),
                            "--psm", "6", "-l", language, *model_options,
                            "-c", "tessedit_create_tsv=1", "-c", "tessedit_create_txt=1"],
                           check=True, env=env, capture_output=True, text=True)
            lines = read_lines(stem.with_suffix(".tsv").read_text(), left)
            for line in lines:
                ledger.append({"pdf_page": number, "printed_page_candidate": number - 2,
                               "column": column, **line})
            counts[column] = len(lines)
        inventory.append({"pdf_page": number, "size": list(image.size),
                          "columns": columns,
                          "render_sha256": sha256(args.output / f"p{number:03}.png"),
                          "raw_line_counts": counts})
        print(f"PDF {number}: {counts}", flush=True)
    document.close()
    (args.output / "raw-lines.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in ledger))
    manifest = {"status": "raw OCR only; not installed", "pdf_sha256": PDF_SHA256,
                "pdf_pages": 110, "render_scale": SCALE, "columns": COLUMNS,
                "tesseract": version, "psm": 6, "threads": 1,
                "boundary_overrides": boundaries,
                "bengali_model_sha256": (sha256(args.bengali_model_dir / "ben.traineddata")
                                          if args.bengali_model_dir else None),
                "bengali_sublanguages": "disabled" if args.bengali_model_dir else "model default",
                "pagination": "Printed page = PDF page - 2 remains subject to visual verification",
                "raw_lines": len(ledger), "pages": inventory}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
