"""Render target-cell review sheets from the pinned source; no OCR or builds."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import pdfplumber
import pypdfium2
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first", type=int, default=21)
    parser.add_argument("--last", type=int, default=50)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "source-manifest.json").read_text())
    if hashlib.sha256(args.pdf.read_bytes()).hexdigest() != manifest["source_pdf_sha256"]:
        raise ValueError("Pinned PDF checksum mismatch")
    with (ROOT / "source-cells.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = []
    doc = pypdfium2.PdfDocument(args.pdf)
    with pdfplumber.open(args.pdf) as textdoc:
        for number in range(args.first, args.last + 1):
            page = textdoc.pages[number - 1]
            words = page.extract_words()
            selected = []
            for column in (1, 2):
                left, right = (90, 303) if column == 1 else (338, 554)
                for code, label in (("GUT", "Tikrapada"), ("PAR", "Kinumun")):
                    # PDF37's no-response row omits spaces in its site label.
                    labels = {label, "KinumunParengaParja"} if code == "PAR" else {label}
                    anchors = sorted((w for w in words if w["text"] in labels and left <= w["x0"] <= right), key=lambda w: w["top"])
                    cells = sorted((r for r in rows if int(r["PDF_Page"]) == number and int(r["Column"]) == column and r["Site_Code"] == code and r["Extraction_Status"] != "disqualified"), key=lambda r: int(r["Item"]))
                    if len(anchors) != len(cells):
                        raise ValueError(f"Anchor count mismatch: {number}, {column}, {code}")
                    for cell, anchor in zip(cells, anchors):
                        following = [w["top"] for w in words if left <= w["x0"] < left + 20 and w["top"] > anchor["top"] + 5]
                        bottom = min(following) - 1 if following else anchor["bottom"] + 4
                        selected.append((cell, [left, anchor["top"] - 3, right, bottom]))
            selected.sort(key=lambda pair: (int(pair[0]["Item"]), pair[0]["Site_Code"]))
            rendered = doc[number - 1].render(scale=3).to_pil()
            crops = [(cell, box, rendered.crop(tuple(round(v * 3) for v in box))) for cell, box in selected]
            sheet = Image.new("RGB", (700, sum(crop.height + 30 for _, _, crop in crops)), "white")
            draw, y = ImageDraw.Draw(sheet), 0
            for cell, box, crop in crops:
                draw.text((5, y + 3), f"Item {cell['Item']} {cell['Site_Code']} {cell['Gloss']}", fill="black")
                sheet.paste(crop, (0, y + 25))
                evidence.append({"item": int(cell["Item"]), "site_code": cell["Site_Code"], "pdf_page": number, "bbox_points": box, "sheet": f"p{number:03d}-targets.png"})
                y += crop.height + 30
            sheet.save(args.output / f"p{number:03d}-targets.png")
    (args.output / f"crop-locators-{args.first}-{args.last}.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
