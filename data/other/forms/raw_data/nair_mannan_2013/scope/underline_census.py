"""Record every short source-drawn lowline, without silently applying it."""
import hashlib
import json
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "data/SOURCE_INGESTION_CHECKLIST.md").exists())
PDF = ROOT / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf"
EXPECTED = "4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af"
assert hashlib.sha256(PDF.read_bytes()).hexdigest() == EXPECTED
records = []
with fitz.open(PDF) as document:
    for number in range(45, 50):
        page = document[number - 1]
        lines = [[c for s in line["spans"] for c in s["chars"]]
                 for b in page.get_text("rawdict")["blocks"] for line in b.get("lines", [])]
        for drawing in page.get_drawings():
            rect = drawing["rect"]
            if not (rect.width < 18 and rect.y0 < 720 and rect.height < 1):
                continue
            center = (rect.x0 + rect.x1) / 2
            candidates = []
            for chars in lines:
                for i, char in enumerate(chars):
                    bbox = fitz.Rect(char["bbox"])
                    if bbox.x0 - .8 <= center <= bbox.x1 + .8 and abs(rect.y0 - bbox.y1) <= 12:
                        candidates.append((round(abs(rect.y0 - bbox.y1), 2), char["c"],
                            "".join(c["c"] for c in chars), i, [round(v, 2) for v in bbox]))
            candidates.sort()
            records.append({"printed_page": number, "pdf_rect": list(rect), "candidates": candidates[:3]})
(HERE / "underline-vector-census.json").write_text(json.dumps({
    "source_pdf_sha256": EXPECTED, "count": len(records), "records": records,
    "status": "original geometry evidence; all character attachments require visual review",
}, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"drawn_lowlines": len(records), "identified": sum(bool(r["candidates"]) for r in records)}))
