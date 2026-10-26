"""Locate source-drawn lowlines on all five Malaveda/Veda chapter pages."""
import hashlib
import json
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
PDF = next(p for p in HERE.parents if (p / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf").exists()) / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf"
EXPECTED = "4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af"
assert hashlib.sha256(PDF.read_bytes()).hexdigest() == EXPECTED
document = fitz.open(PDF)
results = []
for number in range(40, 45):
    page = document[number - 1]
    lines = [[c for span in line["spans"] for c in span["chars"]]
             for block in page.get_text("rawdict")["blocks"] for line in block.get("lines", [])]
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if not (rect.width < 8 and rect.y0 < 720 and rect.height < 1):
            continue
        cx = (rect.x0 + rect.x1) / 2
        candidates = []
        for chars in lines:
            for i, char in enumerate(chars):
                bbox = fitz.Rect(char["bbox"])
                if bbox.x0 - 0.8 <= cx <= bbox.x1 + 0.8 and abs(rect.y0 - bbox.y1) <= 12:
                    candidates.append((round(abs(rect.y0 - bbox.y1), 2), char["c"],
                                       "".join(c["c"] for c in chars), i, [round(v, 2) for v in bbox]))
        candidates.sort()
        results.append({"pdf_page": number, "rect": [round(v, 2) for v in rect], "candidate_characters": candidates[:3]})
(HERE / "underline-vector-census.json").write_text(json.dumps({"source_pdf_sha256": EXPECTED,
    "status": "complete chapter geometry census; visual/key mapping pending", "drawn_low_lines": len(results),
    "records": results}, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"drawn_low_lines": len(results), "identified": sum(bool(r["candidate_characters"]) for r in results)}))
