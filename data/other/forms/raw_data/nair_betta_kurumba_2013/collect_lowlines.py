"""Reproducibly census the source's separately drawn short black low lines."""
import hashlib
import json
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
PDF = HERE.parents[5] / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf"
pdf_sha = hashlib.sha256(PDF.read_bytes()).hexdigest()
assert pdf_sha == "4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af"
doc = fitz.open(PDF)
marks = []
for n in range(12, 19):
    page = doc[n - 1]
    lines = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            chars = [(c["c"], fitz.Rect(c["bbox"])) for span in line["spans"] for c in span["chars"]]
            lines.append((line, chars))
    for d in page.get_drawings():
        rect = d["rect"]
        if d["fill"] != (0.0, 0.0, 0.0) or rect.width > 20 or rect.height > 1:
            continue
        matches = []
        for line, chars in lines:
            chosen = [(c, b) for c, b in chars if b.x0 >= rect.x0 - .35 and b.x1 <= rect.x1 + .35 and abs(b.y1 - rect.y0) < 10]
            if chosen:
                matches.append({"printed_line": "".join(c for c, _ in chars).strip(),
                                "underlined_characters": "".join(c for c, _ in chosen),
                                "characters": [{"char": c, "bbox": [round(v, 2) for v in b]} for c, b in chosen]})
        assert len(matches) == 1, (n, rect, matches)
        marks.append({"printed_page": n, "pdf_page": n,
                      "rectangle": [round(v, 2) for v in rect], "match": matches[0]})
assert len(marks) == 20
out = {"source_pdf_sha256": pdf_sha, "status": "complete vector low-line census for pp. 12–18",
       "rendering_policy": "Excludes long blue URL underline and retains 20 separate short black rectangles; coordinates are original PDF points",
       "page_counts": {str(p): sum(m["printed_page"] == p for m in marks) for p in range(12, 19)},
       "marks": marks}
(HERE / "lowline-vector-census.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"marks": len(marks), "page_counts": out["page_counts"]}))
