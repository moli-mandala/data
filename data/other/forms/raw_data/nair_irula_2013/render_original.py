"""Bounded full-chapter page and text capture from the retained Nair 2013 PDF."""
import hashlib
import json
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
PDF = HERE.parents[5] / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf"
EXPECTED_SHA = "4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af"
assert hashlib.sha256(PDF.read_bytes()).hexdigest() == EXPECTED_SHA
doc = fitz.open(PDF)
assert len(doc) == 58
out = HERE / "original-review"
out.mkdir(exist_ok=True)
manifest = []
for printed in range(22, 28):
    page = doc[printed - 1]
    text_file = out / f"page-{printed}-raw-text.txt"
    image_file = out / f"page-{printed}.png"
    text_file.write_text(page.get_text(sort=True))
    pix = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
    pix.save(image_file)
    manifest.append({"printed_page": printed, "pdf_page": printed,
                     "text_sha256": hashlib.sha256(text_file.read_bytes()).hexdigest(),
                     "image_sha256": hashlib.sha256(image_file.read_bytes()).hexdigest(),
                     "image_bytes": image_file.stat().st_size})
(out / "render-manifest.json").write_text(json.dumps({"source_pdf_sha256": EXPECTED_SHA,
                                                     "page_count": len(doc),
                                                     "included_pages": manifest}, indent=2) + "\n")
print(json.dumps({"pages": len(manifest), "total_image_bytes": sum(x["image_bytes"] for x in manifest)}))
