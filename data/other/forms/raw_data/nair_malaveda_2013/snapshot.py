"""Pin the complete Nair Malaveda/Veda chapter's source pages."""
import hashlib
import json
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
PDF = next(p for p in HERE.parents if (p / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf").exists()) / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf"
EXPECTED = "4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af"
assert hashlib.sha256(PDF.read_bytes()).hexdigest() == EXPECTED
document = fitz.open(PDF)
assert len(document) == 58
inventory = []
for printed_page in range(40, 45):
    page = document[printed_page - 1]
    native = page.get_text()
    assert f"\n{printed_page} " in native[:180]
    text_file = HERE / f"pdf{printed_page}-native.txt"
    text_file.write_text(native)
    render_file = HERE / f"pdf{printed_page}-original.png"
    page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(render_file)
    inventory.append({"printed_and_pdf_page": printed_page, "native_text_sha256": hashlib.sha256(text_file.read_bytes()).hexdigest(),
                      "page_render_sha256": hashlib.sha256(render_file.read_bytes()).hexdigest()})
(HERE / "snapshot.json").write_text(json.dumps({"source_pdf_sha256": EXPECTED, "pdf_pages": len(document), "chapter_pages": inventory,
                                         "method": "native embedded PDF text plus original-page renders; no OCR"}, indent=2) + "\n")
