"""Root's independent source renders for the complete Adiya chapter."""
import hashlib
import json
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
PDF = next(p for p in HERE.parents if (p / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf").exists()) / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf"
assert hashlib.sha256(PDF.read_bytes()).hexdigest() == "4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af"
document = fitz.open(PDF)
pages = []
for number in range(4, 9):
    path = HERE / f"pdf{number}-original.png"
    document[number - 1].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(path)
    pages.append({"page": number, "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
(HERE / "render-manifest.json").write_text(json.dumps({"source_pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(), "pages": pages}, indent=2) + "\n")
