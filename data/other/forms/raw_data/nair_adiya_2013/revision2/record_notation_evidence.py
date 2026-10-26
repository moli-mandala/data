"""Pin the chapter's drawn underlines, which PDF text extraction omits."""
import hashlib
import json
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
PDF = HERE.parents[5] / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf"
doc = fitz.open(PDF)
expected = {4: 4, 5: 4, 6: 7, 7: 2}
evidence = []
for printed, count in expected.items():
    page = doc[printed - 1]
    words = page.get_text("words")
    found = []
    for drawing in page.get_drawings():
        r = drawing["rect"]
        if not (0.2 < r.height < 2 and 2 < r.width < 30 and r.y0 < 740):
            continue
        hits = [w[4] for w in words
                if w[0] - 1 <= r.x0 <= w[2] + 1 and w[1] <= r.y0 <= w[3] + 3]
        if hits:
            found.append({"rect_pdf_points": [round(x, 2) for x in r],
                          "pdf_extracted_word": hits[0],
                          "evidence": "A separate horizontal line is drawn beneath the word; the line is absent from extracted text."})
    assert len(found) == count, (printed, found)
    evidence.append({"printed_page": printed, "pdf_index": printed - 1,
                     "underline_count": count, "drawn_underlines": found})

result = {
    "status": "complete chapter drawn-underline census, original vector geometry",
    "source_pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
    "total_drawn_underlines": 17,
    "source_policy": "A drawn underline is preserved by U+0332 on the affected source glyph(s), without a phonological or IPA interpretation. The p4 prose symbol l is documentary; 16 marked lexical witnesses are represented in first-reading and proposed CSV. p7 parraa has one line spanning both r glyphs, encoded r̲r̲ with typography uncertainty noted.",
    "pages": evidence,
}
(HERE / "notation-evidence.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print("17 source underline drawings across printed pages 4–7")
