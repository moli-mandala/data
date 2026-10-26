"""Crosscheck literal readings against native text and retain original line boxes.

This is evidence only: drawn underlines require the separate original/geometry
review, and text-layer matches are never independent acceptance.
"""
import hashlib
import json
import re
import unicodedata
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
ROOT = next(p for p in HERE.parents if (p / "data/SOURCE_INGESTION_CHECKLIST.md").exists())
PDF = ROOT / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf"
READING = PACKAGE / "physical-units-first-reading.jsonl"


def native(text):
    text = text.replace("\uf02c", "\u0323").replace("\uf019", "\u0303")
    return unicodedata.normalize("NFC", text)


def comparable(text):
    return re.sub(r"\s+", " ", native(text).replace("\u0332", "")).strip()


def main():
    units = [json.loads(s) for s in READING.read_text().splitlines()]
    evidence = []
    with fitz.open(PDF) as document:
        for unit in units:
            pages = unit.get("printed_pages", [unit["printed_page"]])
            for witness in unit["source_witnesses"]:
                form = comparable(witness["literal_first_reading"])
                # Source spells “suffix-kku”/“suffix-kaḷ” without a space.
                # Match the literal dash-led suffix, without requiring a word
                # boundary before its printed dash; never use bare substrings.
                prefix = "" if form.startswith(("-", "–")) else r"(?<!\w)"
                pattern = re.compile(prefix + re.escape(form) + r"(?!\w)")
                matches = []
                for p in pages:
                    for block in document[p - 1].get_text("dict")["blocks"]:
                        for line in block.get("lines", []):
                            raw = "".join(s["text"] for s in line["spans"])
                            if pattern.search(comparable(raw)):
                                matches.append(dict(printed_page=p, raw_line=raw, native_decoded_line=native(raw),
                                    line_rect=line["bbox"], scope="whole original line, not a guessed glyph rectangle"))
                evidence.append(dict(source_unit_key=unit["source_unit_key"], witness_key=witness["witness_key"],
                    form=witness["literal_first_reading"], original_line_matches=matches,
                    status="text layer agrees apart from independently recovered lowlines" if matches else "requires manual original/source-range check"))
        outside = {}
        for p in (3, 44, 50):
            raw = document[p - 1].get_text("text")
            path = HERE / f"outside-page-{p}-native.txt"
            path.write_text(raw)
            outside[str(p)] = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                disposition="contents documentary" if p == 3 else "adjacent chapter boundary; not Mannan lexical witnesses")
            if p == 50:
                image = PACKAGE / "original-review/page-50-boundary-150dpi.png"
                if not image.exists(): document[p - 1].get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72),alpha=False).save(image)
                outside[str(p)]["original_image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
    result = dict(status="owner crosscheck, not independent acceptance", source_pdf_sha256=hashlib.sha256(PDF.read_bytes()).hexdigest(),
        first_reading_sha256=hashlib.sha256(READING.read_bytes()).hexdigest(), witness_records=len(evidence),
        matched=sum(bool(x["original_line_matches"]) for x in evidence), records=evidence, outside_boundary_evidence=outside)
    (HERE / "native-text-crosscheck.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("witness_records","matched","first_reading_sha256")},ensure_ascii=False))
    for x in evidence:
        if not x["original_line_matches"]: print(x["witness_key"], x["form"])


if __name__ == "__main__": main()
