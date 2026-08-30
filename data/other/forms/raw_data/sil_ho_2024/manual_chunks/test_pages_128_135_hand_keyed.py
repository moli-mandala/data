import csv, unicodedata
from pathlib import Path
HERE=Path(__file__).resolve().parent
TSV=HERE/"pages_128_135_hand_keyed.tsv"; SRC=HERE/"hand_keyed_pages_128_135.py"
DECL="hand-keyed-from-rendered-source; OCR-not-copied"
def rows():
    with TSV.open(encoding="utf8",newline="") as f:return list(csv.DictReader(f,delimiter="\t"))
def test_ocr_blind_explicit_nfc():
    rs=rows(); assert len(rs)==648 and all(not c.startswith("OCR") for c in rs[0]); assert all(r["Reviewer_Declaration"]==DECL for r in rs); assert all(unicodedata.is_normalized("NFC",v) for r in rs for v in r.values()); s=SRC.read_text(); assert "manual_review.tsv" not in s and "OCR_Evidence" not in s and "ocr_scaffold" not in s
def test_topology():
    rs=rows(); assert len({(r["Item"],r["Site_Code"]) for r in rs})==648; assert {int(r["Item"]) for r in rs}==set(range(169,193)); assert all(int(r["Printed_Page"])==int(r["PDF_Page"])-9 for r in rs); assert all(bool(r["Manual_Transcription"])==(r["Review_Status"]!="blank") for r in rs)
