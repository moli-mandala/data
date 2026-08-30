import csv,unicodedata
from pathlib import Path
H=Path(__file__).resolve().parent; T=H/"pages_136_141_hand_keyed.tsv"; S=H/"hand_keyed_pages_136_141.py"; D="hand-keyed-from-rendered-source; OCR-not-copied"
def rows():
 with T.open(encoding="utf8",newline="") as f:return list(csv.DictReader(f,delimiter="\t"))
def test_ocr_blind_explicit_nfc():
 r=rows();assert len(r)==486 and all(not c.startswith("OCR") for c in r[0]);assert all(x["Reviewer_Declaration"]==D for x in r);assert all(unicodedata.is_normalized("NFC",v) for x in r for v in x.values());s=S.read_text();assert "manual_review.tsv" not in s and "OCR_Evidence" not in s and "ocr_scaffold" not in s
def test_topology():
 r=rows();assert len({(x["Item"],x["Site_Code"]) for x in r})==486;assert {int(x["Item"]) for x in r}==set(range(193,211));assert all(int(x["Printed_Page"])==int(x["PDF_Page"])-9 for x in r);assert all(bool(x["Manual_Transcription"])==(x["Review_Status"]!="blank") for x in r)
