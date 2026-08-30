import csv
import unicodedata
from pathlib import Path

HERE=Path(__file__).resolve().parent
TSV=HERE/"pages_120_127_hand_keyed.tsv"
DECISIONS=HERE/"hand_keyed_pages_120_127.py"
DECL="hand-keyed-from-rendered-source; OCR-not-copied"

def rows():
    with TSV.open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f,delimiter="\t"))

def test_ocr_blind_schema_and_declaration():
    rs=rows(); assert len(rs)==648
    assert all(not name.startswith("OCR") for name in rs[0])
    assert all(r["Reviewer_Declaration"]==DECL for r in rs)
    assert all(unicodedata.is_normalized("NFC", value) for r in rs for value in r.values())
    source=DECISIONS.read_text(encoding="utf-8")
    assert "manual_review.tsv" not in source
    assert "OCR_Evidence" not in source
    assert "ocr_scaffold" not in source

def test_explicit_topology_and_statuses():
    rs=rows(); keys={(int(r["Item"]),r["Site_Code"]) for r in rs}
    assert len(keys)==648
    assert {i for i,_ in keys}==set(range(145,169))
    assert all(120<=int(r["PDF_Page"])<=127 for r in rs)
    assert all(int(r["Printed_Page"])==int(r["PDF_Page"])-9 for r in rs)
    assert all(r["Review_Status"] in {"attested","blank","ambiguous","illegible"} for r in rs)
    assert all(bool(r["Manual_Transcription"])==(r["Review_Status"]!="blank") for r in rs)
