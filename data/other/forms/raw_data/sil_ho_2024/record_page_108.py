#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 108 (items 109-111)."""
from __future__ import annotations
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "manual_review.tsv"
SITES = "HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE = {
    109: ("older sister", ["1 adʒi, 4 ɖai", "1 adʒite", "1 adʒite, 4 ɖai", "1 adʒi, 4 ɖai", "1 adʒiŋ", "4 ɖai", "4 ɖai", "4 ɖai", "4 ɖai", "1 adʒi", None, "1 adʒi", "2 nana", None, "4 ɖai", "2 nani", "4 ɖai", "4 ɖai", "2 nana", "4 ɖai", "2 nana", "3 ɖiɖi", None, "2 nana", "1 adʒi, 4 ɖai", "4 ɖʌ·i", "2 nan·i, 3 ɖiɖi"]),
    110: ("younger sister", ["1 unɖi kui, 4(3 misi)era, 8 mai", "1 unɖite kui", "1 undiŋ kui", "1 undi kui", "1 undiŋ kui", "1 undi kui", "1 undi kui", "10 misi kui", "1 undiŋ kui", "1 undi kui", "3 uriŋ misi", "1 unɖi kui, 4 misi era", "6 biti", "3 misi", "5 buɖi", "9 huɖiŋ nani", "7 uriŋ bokoŋ", "3 uriŋ misi", "4 misi ira", "5 buɖi", "4 misi ira", "11 hon misi", "3 misi", "3 misi", "7 bokot kuri", "8 mʌ·i", "2 sana bʰouni"]),
    111: ("son", ["1 kowa hon, 3 hon", "1 kuwa hon", "1 kuwa hon", "1 kuwa hon", "1 kuwa hon", "1 kuwa hon", "1 kuwa hon", "1 kuwa hon", "1 kuwa hon", "3 hon", "1 kora hon", "1 kowa hon, 3 hon", "1 kuwa hon", "1 kowa hon", "5 horel hon", "1 kuwa hon", "5 hon herel", "1 kora hon", "1 kuɖa hone", "1 korʌ hon", "3 hone", "5 hon herel", "1 kora", "3 hon", "1 korʌ hopon, 3 hon", None, "2 pu·o"]),
}

def main():
    with LEDGER.open(encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f, delimiter="\t"); fields = rd.fieldnames; rows = list(rd)
    assert fields
    expected = {(i, s) for i in PAGE for s in SITES}; seen = set()
    for row in rows:
        key = (int(row["Item"]), row["Site_Code"])
        if int(row["PDF_Page"]) != 108 or key not in expected: continue
        i, s = key; gloss, forms = PAGE[i]; form = forms[SITES.index(s)]
        special = bool(form and any(c in form for c in "ʌɖʒ·ʰ"))
        vals = {"Gloss": gloss, "Manual_Transcription": form or "", "Review_Status": "attested" if form else "blank", "Confidence": "medium" if special else "high", "Uncertainty": "diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "", "Reviewer_Method": "manual-source-image; rendered-180dpi; OCR-not-accepted", "Reviewed_At": "2026-08-28"}
        if row["Review_Status"] == "unreviewed": row.update(vals)
        else:
            for k, v in vals.items():
                if row[k] != v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
        seen.add(key)
    if seen != expected: raise AssertionError(f"page topology drift {len(seen)}")
    with LEDGER.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, delimiter="\t"); wr.writeheader(); wr.writerows(rows)
    print("recorded 81 manually reviewed cells for PDF page 108 (items 109-111)")

if __name__ == "__main__": main()
