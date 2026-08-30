#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 107 (items 106-108)."""
from __future__ import annotations
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "manual_review.tsv"
SITES = "HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE = {
    106: ("mother", ["1 ma, 2 eŋga", "2 eŋga", "2 eŋga", "2 iŋga", "1 ma, 2 eŋga", "2 iŋga", "2 eŋga", "2 eŋga", "2 eŋga", "2 iŋga", "2 iŋga", "2 eŋga", "1 maŋ, 2 eŋga", "2 eŋga", "2 iŋga", "1 mʌŋ", "1 maŋ", "1 ma·", "1 maŋ", "1 ma", "1 maŋ", "1 maŋ", "2 eŋga", "2 eŋga", "2 eŋga, 3 ayo", "3 ayo", "1 ma?"]),
    107: ("older brother", ["1 bau, 3 ɖaɖa", "1 baute", "1 baute, 3 ɖaɖa", "1 bau", "1 baun", "1 bauŋ", "3 ɖaɖa", "3 ɖaɖa", "3 mʌrʌŋ ɖaɖa", "1 bau, 3 ɖaɖa", "3 ɖaɖa", "1 bauu", "3 ɖaɖa", "1 bao", "3 marʌŋ ɖaɖa", "3 mʌrʌŋ ɖaɖa", "3 maraŋ ɖaɖa", "3 ɖaɖa", "3 ɖaɖa", "3 maraŋ ɖaɖa", "3 maraŋ ɖaɖa", "3 ɖaɖa", None, None, "3 ɖada", "3 ɖaɖa", "2 nõna?"]),
    108: ("younger brother", ["1 unɖi, 3 boko, 5 babu", "1 unɖite", "1 undiŋ kuva", "1 undi", "1 undiŋ", "1 undi", "1 undi kuva", "1 undi", "1 undiŋ kuva", "1 undi", "1 huriŋ unɖi", "1 unɖi, 4 haga", "3 boyo", "1 undite", "4 hudiŋ haga", "4 huɖiŋ haga", "4 undiŋ hagʌ", "3 boko", "3 boko", "5 babu", "3 huɖiŋ bako", "4 hon haga", "4 haga", "4 haga", "3 bokot kora", "5 babu", "2 tʃʰoṯa bʰai"]),
}

def main():
    with LEDGER.open(encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f, delimiter="\t"); fields = rd.fieldnames; rows = list(rd)
    assert fields
    expected = {(i, s) for i in PAGE for s in SITES}; seen = set()
    for row in rows:
        key = (int(row["Item"]), row["Site_Code"])
        if int(row["PDF_Page"]) != 107 or key not in expected: continue
        i, s = key; gloss, forms = PAGE[i]; form = forms[SITES.index(s)]
        special = bool(form and any(c in form for c in "̱ʌɖ·ʰõ"))
        vals = {"Gloss": gloss, "Manual_Transcription": form or "", "Review_Status": "attested" if form else "blank", "Confidence": "medium" if special else "high", "Uncertainty": "diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "", "Reviewer_Method": "manual-source-image; rendered-180dpi; OCR-not-accepted", "Reviewed_At": "2026-08-28"}
        if row["Review_Status"] == "unreviewed": row.update(vals)
        else:
            for k, v in vals.items():
                if row[k] != v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
        seen.add(key)
    if seen != expected: raise AssertionError(f"page topology drift {len(seen)}")
    with LEDGER.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, delimiter="\t"); wr.writeheader(); wr.writerows(rows)
    print("recorded 81 manually reviewed cells for PDF page 107 (items 106-108)")

if __name__ == "__main__": main()
