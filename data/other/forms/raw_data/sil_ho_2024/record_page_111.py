#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 111 (items 118-120)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
118:("night",["1 niɖe","1 niɖe","1 niɖe","1 niɖe","1 niɖe","1 niɖe","1 niɖe","1 niɖe","1 niɖe","1 niɖe","1 niɖe","1 niɖa","1 niɖe",None,"1 niɖe","1 niɖe",None,"1 niɖʌ","1 niɖe","1 niɖe","1 nɖe","1 niɖa",None,"1 niɖa","1 ninda","1 niɖe","2 raṯi"]),
119:("morning",["1 seṯaʔa","1 siṯaʔa","1 siṯaʔa","1 siṯaʔa","1 siṯaʔa","1 siṯaʔa","1 siṯaʔa","1 siṯaʔa","1 siṯaʔa","1 siṯaʔa","1 siṯa","1 seṯa","1 siṯaʔa",None,"1 seṯaʔa","1 siṯaʔa","1 seṯaʔ","1 suṯ·a","1 siṯaʔa","1 seṯaʔa","1 siṯaʔa","1 siṯaʔ",None,"1 seta, 3 idaŋ","1 setak","1 seṯaʔ","2 sakalə"]),
120:("noon",["1 ṯikin","1 ṯikin","1 ṯikin","1 ṯikin","1 ṯikin","1 ṯikin","1 ṯikin","1 ṯikin","1 ṯikin","1 ṯikin","1 ṯikin","1 ṯikin","1 ṯikin",None,"1 ṯikin","1 ṯikin","1 ṯikin","1 ṯikin","1 ṯikin","1 ṯikin","1 ṯikin","3 dʰupar",None,"1 ṯikin","1 ṯikin","1 ṯikin","2 maɖʰjan·ə"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=111 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=bool(form and any(c in form for c in "̱ɖʌʔ·əʰ"))
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"attested" if form else "blank","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 111 (items 118-120)")
if __name__=="__main__": main()
