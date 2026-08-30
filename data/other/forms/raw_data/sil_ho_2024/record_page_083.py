#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 83 (items 34-36)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
34:("knife",["1 katu","1 kaṯu","1 kaṯu","1 kaṯu","1 kaṯu","1 kaṯu","1 kaṯu","1 kaṯu","1 kaṯu","1 kaṯu","1 kaṯu, 2 tʃuri","1 kaṯu","1 kaṯu",None,"1 kaṯu, 2 tʃʰuri","3 puŋki","2 tʃuri","2 tʃuri","2 tʃʰuri, 3 puŋki","2 tʃʰuri","2 tʃʰuri","2 tʃʰuri",None,"1 kʌṯu","2 tʃʰuri","2 tʃʰuri","2 tʃʰuri"]),
35:("axe",["1 hake","1 hake","1 hake","1 hake","1 hake","1 hake","1 hake","1 hake","1 hake","1 hake","1 hake","1 hake, 3 kapl","1 hake",None,"1 hake","1 hake","1 hake","1 hake","4 boɖeʃa","1 hake","4 boɖia","4 boɖia",None,"3 kapl","2 taŋgu, 5 potam","2 taŋga","2 taŋgi·a"]),
36:("rope",["1 bajer, 3 bor, 4 paga","1 bʌjʌr","1 bʌjʌr","1 bajʌr","1 bʌjʌr","1 bajer","1 bʌjʌr","1 bʌjʌr","1 bʌjʌr","1 bajʌr","1 bajʌr","1 bajʌr, 3 bor, 4 paga","1 bʌjʌr",None,"1 bajar","1 bʌjʌr","1 bʌjʌr","1 baʲjar","1 baber","1 bajar","1 baber","1 bajar",None,"3 bor","1 bahari, 3 bor","1 baber","2 dowudi"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=83 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=form is not None and any(c in form for c in "̱ʌɖʰʲ·")
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 83 (items 34-36)")
if __name__=="__main__": main()
