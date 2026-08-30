#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 74 (items 7-9)."""
from __future__ import annotations
import csv
from pathlib import Path

HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"
SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
7:("nose",[
"1 muʃa, 3 muwa","1 mute","1 mute","1 mute","1 mute","1 mute","1 mute","1 mute","3 muve","1 mute","1 mut·e","1 muʃa, 3 muã","3 muẽ","1 muʃa, 3 muã","1,3 mɨe","5 mo dʒaŋgi","3 mɨa","4 mɨ","4 mu","4 mu·","4 mɨ","4 mu·","4 muhõn","4 muhu","4 mɨ","4 mu·","2 nakho"]),
8:("mouth",[
"1 a·","1 a","1 a","1 a·a","1 a","1 a·","1 a","1 a","1 a","1 a·a","1 a","1 a","3 motʃa","1 a","3 motʃa","3 motʃa","3 mʌtʃʌm","3 motʃa","3 motʃoŋ","3 motʃa","3 motʃoŋ","3 motʃʌŋ","3 motʃa","3 motʃa, 4 tʰotna","1 a","3 motʃa","2 pat·i"]),
9:("teeth",[
"1 daṯa","1 ḏʌta","1 ḏʌta","1 data","1 dʌta","1 ḏata","1 dʌta","1 ḏʌta","1 dʌta","1 data","1 datʌ","1 ḏaṯa","1 dʌ·ta","1 danta","1 data","1 dʌta","1 dʌta","1 ḏa?ta","1 data","1 data","1 data","1 data","1 ḏaṯa","1 ḏata","1 ḏaṯa","1 data","1 ḏanto"]),
}

def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=74 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]
  # This page contains dense legacy-alphabet distinctions. The transcription
  # is diplomatic; combining underbars, ʌ/ɨ, length dots and nasalization are
  # retained instead of normalized to another phonemic analysis.
  uncertainty="diplomatic Unicode rendering of legacy survey-alphabet glyphs" if any(c in form for c in "̱ʌɨ·ãẽõ") else ""
  vals={"Gloss":gloss,"Manual_Transcription":form,"Review_Status":"attested","Confidence":"medium" if uncertainty else "high","Uncertainty":uncertainty,"Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 74 (items 7-9)")
if __name__=="__main__": main()
