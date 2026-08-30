#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 84 (items 37-39)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
37:("thread",["1 suṯam","1 suṯem","1 suṯem","1 suṯem","1 suṯem","1 suṯem","1 suṯem","1 suṯem","1 suṯem","1 suṯem","1 suṯem","1 suṯam","1 suṯem",None,"1 suṯem","1 suṯem","1 suṯem","1 suṯʌm","1 suṯem","1 suṯam","1 suṯem","1 suṯem",None,"1 suṯam","1 suṯam","1 suṯam","1 su·ṯa"]),
38:("needle",["1 sudʒa, 2 sui, 3 susi","1 sudʒe","1 sudʒe","1 sudʒe, 3 susi","1 sudʒe","1 sudʒe","1 sudʒe","2 sui","1 sudʒe","1 sudʒe","2 sui","1 sudʒa, 2 sui","2 sui",None,"3 susi","3 susi","3 susi","2 sui","3 susi","2 sui","2 sui","3 susi",None,"2 sui","2 sui","2 sui","1, 3 sɯn·tʃi"]),
39:("cloth",["1 lidʒa?a","1 lidʒe","1 lidʒe","1 lidʒe","1 lidʒe","1 lidʒe","1 lidʒe","1 lidʒe","1 lidʒe","1 lidʒe?","1 lidʒe","1 lidʒa","1 lidʒe",None,"1 lidʒa?a","1 lidʒe","1 lidʒʌ","3 kitʃtʃr","5 tiʃaŋ","3 kitʃiri","4 tieŋ","6 ulu",None,"1 lidʒa, 2 luga, 3 kitʃri","2 lugri, 3 kitʃitʃ","2 lugri","2 lu·ga"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=84 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=form is not None and any(c in form for c in "̱ʌɯ·")
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 84 (items 37-39)")
if __name__=="__main__": main()
