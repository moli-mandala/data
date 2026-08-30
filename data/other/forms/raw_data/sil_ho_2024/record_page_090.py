#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 90 (items 55-57)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
55:("fire",["1 seŋgel"]*17+["1 siŋgel"]+["1 seŋgel"]*8+["2 nɨ·a"]),
56:("smoke",["1 mo?o, 3 sukul","1 mo?o","1 mo?o","1 mo?o","1 mo?o","1 mo?o","1 mo?o","1 mo?o","1 mo?o","1 mo?o","1 mo?o, 3 sukul","3 sukul","1 mo?o, 4 ɖuŋgi",None,"1 mo?o, 2 dʰuã","3 sukul","3 sukul","3 sukur","3 sukul","3 sukul","3 sukul","3 sukul",None,"3 sukul, 4 dʰuŋgia","2 dʰuã, 4 dʰuŋgia","2 dʰuã","2 dʰuã"]),
57:("ash",["1 ṯoro?e","1 ṯoroe","1 ṯoroe","1 ṯʌrʌe","1 ṯoroe","1 ṯʌrʌje","1 ṯoroe","1 ṯo?roe","1 ṯoro?e","1 ṯʌrʌe?","1 raṯorne?","1 ṯoroe","1 ṯʌroe",None,"1 ṯʌrae","1 ṯoroe","1 ṯoro?e","1 ṯoro?e","1 ṯorodʒ?","1 ṯʌre","1 ṯorʌ?ṯ","1 ṯoret?",None,"1 ṯoroe","1 ṯorotʃ?","1 ṯʌrʌtʃ?","2 pausa"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=90 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=form is not None and any(c in form for c in "̱ʌɖʰɨã·")
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 90 (items 55-57)")
if __name__=="__main__": main()
