#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 76 (items 13-15)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"
SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
13:("arm",["1 supu, 3 rupi, 4 ṯi","3 rupi","3 rupi","1 supu","1 supu, 3 rupi","3 rupi","1 supu","3 rupi","1 supu","1 supu?u, 3 rupi","4 ṯi","4 ṯi","1 supu","4 ṯi","4 ṯi","1 supu","4 ṯi","4 ṯi","1 supu","4 ṯi","1 supu","4 ṯi","4 ṯi","4 ṯi","1 sopo, 4 ṯi","4 ṯi","2 haṯo"]),
14:("elbow",["1 uka",None,None,"1 uke","1 unuke","1 uke","1 uke","1 uke","5 kuvem dʒaŋ","1 uke","1 uk?ʌ","1 uka","1 uke",None,"2 kʰuni","1 uke","2 koṇi","1 uk·a","1 uke","3 gonti","1 uka?","1 uka?",None,"1 uka","4 moka","4 moka","2 koɨni"]),
15:("palm",["1 ṯi ṯalka","1 ṯi ṯʌlka","1 ṯi ṯʌlka","1 ṯalka","1 ṯi ṯalka","1 ṯi ṯalka","1 ṯi ṯʌlka","1 ṯi ṯʌlka","1 ṯʌlka","1 ṯi ṯʌlka","1 ṯi ṯʌlka",None,"1 ṯi ṯʌlka",None,"1 ṯalka","1 ṯi ṯʌlka","3 ṯi papuli","1 ti telka","1 ṯalʌka ṯi","1 ṯalka","1 ṯi ṯalka","1 ṯʌlka",None,"1 ṯi ṯalka","1 ṯalka","1 ṯʌlka","2 tolohaṯo, 3 papuli"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=76 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]
  special=form is not None and any(c in form for c in "̱ʌɨṇʰ·")
  uncertainty="diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else ""
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if uncertainty else "high","Uncertainty":uncertainty,"Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 76 (items 13-15)")
if __name__=="__main__": main()
