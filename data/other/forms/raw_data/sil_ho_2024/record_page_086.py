#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 86 (items 43-45)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
43:("sky",["1 sirma","1 sirumʌ","1 sirumʌ","1 sirme","1 sirmʌ","1 sirme","1 sirumʌ","1 sirmʌ","1 sirmʌ","1 sirma","1 sirmʌ",None,None,None,"1 sirma",None,None,"1 sirma","1 sirmʌ",None,"1 sirma","1 sirma",None,"1 sirma","1 secma","1 sermʌ","2 akasau"]),
44:("star",["1 ipil"]*26+["2 ṯara, 3 nakʃaṯra"]),
45:("rain",["1 gama","1 gʌma","1 gʌma","1 gʌma","1 gʌma","1 gʌma","1 gʌma","1 gʌma","1 gʌma","1 gʌma","1 gɔma","1 gama","1 gama",None,"1 gama","1 gʌma","1 gamʌ","1 gɔmʌ","1 gʌma","1 gama","1 gama","1 gʌma",None,"3 dʒargi",None,None,"2 borosa"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=86 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=form is not None and any(c in form for c in "̱ʌɔ")
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 86 (items 43-45)")
if __name__=="__main__": main()
