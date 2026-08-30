#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 94 (items 67-69)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
67:("mango",["1 uli"]*13+[None]+["1 uli"]*9+["1 uli","1 ul","1 ul","2 ambo"]),
68:("banana",["1 kaɖʌl","1 kaɖʌl","1 kaɖʌl","1 kaɖal","1 kaɖal","1 kʌɖʌl","1 kaɖʌl","1 kaɖʌl","1 kaɖʌl","1 kaɖal","1 kaɖal",None,"1 kaɖal",None,"1 kaɖal","1 kaɖʌl","1 kaɖʌl","1 kʌɖʌla","1 kaɖal","1 kaɖala","1 kʌɖal","1 kaɖal",None,None,None,"2 kaerca","1 kodoli"]),
69:("wheat (husked)",["1 gom","1 gom","1 gom","1 gʌm","1 gom","1 gʌm","1 gom","1 gom","1 gom","1 gʌm","1 gom","1 gom","1 gom",None,"1 gʌhʌm","1 gohom","1 gomo","1 gohom","1 gohom","1 gʌm","1 gʌhʌm","1 gʌhʌm",None,"1 gohom","1 guhum","1 gɔhɔm","1 gohomõ"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=94 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=form is not None and any(c in form for c in "ʌɖɔõ")
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 94 (items 67-69)")
if __name__=="__main__": main()
