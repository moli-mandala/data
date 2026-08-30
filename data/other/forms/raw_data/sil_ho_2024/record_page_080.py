#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 80 (items 25-27)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"
SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
25:("village",["1 hatu","1 haṯu","1 haṯu","1 haṯu","1 haṯu","1 haṯu","1 haṯu","1 haṯu","1 haṯu","1 haṯu","1 haṯu","1 haṯu","1 haṯu",None,"1 hatu","1 haṯu","1 haṯu","1 haṯu","1 haṯu","1 haṯu","1 haṯu","1 haṯu",None,"1 hatu, 3 ɖi","1 ato","1 ʌṯo","2 gra·mõ"]),
26:("house",["1 owa?a","1 owa?a","1 owa?a","1 owa?a","1 owa?a","1 owa?a","1 owa?a","1 owa?a","1 owa?a","1 owa?a","1 uwa?","1 oa","1 wo?a","1 oa","1 owa?a","1 owa?a","1 owa?a","1 oɽa","1 oɖa?a","1 oɖa?a","1 oɖ?a","1 oɖa?","1 ora","1 ora","1 orak","1 ora?a","2 gʰoro"]),
27:("roof",["1 ɖal",None,"4 owa?a tʃiṯan","7 ɖʌlʌb","1 owa?a ɖal","1 owa?a ɖal","1 owa?a ɖal","1 owa?a ɖal","1 owa?a ɖal","1 ɖal","4 uwa? tʃiṯan","3 salandi","7 ɖʌlop",None,"1 ɖal","1 owa?a ɖal","1 owa?a ɖal","3 sarima","3 saɖimi ṯed","6 ɖabea","7 ɖʌlʌb","3 saɖami",None,"3 sarami","5 tʃal",None,"2 tʃʰa·to"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=80 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]
  special=form is not None and any(c in form for c in "̱ʌɖɽʰõ·")
  uncertainty="diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else ""
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if uncertainty else "high","Uncertainty":uncertainty,"Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 80 (items 25-27)")
if __name__=="__main__": main()
