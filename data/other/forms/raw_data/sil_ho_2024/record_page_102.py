#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 102 (items 91-93)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
91:("milk",["1 ṯowa","1 ṯuwa","1 ṯuwa","1 ṯʌwa","1 ṯuwa","1 ṯua","1 ṯuwa","1 ṯuwa","1 ṯuwa","1 ṯuwa","1 ṯowa","1 ṯoa","1 ṯowa",None,"1 ṯowa","1 ṯuwa","1 ṯua?","1 ṯowʌ","1 ṯowa","1 ṯuwa","1 ṯua","1 ṯuwa",None,"1 ṯoa","1 ṯoa","1 ṯowa","2 kʰiro"]),
92:("horns",["1 ḏiriŋ"]*13+[None]+["1 ḏiriŋ"]*8+[None,"1 ḏiriŋ","2 siŋga, 3 ɖabe","1 ɖereŋ","2 siŋga"]),
93:("tall",["1 tʃaɖlʌm","1 tʃa?lʌm","1 tʃa?lʌm","1 tʃʌ?lʌm","1 tʃa?lʌm","1 tʃʌ?tʌm","1 tʃa?lʌm","1 tʃa?lʌm","1 tʃa?lʌm","1 tʃa?tlam","1 tʃa?lom","1 tʃaɖlʌm","1 tʃa?lʌm",None,"1 tʃa?elʌm","1 tʃa?lʌm","1 tʃa·lʌm","1 tʃalʌm","1 tʃʌ?lom","1 tʃa?tlam","1 tʃʌ?lom","1 tʃʌ?tlʌm",None,"1 tʃa?lʌm","1 tʃaṇɖbol","1 tʃae·lʌm","2 landʒo"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=102 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=form is not None and any(c in form for c in "̱ʌɖṇʰ·")
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 102 (items 91-93)")
if __name__=="__main__": main()
