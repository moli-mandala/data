#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 104 (items 97-99)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
97:("monkey",["1 gai","1 gai","1 gai","1 gʌi","1 gai","4 sai","1 gai","1 gai","1 gai","1 gai","1 gai","1 gai, 3 sara","1 gaɭi",None,"2 makʌɖ","2 mʌkodo",None,"1 gaɖi, 5 hanuman","1 gaɖi","1 gaɭi","1 gaɭi","1 gaɖi",None,None,"1 gãrɨ",None,"2 maŋkərə"]),
98:("mosquito",["1 siki","1 sikiŋ","1 sikiŋ","1 sikɨ","1 siki","1 sikɨ","1 sikiŋ","1 sikiŋ","1 sikiŋ","1 siki·","1 sikɨ","1 sikin","1 sikɨ",None,"1 sikiŋ","1 sikiŋ","1 siki","1 sikʌŋi","1 sikiŋi","1 sikŋi","1 siknɨ","3 luti",None,"3 bʰusci","1 sikritʃ","1 sikŋi","2 motʃʰa"]),
99:("ant",["1 mu?i","1 moi","1 moi","1 mu?i","1 mo?i","1 mu?i","1 moi","1 moi","1 moi","1 mu?i","1 mɨi","1 muin","1 mu?i",None,"1 mɨi","1 moi","1 mo?i","1 mɨi","1 mui?","1 mɨ?i","1 mo?t","1 mu·i",None,"1 mui","2 mutʃ?","1 mui?","2 matʃi"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=104 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=form is not None and any(c in form for c in "ʌɖɭɨãəʰ·")
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 104 (items 97-99)")
if __name__=="__main__": main()
