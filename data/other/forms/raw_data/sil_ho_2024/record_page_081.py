#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 81 (items 28-30)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
28:("door",["1 silpiŋ","1 silpiŋ","1 silpiŋ","1 silpiŋ","1 silpiŋ, 3 handed","1 silpiŋ","1 silpiŋ","1 silpiŋ","1 silpiŋ","1 silpiŋ","1 silpiŋ","1 silpiŋ, 4 duar","1 silpiŋ",None,"1 silpiŋ","4 ɖuver boŋ","1 silpiŋ, 4 ɖuwar","1 silpiŋ","5 tati","1 silpiŋ","5 tati","5 tati",None,"4 duar","1 silpiŋ, 2 kapat","1 silpiŋ","2 kɔbatɔ"]),
29:("firewood",["1 sa·n","1 san","1 san","1 sa·n","1 san","1 sʌ·n","1 san","1 san","1 san","1 sa·n","1 saŋ",None,"1 san",None,"1 sa·n","1 san","1 sahan","1 saŋ","1 san","1 sa·n","1 sahʌn","1 san",None,"1 sahan","1 sahan","1 sahan","2 ka·to"]),
30:("broom",["1 dʒono?o","1 dʒono?o","1 dʒono?o","1 dʒʌnʌ?ʌ","1 dʒono?o","1 dʒʌnʌ?o","1 dʒono?o","1 dʒono?o","1 dʒono?o","1 dʒʌnʌ?o","1 dʒono",None,"1 dʒo?no",None,"1 dʒono?","1 dʒono?o","1 dʒono","1 dʒono?","1 dʒono?o","1 dʒono?o","1 dʒono?","1 dʒono?",None,None,"1 dʒonok","1 dʒʌnɔ?ɔ","2 tʃʌntʃoni"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=81 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=form is not None and any(c in form for c in "ʌɔɖ·")
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 81 (items 28-30)")
if __name__=="__main__": main()
