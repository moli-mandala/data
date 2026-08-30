#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 89 (items 52-54)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
52:("stone",["1 ḏiri"]*13+[None]+["1 ḏiri"]*5+["1 ḏiri","1 dʰiri","1 ḏiri",None,"1 ḏiri","1 dʰiri","1 dʰiri","2 paṯhara"]),
53:("path",["1 hora"]*11+[None,"1 hora",None,"1 hora","1 hora","1 hora","1 hora","1 horen","1 hora","1 horeŋ","1 horen",None,"1 hora","1 hor, 4 sesa","1 hʌ·r","2 raʃṯra, 3 bato"]),
54:("sand",["1 giṯil","1 giṯil","1 giṯil","1 giṯil","1 giṯil","1 giṯil","1 giṯil","1 giṯil","1 giṯi","1 giṯil","1 giṯil","1 giṯil","1 giṯil",None,"1 giṯil","1 giṯil","1 giṯil","1 giṯil","1 giṯil","1 giṯil","1 giṯil","1 giṯil",None,"1 giṯil","1 giṯil","1 giṯil","2 bali"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=89 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=form is not None and any(c in form for c in "̱ʌʰ·")
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 89 (items 52-54)")
if __name__=="__main__": main()
