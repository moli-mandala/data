#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 98 (items 79-81)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
79:("cauliflower",["1 ba kobi","1 ba kobi","1 ba kobi","1 ba kubi","1 ba kobi","1 ba kubi","1 ba kobi","1 ba kobi","1 ba kobi","1 ba kubi","1 ba kobi",None,"1 ba kobi",None,"1 ba?a kobi","1 ba kobi","1 ba kopi","1 ba kobi","1 bo kobi","1 ba kobi","1 baha kobi","1 baha kobi",None,None,None,"1 baha kobi","2 pʰul kobi"]),
80:("tomato",["1 belaṯi beŋga","1 bilaṯi biŋga","1 bilaṯi biŋga","1 bilaṯi biŋga","1 bilaṯi biŋga","1 bilaṯi biŋga","1 bilaṯi","1 bilaṯi","1 bilaṯi biŋga","1 bilaṯi biŋga","1 bilaṯi",None,"1 bilaṯi",None,"1 bileṯi biŋga","1 bilaṯi","1 bilaṯi","1 bilaṯi","1 bilaṯi","1 bilaṯi beŋgeɖa","1 bilaṯi","1 bilaṯi",None,"1 bilaiṯi",None,"1 bilaṯi biŋga","1 bilaṯi"]),
81:("cabbage",["1 potom kobi, putuwa kobi","1 putuve kobi","1 putuve kobi","1 puto kubi","1 potom kobi","1 putue kubi","1 putuve kobi, 2 bʌṇɖa kobi","4 ṯol kobi","1 putuve kobi","1 potʌm kubi, putuve kobi","1 potom kobi",None,"1 potom",None,"1 potʌm kobi","1 potom kobi","1 potʌm kobi","1 pʌṯom kobi","1 potom kobi","2 boṇɖa kobi","1 pʌtʌr kobi","1 potʌm kopi",None,None,"3 kubi arak","1 potəm kobi","2 baṇɖʰa kobi"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=98 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=form is not None and any(c in form for c in "̱ʌɖṇʰə")
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 98 (items 79-81)")
if __name__=="__main__": main()
