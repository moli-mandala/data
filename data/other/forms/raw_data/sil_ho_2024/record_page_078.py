#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 78 (items 19-21)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"
SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
19:("skin",["1 ɖr","1 ɖr","1 ɖr","1 ɖr","1 ɖr","1 ɖr","1 ɖr","1 ɖr","1 ɖr","1 ɖr","1 ɖr","1 ɖr, 3 harṯa","3 harṯa",None,"1 ɖr","3 harṯa","1 ɖr","1 ɖr","3 harṯa","3 harta","3 harṯa","1 ɖr",None,"1 ɖr, 3 harta","3 harta","3 harta","2 tʃarᵉmõ"]),
20:("bone",["1 dʒaŋ","1 dʒaŋ","1 dʒaŋ","1 dʒaŋ","1 dʒaŋ","1 dʒaŋ","1 dʒaŋ","1 dʒaŋ","1 dʒaŋ","1 dʒaŋ","1 dʒaŋ","1 dʒaŋ","1 dʒaŋ",None,"1 dʒaŋ","1 dʒaŋ","1 dʒaŋ","1 dʒaŋ","1 dʒaŋ","1 dʒaŋ","1 dʒaŋ","1 dʒaŋ",None,"1 dʒaŋ","1 dʒaŋ","1 dʒaŋ","2 ha·ɖo"]),
21:("heart",["1 dʒibon, 9 su·r","4 tik tik","1 dʒibon","1 dʒibon","1 dʒibon","1 dʒibon","1 dʒibon owa?a","10 kaldʒa",None,"1 dʒibon",None,"1 dʒi",None,None,"1 dʒibon owa?a","11 majam handa ṯe","1 dʒibon","6 majam oɽa","6 majam oɖa?",None,"5 majam kundi","6 majam oɖa?",None,"1 dʒi, 3 buka","3 boko, 7 ontor",None,"2 heruɖalo"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=78 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]
  special=form is not None and any(c in form for c in "̱ɖɽᵉõ·")
  uncertainty="diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else ""
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if uncertainty else "high","Uncertainty":uncertainty,"Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 78 (items 19-21)")
if __name__=="__main__": main()
