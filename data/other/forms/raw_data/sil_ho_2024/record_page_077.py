#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 77 (items 16-18)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"
SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
16:("finger",["1 gaṇɖa","2 ṯi aŋudi","2 ṯi aŋgudi","2 aŋguli","1 ganda","2 aŋgul","2 aŋgudi","1 ganda","2 aŋgudi","1 ṯi ganda, 5 ṯi rama","2 aŋguri","1 gaṇɖa, 3 katub, 4 daro","3 ṯi katup",None,"2 aŋguli","2 aŋgudi","2 ɔŋguri","4 ɖaɖo","2 aŋgiɖi","2 aŋguli","2 aŋli","2 aŋgudi",None,"3 katu","3 katup","3 katub","2 aŋguli"]),
17:("nail",["1 sarsar","1 sarsar","1 sarsari","1 sarsar","1 sarsar","1 sarsar","1 sarsar","1 sarsar","1 sarsar","1 sarsar","1 sarsar",None,"1 sarsar",None,"1 sarsar","1 sarsar","1 sarsar","1 sarsar","1 sarsar","2 nokʰʌ","1 sarsar","1 sarsar",None,"3 rama","3 ṯi rama","3 rama","2 no·kʰo"]),
18:("leg",["1 kaṯa","1 kaṯa","1 kaṯa","1 kaṯa","1 kaṯa","1 kaṯa","1 kaṯa","1 kaṯa","1 kaṯa","1 kaṯa","1 kaṯa","1 kaṯa","1 kaṯa","1 kaṯa","1 kaṯa","1 kaṯa","1 kaṯa","1 karta","1 kaṯa","1 kaṯa","1 kaṯa","1 kaṯa","1 kaṯa","1 kaṯa","3 dʒaŋga","3 dʒaŋga","2 guɖo"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=77 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]
  special=form is not None and any(c in form for c in "̱ʌɔɖṇʰ·")
  uncertainty="diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else ""
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if uncertainty else "high","Uncertainty":uncertainty,"Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 77 (items 16-18)")
if __name__=="__main__": main()
