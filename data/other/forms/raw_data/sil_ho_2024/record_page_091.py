#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 91 (items 58-60)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
58:("mud",["1 losod","1 losod","1 losod","1 lʌsʌ?n","1 losod","1 lʌsʌ?n","1 losod","1 losod","1 losod","1 lʌsʌ?t","1 losor","1 losod","1 loso?n","1 losod","1 lʌsʌ?ʌ","1 losod","1 losor","1 losad?","1 losod, 2 kaɖom","1 lʌsʌ?ʌ","1 loso?n","2 kaɖʌm",None,"1 losod","1 losot","1 lʌsʌ?t","2 kaɖua"]),
59:("dust",["1 ɖulid","1 dulʌd?","1 dulʌd?","1 duled","1 dulid","1 duli?d","2 ɖuɖuger","1 dulʌd","2 ɖuɖuger","1 dulid?","1 duled?",None,"1 duli?d",None,"1 dulid?","1 dulid","1 ɖul?","1 dura?","1 ɖude","1 dʰura","1 dʰulʌ","1 dʰura",None,"3 garda","1 dʰuri","1 dʰuli","1 dʰuli"]),
60:("gold",["1 sona","1 sona","1 sona","1 sona, sune","1 sona","1 sono","1 sona","1 sona","1 sona","1 sona","1 sona","2 samom","1 sona","2 samrom","1 sona","1 sona","1 sona","2 sãmᵉrom","1 sona","1 sona","1 sona","1 sona","2 samrom","2 samrom","1 sona, 2 samarom","1 sɔna","1 sun·a"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=91 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=form is not None and any(c in form for c in "ʌɖʰãᵉɔ·")
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 91 (items 58-60)")
if __name__=="__main__": main()
