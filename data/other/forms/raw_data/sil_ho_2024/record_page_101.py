#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 101 (items 88-90)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
88:("egg",["1 dʒarom","1 dʒarʌm","1 dʒarʌm","1 dʒarʌm","1 dʒarʌm","1 dʒarʌm","1 dʒarʌm","1 dʒarʌm","1 dʒarʌm","1 dʒʌrʌm","1 dʒarom","1 dʒarom","1 dʒarʌm",None,"1 dʒarʌm, 2 ɔnɖa","4 bitʃaɖi","2 ɔnɖa","2 ʌnɖʌ","4 bitʃaɖi","4,5 petal","4 bitʃʌli","5 peɖʌo",None,"1 dʒarom, 3 bili","3 bele","3 bili","2 oṇɖa"]),
89:("cow",["1 gundi, 3 uri","1 gundi","1 gundi","1 gundi","1 gundi","1 gundi","3 uri","3 uri","1 gundi","1 gundi","1 gundi","1 gundi","1 gundi","2 gai, 3 uri","2 gai","3 uri","2 gai, 3 uri","1 gundi","2 gel, 3 uri","2 gai","2 gʌi","2 gʌi","2 gai, 3 urig","1 gundi, 2 gai, 3 uri","2 gai, 4 daŋgri","2 gʌ·e","2 gai"]),
90:("buffalo",["1 keɖa, 3 birkeɖa","1 kiɖa","1 kiɖa","1 keɭa","1 kiɖa","1 kiɖa","1 kiɖa","1 kiɖa","1 kiɖa","1 kiɖa","1 keɖa","1 kera","1 kiɖa",None,"1 keɭa","1 kiɖa","1 keɖa","1 keɖa","1 keɖa, 2 mõisi","1 keɖa","1 kiɖa","1 kiɖa, 2 mõs",None,"1 keɖa, 3 birkera","1 kaɖa, 5 bitkil","1 kaɖa","2 moi·ʃa"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=101 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=form is not None and any(c in form for c in "ʌɔɖṇɭõ·")
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 101 (items 88-90)")
if __name__=="__main__": main()
