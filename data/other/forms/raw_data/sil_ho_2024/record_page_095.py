#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 95 (items 70-72)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
70:("millet (husked)",["1 gaŋgai","1 ṯileigaŋgai","1 ṯileigaŋgai","1 lutigaŋgʌi","1 ṯileigaŋgai","1 gaŋgai","1 gaŋgai","1 gaŋgai","1 ṯileigaŋgai","1 ṯileigaŋgai","1 gaŋgai","4 kode","2 dʒandʒaɖa",None,"2 dʒandʒada","2 dʒʌdʒaɖa","2 dʒandʒara","1 gaŋgai","1 gaŋgai","2 dʒandʒada","1 gaŋgai","1 gaŋgai",None,"4 kode","3 gundli","2 dʒandʒada",None]),
71:("rice (husked)",["1 tʃauli","1 tʃauli","1 tʃauli","1 tʃauli","1 tʃawli","1 tʃauli","1 tʃauli","1 tʃauli","1 tʃavli","1 tʃauli","1 tʃauli","1 tʃauli","1 tʃauli",None,"1 tʃauli","1 tʃauli","1 tʃauli","1 tʃauli","1 tʃauli","1 tʃauli","1 tʃʌuli","1 tʃauli",None,"1 tʃauli","1 tʃaoli","1 tʃauli","1 tʃawulo"]),
72:("potato",["1 alu"]*11+["2 saŋga","1 alu",None,"1 alu","1 gulalu","1 gularu","1 alu","1 golalu, 2 saŋga","2 saŋga","1 golaluli","1 alu",None,"1 alu","1 alu","1 ʌlu","1 ʌlu"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=95 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=form is not None and any(c in form for c in "̱ʌɖ")
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 95 (items 70-72)")
if __name__=="__main__": main()
