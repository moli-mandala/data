#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 85 (items 40-42)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
40:("ring",["1 pola, 2 mudam","1 pula","1 pula","1 pola, 2 muɖi","1 pula","1 pola","1 pola","1 pola","1 pula","1 pola","1 pola","1 pola, 2 mundam","1 pola",None,"1 pola","1 pola","1 pola","2 muldʌm","2 muŋɖem","2 muɖam","2 mudem","2 mudʌm",None,"1 pola","2 mundam","2 muɖem","2 muɖi"]),
41:("sun",["1 siŋgi"]*24+["3 siŋ tʃando","3 siŋ tʃaṇɖʌ","2 sudʒo"]),
42:("moon",["1 tʃaṇɖu","1 tʃandu","1 tʃandu","1 tʃʌndu?u","1 tʃandu","1 tʃʌdu?u","1 tʃandu","1 tʃandu","1 tʃandu","1 tʃandu?","1 tʃaṇɖu","1 tʃaṇɖu","1 tʃandu","1 tʃandu","1 tʃandur","1 tʃandu","1 tʃandu","1 tʃaṇɖu","1 tʃandup?","1 tʃandu","1 tʃadub","1 tʃʌdup","1 tʃandu","1 tʃandu","1 ninda tʃando","1 niɖe tʃaṇɖʌ","2 dʒanha"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=85 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=form is not None and any(c in form for c in "ʌɖṇ")
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 85 (items 40-42)")
if __name__=="__main__": main()
