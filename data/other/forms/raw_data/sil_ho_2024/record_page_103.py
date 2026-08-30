#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 103 (items 94-96)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
94:("goat",["1 merom","1 merʌm","1 merʌm","1 merʌm","1 merʌm","1 merʌm","1 merʌm","1 merʌm","1 merʌm","1 merʌm","1 merom","1 merom","1 merom","1 merom","1 merʌm","1 merʌm","1 merom","1 merom","1 merʌm","1 meram","1 merom","1 merʌm","1 merom","1 merom","1 merom","1 merʌm","2 tʃʰeli"]),
95:("dog",["1 seṯa","1 siṯa","1 siṯa","1 seṯa","1 siṯa","1 siṯa","1 siṯa","1 siṯa","1 siṯa","1 seṯa","1 siṯa","1 seṯa","1 siṯa","1 seṯa","1 seṯa","1 siṯa","1 seṯa","1 seṯa","1 siṯa","1 seṯa","1 seṯa","1 seṯa","1 seṯa","1 seṯa","1 seṯa","1 seṯa","2 kukura"]),
96:("snake",["1 biŋ"]*13+[None]+["1 biŋ"]*8+[None,"1 biŋ","1 bin, 3 kal","1 biŋ","2 sapo"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=103 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=form is not None and any(c in form for c in "̱ʌʰ")
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 103 (items 94-96)")
if __name__=="__main__": main()
