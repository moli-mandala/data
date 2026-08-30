#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 106 (items 103-105)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
103:("woman",["1 era, 3 kui","1 ira","1 ira","3 kui","1 ira","1 ira hon","1 ira","3 kui","1 ira","1 ira, 3 kui hon","1 ira","1 era, 3 kui","1 ira","1 era","1 ira hon","1 ira","1 era hon","3 kuri","1 ira","3 kuɖi hon","3 kuɖi","1 ira","1 era, 3 kori","3 kuri","4 maedʒiu","3 kuri","2 sṯri"]),
104:("child",["1 sitija, 3 hon","3 hon","3 hon","3 hon","1 sitie","3 hon","3 hon","3 hon","3 hon","3 hon","3 hon","1 sitia, 3 hon","3 hon","3 hon","3 hon","3 hon","3 hon","3 hon","3 hone","3 hon","4 koɖa","3 hone","3 hon","3 hon","5 gidra","5 gidra","2 pil·a"]),
105:("father",["1 apu","1 ʌpu","1 ʌpu","1 apu","1 ʌpu, 2 baba","1 apu","1 ʌpu","1 ʌpu","1 ʌpu","1 apu","1 apu","1 apu","2 babu","1 apu","1 apu","2 bʌpa","2 bapa","1, 2 aba","2 babu","1, 2 aba","2 ba","2 babu","1 apu","1 apu, 1, 2 aba","1, 2 apa, 2 ba","2 baba","2 bapa"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=106 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=any(c in form for c in "̱ʌɖ·")
  vals={"Gloss":gloss,"Manual_Transcription":form,"Review_Status":"attested","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 106 (items 103-105)")
if __name__=="__main__": main()
