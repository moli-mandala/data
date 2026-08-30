#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 96 (items 73-75)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
73:("eggplant",["1 beŋga",None,None,"1 ɖarubiŋga","1 ɖarubiŋga","1 biŋga","1 biŋga","1 ɖarubiŋa",None,"1 ɖarubiŋga","1 biŋga",None,"1 biŋga",None,"1 biŋga","1 biŋga","1 biŋgʌ","1 biŋᵉra","1 biŋgaɖ","1 beŋgaɖa","1 biŋgaɭ","1 biŋgaɖ",None,None,None,"1 beŋgaɖ","1 baiŋgoŋõ"]),
74:("groundnut",["1 tʃina baɖam","1 baɖam","1 baɖam","1 baɖʌm","1 baɖam","1 baɖʌm","1 baɖam","1 baɖam","1 baɖam","1 baɖam","1 bʌɖam",None,"1 baɖam",None,"1 baɖam","1 baɖam","1 baɖam","1 bʌɖʌm","1 baɖam","1 baɖam","1 bʌɖam","1 bʌɖam",None,None,"1 baɖam","1 baɖam","1 tʃinbaɖam"]),
75:("chili",["1 martʃi","1 martʃi","1 martʃi","1 martʃi","1 martʃi","1 martʃi","1 mʌrtʃi","1 mʌrtʃi","1 mʌrtʃi","1 martʃi","1 martʃi","1 martʃi","1 martʃi",None,"1 martʃi","1 martʃi","1 martʃi","1 murtʃi","1 murtʃi","1 murtʃi","1 murtʃi","1 murtʃi",None,"1 martʃi","1 maritʃ","1 mʌritʃ","1 maritʃə"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=96 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=form is not None and any(c in form for c in "ʌɖɭᵉõə")
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 96 (items 73-75)")
if __name__=="__main__": main()
