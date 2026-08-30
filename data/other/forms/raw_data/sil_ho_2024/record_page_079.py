#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 79 (items 22-24)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"
SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
22:("blood",["1 majom","1 majam","1 majam","1 majam","1 majam","1 majam","1 majam","1 majam","1 majam","1 majam","1 majam","1 maiom","1 majam",None,"1 majam","1 majam","1 majam","1 maja","1 majam","1 majam","1 majam","1 majam",None,"1 majom, 2 rokot","1 majãm","1 majaŋ","2 rakto"]),
23:("urine",["1 ɖuki","1 doki","1 doki","1 duki","1 doki","1 duki","1 doki","1 doki","1 doki","1 duki","1 duki","1 duki","1 duki",None,"1 duki","1 doki","1 duki","1 duki",None,"1 duki","3 dʌdʌ","3 dʌɖo",None,"1 ɖuki, 3 ɖoɖo","4 aɖoeak","3 dʒarea","2 muṯṯo"]),
24:("feces",["1 i?i",None,None,"1 e?e","1 i?i","1 e?e","1 i?i","1 i?i",None,"1 e?e","1 i?i","1 i","1 e?e",None,"1 i?i","1 i?i","1 i?i","1 i?i","1 i?i","2 dʒʌḏa","1 e?","1 i·i?",None,"1 eee, 5 idʒʰ","5 itʃ?, 4 dʒidʒa",None,"2 dʒara"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=79 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]
  special=form is not None and any(c in form for c in "̱ʌɖʰã·")
  uncertainty="diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else ""
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if uncertainty else "high","Uncertainty":uncertainty,"Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 79 (items 22-24)")
if __name__=="__main__": main()
