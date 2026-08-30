#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 82 (items 31-33)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
31:("mortar",["1 sʌsaŋ rɨḏ ḏiri","1 sʌsaŋ ḏiri",None,"2 ri?ṯ ḏiri","1 sʌsaŋ rɨḏ ḏiri","2 riṯ ḏiri","1 sʌsaŋ rɨḏ ḏiri","1 sʌsaŋ rɨḏ ḏiri","1 sʌsaŋ rɨḏ ḏiri, 3 ʌtini","1 sʌsaŋ rɨḏ ḏiri","2 riɖ? ḏiri",None,"1 sʌsaŋ rɨḏ ḏiri",None,"1 sʌsaŋ rɨḏ ḏiri","1 sʌsaŋ rɨḏ ḏiri","1 sʌsaŋ rɨḏ ḏiri","1 sʌsaŋ rɨḏ ḏiri","1 sʌsaŋ (2) rɨḏ ḏiri, 7 sil",None,"1 sʌsaŋ rɨḏ ḏiri","1 sʌsaŋ rɨḏ ḏiri",None,None,"5 ukʰur, 6 kandi","1 sʌsaŋ rɨḏ ḏiri","4 koṯṯuni, 7 silo"]),
32:("pestle",["2 gugu ḏiri",None,None,"1 hudiŋ ri?t ḏiri","1 rid ḏiri","2 gugu ḏiri","1 hudiŋ ḏiri",None,"2 gudugu ḏiri","1 hudiŋ red ḏiri","1 rid ṯiʌ",None,"2 gudgu",None,"2 gudgu","2 gudugu ḏiri","2 gurgu ḏiri","2 gorgi ḏiri","2 gudugu ḏiri","2 gudgu","6 hone dʰiri","6 hone ḏiri",None,None,"4 tok, 5 dʰusra","2 gudgʰu dʰiri","3 potʰoro"]),
33:("hammer",["1 kotasi, 3 marṯul, 4 koram","3 marṯul","3 marṯul","3 marṯul","1 kotasi","3 marṯul","3 marṯul","3 marṯul","3 marṯul","3 marṯul","1 kotasi, 3 matul","1 kotasi","3 marṯul",None,"2 haṯudi","3 marṯul","2 haṯuri","3 marṯur","3 marṯud","3 marṯul","3 marṯul","3 marṯul",None,"1 kutasi, 2 hataori","1 kutasi, 3 marṯul","1 kutasi, 3 marṯul","2 haṯuɖi"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=82 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=form is not None and any(c in form for c in "̱ʌɨɖʰ")
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 82 (items 31-33)")
if __name__=="__main__": main()
