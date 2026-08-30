#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 88 (items 49-51)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
49:("lightning",["1 hitʃir","2 bidʒil","2 bidʒil","1 hitʃir","1 hitʃir","1 hitʃir","2 bidʒili","1 hitʃir","2 bidʒil","1 hitʃir","1 hitʃir","1 hitʃir","1 itʃir",None,"2 bidʒli","2 bidʒili","2 bidʒili","2 bidʒʌlʌʷ","1 itʃir ṯʌda, 2 bidʒlo","3 gʰʌdagati","2 bidʒili","2 bidʒʌli",None,"1 hitʃir, 4 ṯʰer",None,"2 bidʒli","2 bidʒuli"]),
50:("rainbow",["1 rulbijoŋ","1 rulbijoŋ","1 rulbijoŋ","1 rulbiŋ","1 rulbijoŋ","1 rulbijoŋ","1 rulbiŋ","1 rulbiŋ","1 rulbijoŋ","1 rulbiŋ, nurbiŋ","1 rulbiŋon","1 rulbiŋ","1 rulbijõŋ",None,"1 rulubiŋ","1 rulbiŋ","1 rulbijoŋ","3 bandelele?","7 luṇɖubiŋ",None,"1 rohoɖbiŋ","6 ram dʰʌnus",None,"3 baṇɖalele","5 liṯa a?k","5 liṯa?a","2 indro ɖanasa"]),
51:("wind",["1 hojo","1 hojo","1 hojo","1 hʌjʌ","1 hojo","1 hʌjʌ","1 hojo","1 hojo","1 hojo","1 hʌja","1 hoja","1 hoio","1 hojo",None,"1 hʌjʌ","1 hojo","1 hojo","1 hojo","1 hojo","1 hʌjʌ","1 hʌjʌ","1 hʌjo",None,"1 hojo","1 hoe","1 hoe","2 dʒʰoraka"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=88 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=form is not None and any(c in form for c in "̱ʌɖṇʰõʷ")
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 88 (items 49-51)")
if __name__=="__main__": main()
