#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 105 (items 100-102)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
100:("spider",["1 bindiram","1 biŋɖirem","1 biŋɖirem","1 bindirʌm","1 biŋɖirem","1 bindirem","1 biŋɖirem","1 bidirem","1 biɖeram","1 bindirʌm","1 biŋɖirem","1 bindiram","1 biŋɖerem",None,"2 tʃanṯul","1 biɖirem","2 ṯanṯula","1 bindiri","2 ṯaŋtale","1 bindiri","1 biŋɖi","2 ṯʌnṯula",None,"1 bindram","1 bindi","1 biŋɖi","3 budʰiaŋi"]),
101:("name",["1 nuṯum, 2 numu","1 noṯum","1 noṯum","1 nuṯum","1 noṯum","1 nuṯum","1 noṯum","1 noṯum","1 noṯum","1 nutum","1 luṯum","1 noṯum, 2 numu","1 nuṯum",None,"1 luṯum, nuṯum","1 noṯum","1 luṯum","1 nuṯum","2 numu","1 nutum","2 numu","2 numu",None,"1 nuṯum, 2 num","1 nuṯum","1 nuṯum","3 na·mɔ"]),
102:("man",["1 ho, 3 horo, 4 kowa"]+["1 ho"]*12+["1 ho, 3 horo"]+["1 ho"]*3+["3 horo","3 hoɖo","3 horo","3 hʌɖʌ","3 hoɖo","3 horo","3 horo, 4 kora","3 hor","3 hor","2 moniʃo"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=105 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; assert len(forms)==27; form=forms[SITES.index(s)]; special=form is not None and any(c in form for c in "̱ʌɖʰɔ·")
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 105 (items 100-102)")
if __name__=="__main__": main()
