#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 110 (items 115-117)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
115:("boy",["1 kowa hon","1 kuwa hon","1 kuwa hon","1 kuwa hon","1 kuwa hon","1 kuwa hon","1 kuwa hon","1 kuwa hon","1 kuwa hon","1 kuwa hon","1 kora hon",None,"1 kowa hon","1 kowa hon","4 here hol","1 kuwa hon",None,"1 kora hon","1 kuɖa hon","1 hon koɖa","1 koɖa hone","1 koɖa hone","1 kora","3 hon","1 kora","1 kora","2 pu·o, pila?"]),
116:("girl",["1 kui hon","1 kui hon","1 kui hon","1 kui hon","1 kui hon","1 kui hon","1 kui hon","3 ira hon","1 kui hon","1 kui hon","1 kuri hon",None,"1 kui hon","1 kui","3 ira hon",None,None,"1 kuri hon","1 kuɖi hon","1 kui hon","1 kuɖi hone","1 kuɖi hone","1 kori","1 kuri hon","1 kuri","1 kuri","2 dʒio pila?"]),
117:("day",["1 siŋgi, 2 ɖin","1 siŋgi","1 siŋgi","1 siŋgi","1 siŋgi","1 siŋgi","1 siŋgi","1 siŋgi","1 siŋgi","1 siŋgi","1 siŋgi","3 hula, 4 betaraŋ","1 siŋgi",None,"1 siŋgi","1 siŋgi","1 siŋgi","1 siŋgi","1 siŋgi","1 siŋgi","1 siŋgi","1 siŋgi",None,"3 ɖin, 3 hulaŋ","5 hilok","2 ɖin","2 ɖino"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=110 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=bool(form and any(c in form for c in "ɖʒ·"))
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"attested" if form else "blank","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 110 (items 115-117)")
if __name__=="__main__": main()
