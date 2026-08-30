#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 109 (items 112-114)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
112:("daughter",["1 kui hon, 3 hon era","1 kui hon","1 kui hon","1 kui hon","1 kui hon","1 kui hon","1 kui hon","1 kui hon","1 kui hon","3 hon ira","1 kuri hon","1 kui hon","1 kui hon","1 kui hon","3 ira hon","3 ira hon","3 hon era","1 kuri hon","1 kuɖi hone","1 kuri hon","1 kuɖi hone","3 hon era","1 kori","1 kuri hon","3 hopon era",None,"2 dʒi·o"]),
113:("husband",["1 ham, 3 herel, 4 kowa","4 kuwa","4 kuwa","1 ham","1 ham","4 kuwa","4 kuwa","4 kuwa","4 kuwa","1 ham","1 haram","1 ham, 3 herel","4 kuwa",None,"1 ham buɖa","3 herel","3 herel","6 kisan","4 kuɖa",None,"4 koɖa","4 koɖa",None,"4 kora, 5 purus","3 herel, 7 dʒãwae",None,"2 suami"]),
114:("wife",["1 era, 3 buɖi","1 ira","1 ira","3 ham buɖi","1 ira","1 ira","1 ira","7 kui ṯani","1 ira","1 ira, 3 ham buɖi","3 buɖi","1 era","1 ira","1 era","3 ham buɖi","1 ira","1 era","4 kuri","1 ira, 3 buɖi","3 haɖam buɖi","1 era","1 ira","1 era","4 kuri 5 ora horo","5 orak hor, 6 bahu",None,"2 sṯri"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=109 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=bool(form and any(c in form for c in "̱ɖʒ·ã"))
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"attested" if form else "blank","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 109 (items 112-114)")
if __name__=="__main__": main()
