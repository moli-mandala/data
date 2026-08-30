#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 87 (items 46-48)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
46:("water",["1 ɖa?a"]*11+["1 ɖa","1 ɖa?a","1 ɖa?","1 ɖa?a","1 ɖa?a","1 ɖa?a","1 ɖa?ʌ","1 ɖa?a","1 ɖa?a","1 ɖa?","1 ɖa?","1 ɖa","1 ɖa","1 dak","1 ɖa?","2 pani"]),
47:("river",["1 gaɖa","1 gaɖa","1 gaɖa","1 garʌ","1 gaɖa","1 gaɖa","1 gaɖa","1 gaɖa","1 gaɖa","1 gaɖa","1 gaɖa","1 gara","1 gaɖa",None,"1 gara","1 gaɖa","1 gaɖʌ","1 gaɖʌ","1 gaɖa","1 gaɖa","1 gaɖa","1 gaɖa",None,"1 gaɖa","1 gaɖa","1 gaɖa","2 naɖi"]),
48:("cloud",["1 rimil","1 remil","1 rimil","1 remil","1 rimil","1 remil","1 rimil","1 rimil","1 rimi","1 rimil","1 rimil","1 rimil","1 remil",None,"1 remil","1 remil","3 gamʌ rakʌpɖʌra","1 remb?l","1 rimil",None,"1 remil","1 rembil",None,"1 rimil","1 rimil, 4 lahra","4 rahʌla","2 megʰ·o"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=87 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=form is not None and any(c in form for c in "ʌɖʰ·")
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 87 (items 46-48)")
if __name__=="__main__": main()
