#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 75 (items 10-12)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"
SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
10:("tongue",["1 le?e","1 le?e","1 le?e","1 le?e","1 le?e","1 le?e","1 lele (le?e)","1 le?e","1 le?e","1 le?e","1 le?e","1 le?e, 3 alaŋ","1 le?e","3 alaŋ (?)","1 le?e","3 ʌlaŋ","3 alaŋ","1 le?e","3 ʌlaŋ","3 alaŋ","3 alaŋ","3 alaŋ","3 alaŋ","1 le?e, 3 alaŋ","3 alaŋ","3 alaŋ","2 dʒibʰe"]),
11:("breast",["1 nunu, 3 ṯowa, 4 kuwam","1 nunu","1 nunu","1 nunu","1 nunu","1 nunu","1 nunu","1 nunu","1 nunu","1 nunu","1 nunu","3 toa","1 nunu",None,"1 nunu","1 nunu","1 nunu","1 nunu","1 nunu",None,"1 nunu","3 ṯuwa",None,"1 nunu","1 nunu, 4 koram","1 nunu","2 tʃaṯi"]),
12:("belly",["1 la?i","1 lai","1 lai","1 lai","1 lai","1 la?i","1 lai","1 lai","1 lai","1 lae?","1 la?i","1 laii","1 lai","1 lai","1 lai","1 lai","1 la?i","1 la?i","1 lei","1 lai","1 le?","1 lai","1 lai?","1 lai","1 la?e, 3 dodʒok","1 lʌ?e","2 peṯṯo"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=75 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]
  source_query=(i,s)==(10,"HO3")
  special=form is not None and any(c in form for c in "̱ʌʰ")
  uncertainty="source prints a parenthesized question mark after alaŋ" if source_query else ("diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "")
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else ("ambiguous" if source_query else "attested"),"Confidence":"medium" if uncertainty else "high","Uncertainty":uncertainty,"Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 75 (items 10-12)")
if __name__=="__main__": main()
