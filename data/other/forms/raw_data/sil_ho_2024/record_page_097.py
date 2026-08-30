#!/usr/bin/env python3
"""Record manual source-image transcription for PDF page 97 (items 76-78)."""
from __future__ import annotations
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent; LEDGER=HERE/"manual_review.tsv"; SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGE={
76:("turmeric",["1 sasaŋ","1 sʌsaŋ",None,"1 sʌsaŋ","1 sʌsaŋ","1 sʌsaŋ","1 sʌsaŋ","1 sʌsaŋ","1 sasaŋ","1 sasaŋ","1 sesaŋ","1 sasaŋ",None,None,"1 sasaŋ","1 sʌsaŋ","1 sʌsaŋ","1 sʌsaŋ","1 sʌsaŋ","1 sasaŋ","1 sasaŋ","1 sʌsaŋ",None,"1 sasaŋ","1 sasaŋ","1 sʌsaŋ","2 holidi"]),
77:("garlic",["1 rasui","1 rasui","1 rasui","1 rãsui","1 rasui","1 sasui","1 rasui","1 rasui","1 rasui","1 sasui (rasui)","1 rasui","1 rasuni","1 rãsui",None,"1 rasuɨ","1 rãsui","1 rasɯi","1 rasuŋi","1 rʌsuni","1 rasuŋi","1 rʌsuŋɨ","1 rasuŋi",None,"1 rãsuni","1 rasun","1 rʌsuŋ","1 rəsuŋə"]),
78:("onion",["1 pejadʒi","1 pijadʒi","1 pijadʒi","1 piadʒi","1 pijadʒi","1 piadʒi","1 pijadʒi","1 pijadʒi","1 pijadʒi","1 piadʒi","1 pijadʒi","1 peadʒi","1 piadʒi",None,"1 piadʒi","1 pijadʒi","1 pijadʒi","1 pijadʒi","1 pijadʒi","1 piadʒu","1 piadʒ","1 piadʒi",None,"1 peadʒu","1 peadʒ","1 piadʒ","1 piadʒo"]),
}
def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields; expected={(i,s) for i in PAGE for s in SITES}; seen=set()
 for row in rows:
  key=(int(row["Item"]),row["Site_Code"])
  if int(row["PDF_Page"])!=97 or key not in expected: continue
  i,s=key; gloss,forms=PAGE[i]; form=forms[SITES.index(s)]; special=form is not None and any(c in form for c in "ʌãɨɯə")
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"blank" if form is None else "attested","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict {i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)}")
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print("recorded 81 manually reviewed cells for PDF page 97 (items 76-78)")
if __name__=="__main__": main()
