#!/usr/bin/env python3
"""Record manual source-image transcriptions for PDF pages 112-119 (items 121-144)."""
from __future__ import annotations
import csv
from pathlib import Path

HERE=Path(__file__).resolve().parent
LEDGER=HERE/"manual_review.tsv"
SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
PAGES={
112:{
121:("evening",["1 ajub","1 ʌup","1 ʌup","1 ajub","1 ʌub siŋgi","1 ajup","1 ʌub","1 ʌub","1 ʌupeŋ","1 ajub","1 aijup","1 aiub","1 ajub",None,"1 ajup","1 ʌub","1 ajup","1 aijupsa","1 ʌub siŋgi","1 ajub","1 ajub","3 siŋad",None,"1 ajub","1 ajup","1 ejub","2 saŋɖʰja"]),
122:("yesterday",["1 hola","1 hula","1 hula","1 hola","1 hola","1 hola","1 hola","1 hola","1 hola","1 hola","1 hola","1 hola","1 hola",None,"1 hola","1 hola","1 hola","1 hola","1 hola","1 hola","1 hola","1 hola",None,"1 hola","1 hola","1 hʌla","2 kali"]),
123:("today",["1 ṯisiŋ","1 ṯisiŋ","1 ṯisiŋ","1 ṯisiŋ","1 ṯisiŋ","1 ṯisiŋ","1 ṯisiŋ","1 ṯisiŋ","1 ṯisiŋ","1 ṯisiŋ","1 ṯisiŋ","1 ṯisiŋ","1 ṯisiŋ",None,"1 ṯisiŋ","1 ṯisiŋ","1 ṯisiŋ","1 ṯisiŋ","1 ṯisiŋ","1 ṯisiŋ","1 ṯisiŋ","1 ṯisiŋ",None,"1 ṯisiŋ","1 ṯehen","1 ṯehen","2 adʒi"]),
},
113:{
124:("tomorrow",["1 gapa","1 gʌpa","1 gapa","1 gʌpa","1 gapa","1 gapa","1 gapa","1 gʌpa","1 gapa","1 gʌpa","1 gəpa","1 gapa","1 gʌpa",None,"1 gʌpa","1 gʌpa","1 gʌpa","1 gəpa","1 gʌpa","1 gapa","1 gʌpa","1 gʌpa",None,"1 gapa","1 gapa","1 gapa","2 asonṯa kali"]),
125:("week",["1 ha·t, hato","1 hato","1 hato","1 hatʌ","1 hato","1 hatʌ","1 hato","1 hato","1 hato","1 hatʌ","2 hapṯa","1 hat","1 hato",None,"1 hatʌ","1 hato","1 sat? ɖin","2 hapṯʌ","1 hat, 2 hapṯa","2 hapṯa","1 hat","2 hapṯa",None,None,"2 hapṯa","2 hapṯa","2 sopṯaha"]),
126:("month",["1 tʃaŋɖuʔu","1 tʃandu","1 tʃandu","1 tʃʌndu","1 tʃandu","1 tʃʌndu","1 tʃandu","1 tʃandu","1 tʃandu","1 tʃandu","1 tʃandu","1 tʃaŋɖu","1 tʃandu",None,"1 tʃandup","1 tʃandu","1 tʃaŋɖu","1 tʃaŋɖuʔ","1 tʃandu","1 tʃandu","1 tʃandu","1 tʃʌndup",None,"1 tʃaŋɖu","1 tʃando","1 tʃʌnɖʌ","2 masoʔ"]),
},
114:{
127:("year",["1 sirma, 3 kalom, 4 botʃor","1 sirumʌ","1 sirumʌ","1 sirma","1 sirmʌ","1 sirme","1 sirumʌ","1 sirumʌ","1 sirmʌ","1 sirma","1 sirma","1 sirma","1 sirme",None,"1 sirma","1 sirum","1 sɪrəmʌ","1 sirma","1 sirumʌ","1 sirma","1 sirma","1 sirma",None,"1 sirma, 3 kalom","1 serma, 4 botʃor","1 serma","2 barsa"]),
128:("old (object)",["1 papari","1 papari","1 papari","1 papari","1 papari","1 papari","1 papari","1 papari","1 papari","1 papari","1 papari","1 papri","3 mare",None,"2 purna","2 purna","2 purʌna","2 purn·a","2 purne","2 purna","2 purna","2 purna",None,"2 purana","3 mare","3 mare","2 poruna"]),
129:("new (object)",["1 nama","1 nʌma","1 nʌma","1 nama","1 nama","1 nama","1 nʌma","1 nʌma","1 nʌma","1 nama","1 nʌma","1 nama","1 nama",None,"1 nana","1 nʌma","1 nʌma","1 naʷwʌ","1 nʌma","1 nama","1 nama","1 nama",None,"1 nawa","1 nawa","1 nãwa","1 nu·a"]),
},
115:{
130:("good",["1 bugin, bugi","1 bugin","1 bugin","1 bugin","1 bugin","1 bugin","1 bugin","1 bugin","1 bugin","1 bugin","1 bugin","1 bugin","1 bugin","1 bugi","1 bugi, 3 bes","1 bugin","3 bẽs","1 bogin","3 bes","3 bes","3 bes","3 bes","1 bugin","1 bugin","1 boge, 3 bes","3 bes","2 bʰolo"]),
131:("bad",["2 eɖka","1 karap","1 karap","1 karʌp, 2 etka","1 karap","1 karʌb","1 karab","1 karab","1 karap","2 etka","1 kʌrab","2 etka","1 karap","2 etka","2 etka","1 kʌrap","1 kʌrap","1 kʰʌrab","1 korab","1 kʰarap","1 kʰʌrʌp","1 kʰʌrap","2 eigkan","2 etkan","1 karap, 3 baritʃ?","3 bʌri","1 karapo"]),
132:("wet",["1 lum, 3 lowaɖ","1 lum","1 lum","1 lum","1 lum","1 lum","1 lum","1 lum","4 dʒaɖi","1 lum","3 loa","1 lum, 2 oɖaɖ","3 load·jana",None,"1 lum, 2 oɖaʔt","1 lum","2 oɖac","1 lum","1 lumdʒ, 2 oɖaɖ","2 aɖaʔt","1 lum","1 lum",None,"1 lum","2 oda, 5 lohot","5 lʌhʌ","2 oɖa"]),
},
116:{
133:("dry",["1 ro·","1 ro","1 ro","1 rʌ","1 ro","1 rʌ","1 ro","1 ro","1 ro","1 rʌ","1 ro","1 ro","1 ro",None,"1 rə","1 ro","1 ro","5 roro","5 roɖu","5 roʔlo","5 rʌhoɖ","5 roɖo",None,"5 ror","3 tʃuṯtʃaṯ, 4 hindʒit","5 rʌhʌ","2 sukʰila"]),
134:("long (object)",["1 dʒiliŋ","1 dʒiliŋ","1 dʒiliŋ","1 dʒiliŋ","1 dʒiliŋ","1 dʒiliŋ","1 dʒiliŋ","1 dʒiliŋ","1 dʒiliŋ","1 dʒiliŋ","1 dʒiliŋ","1 dʒiliŋ","1 dʒiliŋ",None,"1 dʒiliŋ","1 dʒiliŋ","1 dʒiliŋ","1 dʒiliŋ","1 dʒiliŋ","1 dʒiliŋ","1 dʒiliŋ","1 dʒiliŋ",None,"1 dʒiliŋ","1 dʒelen, 3 dʒʰal","3 tʃʰel","2 lomba"]),
135:("short (object)",["1 ɖuŋgui","1 ɖuŋgui","1 ɖuŋgui","1 ɖuŋgui","1 ɖuŋgui","1 ɖuŋgui","1 ɖuŋgui","1 ɖuŋgui","5 kato","3 huɖiŋ",None,"3 huriŋ, 4 tum","5 kʰato",None,"3 huɖiŋ, 6 kʰandia","6 kandi sa","6 kaŋɖiʌ","5 kʌto","5 kʌto","3 huɖiŋ","5 kʰʌto","5 kʰatʌ",None,"3 huriŋ, 4 tum","5 kʰato, 7 geɖa","5 kati","2 tsoṯia"]),
},
117:{
136:("hot",["1 lolo, 3 dʒete","1 lolo","1 lolo","1 lʌlʌ","1 lolo","1 lʌlʌ","1 lolo","1 lolo","1 lolo","1 lolo","1 lolo","1 lolo, 3 dʒete","1 lolo",None,"1 lolo","1 lolo","1 lolo","1 lolo","1 lolo","1 lolo","1 lʌlʌ","1 lʌlʌ",None,"1 lolo, 3 dʒete","1 lolo","1 lolo","2 gorano"]),
137:("cold",["1 sasa, 3 rabin, 4 reja","1 sasa","1 sasa","1 sa·sʌ","1 sasa","1 sa·sʌ","1 sasa, 3 rʌbaŋ","1 sasa, 3 rʌbaŋ","1 sasa","1 sa·sa, 4 reja","1 sasa","1 sasa, 3 rabaŋ","4 rae",None,"1 sasa","3 rʌbaŋ","4 rai","4 reja","3 rʌban, 4 rijaɖ","4 rejada","4 rejaɖa","4 rijaɖ",None,"3 rabaŋ","3 raban","4 rijar","2 ṯhanda"]),
138:("right (not counted in comparison)",["dʒom ṯi, etom","dʒom pa","dʒom pa","dʒʌm ṯi","mandi kuti","dʒʌm kuti","dʒom pa","dʒom pa","mandi kuti","etʌm ṯi, dʒʌm ṯi","dʒom","etom, dʒom ṯi","mandi pe",None,"mundi pa","dʒom pa","dʒomdʒom pa","dʒom","dʒodʒom kuti","dʒʌm ṯi","dʒʌdʒoŋ","mandi kuṯi",None,"dʒom ṯi","dʒodʒom","dʒʌdʒʌsetʔ","ɖahano"]),
},
118:{
139:("left",["1 leŋga, 3 kondʒe","1 liŋga pa","1 liŋga pa","1 liŋga ṯi","1 liŋga kuti","1 liŋga kuti","1 liŋga pa","1 liŋga pa","1 liŋga kuti","1 liŋga ṯi, 3 kone ṯi","1 leŋga","1 leŋga ṯi, 3 koɲe","1 liŋga pa",None,"1 liŋga pa","1 liŋga pa","1 liŋgʌ pa","1 liŋga","1 liŋga kuti","1 leŋga ṯi","1 liŋga","1 liŋga kuṯi",None,"1 leŋga","1 leŋga","1 liŋga setʔ","2 ba·mo"]),
140:("near",["1 dʒapaʔa, 3 naeʔe","1 dʒʌpa","1 dʒʌpa","1 dʒʌpa","1 dʒʌpaʔa","1 dʒʌpʌʔa","1 dʒʌpa","1 dʒʌpaʔa","1 dʒʌpa","1 dʒʌpo","1 dʒʌpa","1 dʒapa, 3 nae","1 dʒʌpa","3 naite","1 dʒapaʔa","1 dʒʌpa","1 dʒʌpa","1 dʒapa","6 sube","1 dʒapa","6 subʌ","6 sube","3 naredʒ","4 nipaṯ","5 sor","5 sur","2 pak·o"]),
141:("far",["1 saniŋ, saŋgin","1 sʌniŋ","1 sʌŋgin","1 saniŋ","1 sʌniŋ","1 saniŋ","1 sʌŋgin","1 sʌŋgin","1 sʌniŋ","1 saniŋ","1 sʌŋgiŋ","1 saŋiŋ","1 saniŋ","1 saŋiŋ","1 sagin","1 sʌŋgin","1 saŋgin","1 sʌŋgin","1 saŋgiŋ","1 saŋgi","1 saŋgin","1 saŋgin","1 saŋiŋ","1 saŋin","1 saŋgin","1 sʌŋgi","2 ɖuro"]),
},
119:{
142:("big",["1 maraŋ","1 mʌrʌŋ","1 mʌrʌŋ","1 mʌrʌŋ","1 mʌrʌŋ","1 mʌrʌŋ","1 mʌrʌŋ","1 mʌrʌŋ","1 mʌrʌŋ","1 mʌrʌŋ","1 mʌraŋ","1 maraŋ","1 maraŋ",None,"1 maraŋ","1 mʌrʌŋ","1 maraŋ","1 mʌraŋ","1 mʌraŋ","1 maraŋ","1 maraŋ","1 mʌrʌŋ",None,"1 maraŋ","1 maraŋ","1 mʌrʌŋ","2 boro"]),
143:("small",["1 huɖiŋ","1 huɖiŋ","1 huɖiŋ","1 huɖiŋ","1 huɖiŋ","1 huɖiŋ","1 huɖiŋ","1 huɖiŋ","1 huɖiŋ","1 huɖiŋ","1 uriŋ","1 huɖiŋ","1 huɖiŋ",None,"1 huɖiŋ","1 huɖiŋ","1 huɖiŋ","1 horiŋ","1 huɖiŋ","1 huɖiŋ","1 huɖiŋ","1 huɖiŋ",None,"1 huriŋ","1 huɖin, 3 katitʃ?","1 huɖiŋ","2 sanõ"]),
144:("heavy",["1 hambal","1 hʌmbal","1 hʌmbal","1 hʌmbal","1 hʌmbal","1 habal","1 hʌmabal","1 hʌmbal","1 hʌmbal","1 habal","1 hʌmbal","1 hʌmbal","1 hʌmbal",None,"1 hambal","1 hʌmbal","1 hʌmbal","1 hambʌl","1 hʌmbala","1 hambal","1 hʌmbal","1 hʌmbal",None,"1 hamal","1 hamal","1 hamal","2 bʰari"]),
},
}

def main():
 with LEDGER.open(encoding="utf-8",newline="") as f: rd=csv.DictReader(f,delimiter="\t"); fields=rd.fieldnames; rows=list(rd)
 assert fields
 expected={(p,i,s) for p,items in PAGES.items() for i in items for s in SITES}; seen=set()
 for row in rows:
  p=int(row["PDF_Page"]); i=int(row["Item"]); s=row["Site_Code"]; key=(p,i,s)
  if key not in expected: continue
  gloss,forms=PAGES[p][i]; form=forms[SITES.index(s)]
  special=bool(form and any(c in form for c in "̱ʌɖʒ·ʔəʰɲãõʷɪ"))
  vals={"Gloss":gloss,"Manual_Transcription":form or "","Review_Status":"attested" if form else "blank","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey-alphabet glyphs" if special else "","Reviewer_Method":"manual-source-image; rendered-180dpi; OCR-not-accepted","Reviewed_At":"2026-08-28"}
  if row["Review_Status"]=="unreviewed": row.update(vals)
  else:
   for k,v in vals.items():
    if row[k]!=v: raise AssertionError(f"ledger conflict p{p}/{i}/{s}/{k}")
  seen.add(key)
 if seen!=expected: raise AssertionError(f"page topology drift {len(seen)} != {len(expected)}")
 for p,items in PAGES.items():
  for i,(gloss,forms) in items.items(): assert len(forms)==len(SITES),(p,i,gloss,len(forms))
 with LEDGER.open("w",encoding="utf-8",newline="") as f: wr=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); wr.writeheader(); wr.writerows(rows)
 print(f"recorded {len(expected)} manually reviewed cells for PDF pages 112-119 (items 121-144)")
if __name__=="__main__": main()
