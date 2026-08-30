#!/usr/bin/env python3
"""Emit OCR-blind, explicitly hand-keyed Ho decisions for PDF pp136--141."""
from __future__ import annotations
import csv, unicodedata
from pathlib import Path
HERE=Path(__file__).resolve().parent; OUT=HERE/"pages_136_141_hand_keyed.tsv"
SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
DECL="hand-keyed-from-rendered-source; OCR-not-copied"; METHOD="manual-source-image; rendered-400dpi; OCR-not-accepted"
def f(s):
    x=[None if v=="-" else v for v in s.split("\t")]; assert len(x)==27; return x
DATA={
193:("kill",f("1 goe?e, go?erena\t1 goekie|goime\t1 goekie|goime\t1 goe?|ina|goe?me\t1 goekie|goime\t1 go?e|ina|goe?me\t1 goekie|goime\t1 gojʌkie|goime\t1 gojʌkie|goime\t1 goe|ena|goeme\t1 goe|ija|goime\t1 goe\t1 goek|ije|go?ime\t-\t1 goe?|ina|go?im\t1 goekie|goime\t1 go?|ijʌ|go?ime\t1 goikijʌ|goime\t1 goe|et?dʒie|godʒidʒme\t1 go?e|ia|goe?me\t1 goe|idʒia|godʒidʒme\t1 goe|idijae|godʒidʒime\t-\t1 goe\t1 gotʃ?, 2 marao\t-\t2 mariba")),
194:("fly (bird)",f("1 apir\t1 ʌpir|ina|ʌpirʌme\t1 ʌpir|ina|ʌpireŋme\t1 apir|jana|apireŋme\t1 ʌpir|ijʌna|apirim\t1 apir|jʌna|apireŋme\t1 ʌpir|ina|ʌpireme\t1 ʌpir|ina|ʌpirime\t1 ʌpir|ina|ʌpireŋme\t1 apir|ena|apireŋme\t1 apirʌ|eŋa|apirʌme\t1 apir\t2 uɖo?|ina|uɖo?me\t-\t1 apir|ina, 2 odo?|ina\t1 ʌpir|ijana|ʌpirem\t2 uḏo|eŋʌ|udʌn·e\t1 apir|dʒeŋa|apirme\t2 udodʒanae|udou\t2 oṯaŋjana|oṯaŋme\t2 uɖo?|inae|uɖo?ene\t2 uɖaedʒanae|uɖʌreŋme\t-\t1 apir\t2 uɖau, 3 pʰarkao\t2 uɖʌu|ena|uɖʌu?me\t2 uɖutʃi")),
195:("walk",f("1 sen\t1 sen|ina|seneme\t1 sen|ina|seneme\t1 sen|ena|senome\t1 sen|ina|seneme\t1 sen|ina|seno?me\t1 sen|ina|seno?me\t1 sen|ina|senome\t1 sen|ina|seno?me\t1 sen|ena|sename\t1 sel·ina|dola senem\t1 sen\t1 sen|ina|seno?me\t-\t1 sen|ida|senem\t1 senkida|senem\t1 sen|eŋʌ|seneme\t1 senodʒanʌ|dola (?)\t1 sendʒana|senem\t1 senkiḏa|senḏogome\t1 sen|ene|seno?me\t1 sen|inaja|seno?me\t-\t-\t3 ṯaram, 4 dara\t2 tʃʌlaw|ina, 3 ṯaramme\t2 tʃaliba?")),
196:("run",f("1 nir\t1 nir|ina|nireme\t1 nir|ina|nireme\t1 nir|eda|nireme\t1 nir|ida|nireme\t1 nir|ja|nirme\t1 nir|ida|nireme\t1 nir|ida?|nireme\t1 nir|ina|nirem\t1 nir|eda|nireme\t1 nir|eda|nireme\t1 nir\t1 nir|ida|nirem\t1 nir\t1 nir|ida|nirem\t1 nirkida|nirem\t1 nir|edʌ|nirʌm\t1 nir|eḏa|nirʌme\t1 nir|ida|nirem\t2 dʰaukida|dʰaudem\t1 nir|ina|nireme\t1 nir|ejae|nireme\t1 nir\t1 nir, 2 dauri\t1 nir, 2 dar\t2 ɖʌ·ɖie?ae|ɖʌ·ɖme\t2 doudiba")),
197:("go",f("1 seno?o\t1 sen|ina|seno?me\t1 sen|inae|seno?ome\t1 sen|ina|senome\t1 sen|ina|seno?me\t1 sen|ina|seno?me\t1 sen|ina|seno?me\t1 sen|ina|seno?me\t1 sen|ina|seno?me\t1 sen|ena|senome\t1 sel·ina|seno?om\t1 sen\t1 sen|ina|seno?me\t1 seno\t1 sen|ina|senem\t1 sen|ina|seno?me\t1 sen|eŋʌ|senem\t1 senodʒanʌ|seno?ome\t1 sendʒana|senme\t1 senojana|senem\t1 sen|ina|seno?me\t1 sen|inaje|seno?me\t1 sen\t1 sen\t1 sen, 3 tʃalak?\t3 tʃʌlena|tʃʌlʌ?me\t2 dʒiba")),
198:("come",f("1 hudʒu\t1 hudʒu|ina|hudʒu?me\t1 hudʒu|ina|hudʒu?me\t1 hudʒu|ina|hudʒu?me\t1 hudʒu|ina|hudʒume\t1 hudʒu|ina|hudʒu?me\t1 hudʒu|ina|hudʒu?me\t1 hudʒu|ina|hudʒu?me\t1 hudʒu?|ina|hudʒu?me\t1 hudʒu|ena|hudʒume\t1 hidʒu|ena?|hudʒum\t1 hudʒu\t1 hudʒu?|ina|hudʒu?me\t1 hudʒu\t1 hudʒu|ina|hudʒum\t1 hudʒu|ina|hudʒu?me\t1 hʌjʌ|eŋʌ|hʌjʌ?me\t1 hudʒu|eŋa|hudʒu?me\t1 hidʒ|inae|hidʒu?me\t1 hi·|ena|hidʒu?me\t1 hidʒ|ina|hidʒi?me\t1 hidʒ|inaje|hidʒume\t1 hidʒu\t1 hidʒu\t1 hidʒuk\t1 hedʒ?|ina|hidʒu?me\t2 a·so")),
199:("speak",f("1 kadʒi, 3 men, 5 dʒagar\t1 kadʒi|ida|kadʒime\t1 kadʒikina|kaime\t1 kadʒi|ina|kadʒime\t1 kadʒi|ida|kadʒime\t1 kadʒi|ida|kadʒime\t1 kadʒi|ida|kadʒime\t1 kadʒi|ida|kadʒime\t1 kadʒi|ina|kadʒime\t1 kadʒi|ena|kadʒime\t1 kadʒi|eda|kadʒime\t1 kadʒi, 3 men\t1 kadʒi|ida|kadʒi?me\t-\t1 kadʒi|ida|kadʒim\t1 kadʒi|ida|kadʒime\t1 kadʒi|eḏa|kadʒime\t1 kadʒi|eḏa|kadʒi?me\t1 kadʒikie|kadʒidʒme\t5 dʒagar|a|dʒarem\t1 kadʒi|uia|kadʒidʒme\t1 kadʒi|ijae, 3 menkejae\t-\t1 kadʒi, 3 men\t3 men, 6 ror\t4 lʌ·ilʌ?ae|lelme\t2 kɔhila|kuha")),
200:("hear",f("1 ajum\t1 ʌjum|ida|ʌumeme\t1 ʌjum|ina|ʌumeme\t1 ajum|inda|ajumeme\t1 ʌjum|ida|ʌumem\t1 ajum|ida|ajumem\t1 ʌjum|ida|ʌumeme\t1 ʌjum|ida|ʌumeme\t1 ʌjum|ina|ʌumeme\t1 ajum|ena|ajumeme\t1 a?jum|eda|a?jumeme\t1 aium\t1 ajum|ida|ajumem\t-\t1 ajum|ida|ajum\t1 ʌjum|ida|ʌumem\t1 a|jum|eda|a|jumʌme\t1 ajum|eda|ajumem\t1 ajul|ejaŋ|ajumem\t1 ajumkeda|ajʌlem\t1 ajum|ea|ajumem\t1 ajum|ijae|ajumem\t-\t1 aium\t3 andʒom\t3 ʌndʒʌm|ena|ʌndʒʌme\t2 suno")),
201:("see",f("1 nel, lel\t1 nel|ida|nelem\t1 nelkina|neleme\t1 nel|ina|neleme\t1 lel|ida|leleme\t1 nilaje|nelem\t1 nel|ida|neleme\t1 nel|ida|neleme\t1 nel|idʌŋ|nelem\t1 nel|ena|neleme\t1 ne|eda|nelim\t1 nel\t1 nel|ida|nelem\t-\t1 lel|ida|lelem\t1 nel|ida|neleme\t1 lel|eda|lelem\t1 lel·|ḏa|lel·ime\t1 nel|ejae|nelem\t1 lelkia|lelim\t1 nel|eja|nelem\t1 nel|ejam|nelem\t-\t1 lel, nel\t1 nel\t1 nel|ida|nelme\t2 dekʰo")),
202:("I (1st person singular)",f("1 aŋ\t1 ʌŋ\t1 ʌŋ\t1 aŋ\t1 ʌŋ\t1 aŋ\t1 ʌŋ\t1 ʌŋ\t1 ʌŋ\t1 aŋ\t1 aŋ\t1 aiŋ, 3 iŋ\t1 aŋ\t1 aiŋ, 3 iŋ\t1 aŋ\t1 ʌŋ\t1 aŋ\t1 aŋ\t3 iŋ\t1 aŋ\t3 iŋ\t3 iŋ\t1 aiŋ, 3 iŋ\t1 aiŋ, 3 iŋ\t3 iŋ\t3 iŋ\t2 mũ")),
203:("you (2nd person singular informal)",f("1 am\t1 ʌm\t1 ʌm\t1 am\t1 ʌm\t1 am\t1 ʌm\t1 ʌm\t1 ʌm\t1 am\t1 am\t1 am\t1 am\t1 am\t1 am\t1 ʌm\t1 ʌm\t1 am·\t1 am\t1 am\t1 am\t1 am\t1 am\t1 am\t1 am\t1 am\t2 ṯu")),
204:("you (2nd person singular formal)",f("1 aben\t-\t-\t1 aben\t1 ʌben\t1 aben\t1 ʌben\t1 ʌben\t-\t1 aben\t1 aben\t-\t1 aben\t-\t1 aben\t1 ʌben\t-\t1 aben\t1 ʌben\t1 aben\t1 aben\t1 aben\t-\t-\t1 aben\t1 abin\t3 aponõ")),
205:("he (3rd person singular masculine)",f("1 a?e, 3 ini?i\t1 ʌ?e\t1 ʌ?e\t1 a?e\t1 ʌ?e\t1 a?e\t1 ʌ?e\t1 ʌ?e\t1 ʌ?e\t1 a?e\t3 ini?i\t1 ai\t1 a?e\t1 aj\t1 a?e\t1 ʌ?e\t1 a?e\t3 ini?\t3 ini\t1 a?e\t3 ini?i\t1 aṯ?\t3 ini?\t1 ae\t3 uni\t3 uni\t2 se")),
206:("she (3rd person singular feminine)",f("1 a?e\t1 ʌ?e\t1 ʌ?e\t1 a?e\t1 ʌ?e\t1 a?e\t1 ʌ?e\t1 ʌ?e\t1 ʌ?e\t1 a?e\t3 ini?i\t1 ai\t1 a?e\t-\t1 a?e\t1 ʌ?e\t1 a?e\t3 ini?\t3 ini\t1 a?e\t3 ini?i\t1 aṯ?\t-\t-\t3 uni\t3 uni\t2 se")),
207:("we (1st person plural inclusive)",f("1 abu\t1 ʌbu\t1 ʌbu\t1 abu\t1 ʌbu\t1 abu\t1 ʌbu\t1 ʌbu\t1 ʌbu\t1 abu\t1 abu\t1 abu\t1 abu\t1 abu\t1 abu\t1 ʌbu\t1 ʌbu\t1 abu\t1 ʌbu\t1 abu\t1 abu\t1 abu\t1 abu\t1 abu\t1 abo\t1 abu\t2 ampe, ame")),
208:("we (1st person plural exclusive)",f("1 ale\t1 ʌle\t1 ʌle\t1 ale\t1 ʌle\t1 ale\t1 ʌle\t1 ʌle\t1 ʌle\t1 ale\t1 ale\t1 ale\t1 ale\t1 ale\t1 ale\t1 ʌle\t1 ale\t1 al·e\t1 ʌle\t1 ale\t1 ale\t1 ale\t1 ale\t1 ale\t1 ale\t1 ale\t2 ampe, ame")),
209:("you (2nd person plural)",f("1 ape\t1 ʌpe\t1 ʌpe\t1 ape\t1 ʌpe\t1 ape\t1 ʌpe\t1 ʌpe\t1 ʌpe\t1 ape\t1 ape\t1 ape\t1 ape\t1 ape\t1 ape\t1 ʌpe\t1 ape\t1 ape?\t1 ʌpe\t1 ape\t1 ape\t1 ape\t1 ape\t1 apea\t1 ape\t1 ape\t2 aponõ")),
210:("they (3rd person plural)",f("1 ako\t1 ʌko\t1 ʌko\t1 ako\t1 ʌko\t1 ako\t1 ʌko\t1 ʌko\t1 ʌko\t1 ako\t1, 3 inko\t1 ako, 3 akiŋ\t1 ako\t1 ako, 3 akiŋ\t1 ako\t1 ʌko\t1 ako\t1 ako\t1 ʌko\t1 ako\t1, 3 inku\t1 ako\t1, 3 inku, 3 inkiŋ\t1 ako, 3 akiŋ\t3 unkiŋ, 1 onko\t1 ako\t2 se man·e")),
}
FIELDS="Item Gloss Site_Code PDF_Page Printed_Page Column Manual_Transcription Review_Status Confidence Uncertainty Reviewer_Method Reviewed_At Reviewer_Declaration".split()
def main():
 rows=[]
 for item in range(193,211):
  gloss,forms=DATA[item];page=136+(item-193)//3
  for i,(site,form) in enumerate(zip(SITES,forms)):
   special=bool(form and any(c in form for c in "ʌʃʒŋɖṇɭ̱·ʰɔẽõ?")); row={"Item":item,"Gloss":gloss,"Site_Code":site,"PDF_Page":page,"Printed_Page":page-9,"Column":"left" if i<14 else "right","Manual_Transcription":form or "","Review_Status":"attested" if form else "blank","Confidence":"medium" if special else "high","Uncertainty":"diplomatic Unicode rendering of legacy survey glyphs" if special else "","Reviewer_Method":METHOD,"Reviewed_At":"2026-08-28","Reviewer_Declaration":DECL};rows.append({k:unicodedata.normalize("NFC",str(v)) for k,v in row.items()})
 assert len(rows)==486 and len({(r["Item"],r["Site_Code"]) for r in rows})==486
 with OUT.open("w",encoding="utf8",newline="") as h:w=csv.DictWriter(h,fieldnames=FIELDS,delimiter="\t");w.writeheader();w.writerows(rows)
 print("wrote 486 explicit hand-keyed cells")
if __name__=="__main__":main()
