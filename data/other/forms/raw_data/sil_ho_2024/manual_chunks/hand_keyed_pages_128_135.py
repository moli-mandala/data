#!/usr/bin/env python3
"""Emit OCR-blind, explicitly hand-keyed Ho decisions for PDF pp128--135."""
from __future__ import annotations
import csv, unicodedata
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/"pages_128_135_hand_keyed.tsv"
SITES="HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
DECL="hand-keyed-from-rendered-source; OCR-not-copied"
METHOD="manual-source-image; rendered-400dpi; OCR-not-accepted"

def f(s):
    x=[None if v=="-" else v for v in s.split("\t")]; assert len(x)==27; return x

DATA={
169:("how many?",f("1 tʃimin, tʃiminaŋ\t1 tʃimin pure\t1 tʃimin pure\t1 tʃimin\t1 tʃimin (pure)\t1 tʃimil\t1 tʃimin pure\t1 tʃimin pure\t1 tʃimin pure\t1 tʃimin\t1 tʃimiŋ\t1 tʃimin\t1 tʃimin pure\t-\t1 tʃimin\t1 tʃimin pure\t1 tʃimeŋ\t1 tʃiminʌŋ\t1 tʃimin\t1 tʃinaŋ\t1, 3 tʃima?\t1, 3 tʃima?\t-\t1 tʃimin\t3 ṯinak\t3 ṯina?\t2 keṯe")),
170:("what kind?",f("1 tʃilikan\t1 tʃilikan\t1 tʃiliken\t1 tʃilike rʌkʌm\t1 tʃilikene\t1 tʃimin pʌrkar\t-\t-\t1 tʃilikan\t1 tʃilike rʌkʌm, tʃilike\t1 tʃiliken·a\t-\t1 tʃilikan\t-\t1 tʃilikana\t-\t1 tʃilikʌ\t1 tʃʌlka\t1 tʃilikana, 3 tʃimin prakar\t-\t1 tʃilekʌn\t1 tʃilikana\t-\t-\t1 tʃekan lekan\t1 tʃet?lekana\t2 kemiṯi")),
171:("this (in hand)",f("1 neja, nena\t1 nija\t1 nija\t1 nina\t1 nina\t1 nina\t1 nija\t1 nija\t1 nina\t1 nina\t1 nija\t1 nea\t1 nia\t-\t1 nia\t1 nija\t1 nẽja\t1 nija\t1 nija ṯed\t1 nia\t1 neã\t1 nia\t-\t1 nea\t1 nia, 3 noa\t3 no?e\t2 eiṯa")),
172:("that (distant)",f("1 hani·, ena, ini\t1 hʌna\t1 hʌna\t1 hana\t1 hʌna\t1 hana\t1 hana\t1 hana\t1 hana\t1 ene\t1 hanʌ\t1 ena\t1 hana\t-\t1 hana\t1 hʌna\t1 hʌna, hʌne\t1 hʌnʌ\t1 hana\t1 hana\t1 hʌ·e\t1 hana, ina\t-\t1 ena\t1 ona, one\t1 hʌne\t2 seiṯa")),
173:("these (in hand)",f("1 neko, 1, 3 nejako, 3 nenako\t1, 3 nijako\t1, 3 nijako\t3 ninak\t1 neko\t3 ninako\t3 ninako\t1, 3 nijako\t3 ninako\t3 ninako\t3 ninako\t1 neko\t1, 3 niako\t-\t1, 3 niako\t1, 3 nijako\t1, 3 nẽjako\t1, 3 nijako\t1, 3 nijako\t1, 3 niako\t1 nẽ?eko\t1 niko\t-\t1, 3 neako, 1 niku\t1 noko, 3 noako\t1, 3 ho?eko\t2 eisabu")),
174:("those (distant)",f("1 hanko, 3 enko\t1 hanako\t1 hʌnako\t1 hanako\t1 hanko\t1 hanako\t1 hanako\t1 hʌnako\t1 hanako\t1 hanako, 3 enku\t1 hanʌko\t1 enko\t1 hanako\t-\t1 hanako\t1 hʌnako\t1 hanʌko\t1 hanʌko\t1 hʌnako\t1 hanako\t1 hʌ·eko\t1 hanako\t-\t1, 3 einiko\t1, 3 onko\t1 hʌneko\t2 seisabu")),
175:("same",f("1 miḏge, 3 leka, 4 miḏ\t3 lika\t3 lika\t1 miṯgia\t1 midgia\t1 midgie\t1 midgie\t1 midgija\t3 midlikʌna\t1 midge\t6 barabari\t1 midge, 4 miḏ\t1 midgia\t-\t1 mid?gia\t1 midgie\t2 soman\t6 bara bʌri\t1 moṯgia, 2 sʌman\t2 sʌman\t1 mod?gea\t2 sʌman\t-\t1 midge\t2 soman, 5 ina\t1 miṯ?gia\t2 səman")),
176:("different",f("1 ṯaŋga, 3 eta·, 5 benga biŋgi\t3 eta\t3 eta\t1 ṯʌŋga\t1 ṯʌŋga\t1 ṯaŋga ṯʌŋga\t1 ṯʌŋga\t1 ṯʌŋga\t4 bigarṯija\t1 ṯʌŋga ṯʌŋga\t1 ṯʌŋʌŋ ṯʌŋʌŋ\t1 ṯaŋga, 3 eta, 8 kilimili\t4 bigar\t-\t3 eta?a eta?a\t4 bigar\t5 benga bengi\t5 bingʌ\t4 begar begar, 6 bʰena, bʰen\t5 binga binga\t4 vegʌr vegʌr\t5, 6 bena bena\t-\t3 eta, 8 kilimili\t7 dʒuda\t6 vinʌvinʌ\t2 alaga, 6 bʰino bʰino")),
177:("whole (unbroken)",f("1 goṯa, 3 saben\t1 gota\t1 gota\t1 gota\t1 gota\t1 goṯaṯica\t1 gota\t1 gota\t1 gota\t1 gota\t-\t3 saben\t1 gota\t-\t1 gota\t1 gota\t4 bẽs giʃa\t1 gota\t1 gota\t1 gota\t4 besṯia\t1 gota\t-\t1 goṯa\t1 goṯar\t1 gota\t2 pura")),
178:("broken",f("1 rapud\t1 rʌpud\t1 rʌpud\t1 raput\t1 rʌpud\t1 rapudṯia\t1 rʌpud\t1 rʌpud\t1 rʌpud\t1 rapuṯ\t1 rapuṯ?\t-\t1 rapu?n\t-\t1 rapu?ṯ\t1 rʌpud\t1 rʌpud?\t1 rapʌṯ?ŋ\t1 rʌpud\t1 rapud?\t1 rapud\t1 rapud\t-\t-\t3 katʃa, 4 ṯuṯa, 5 bʰanga\t1 rʌput?\t2 baŋgila")),
179:("few",f("1 huḏiŋ, 3 dʒokano\t3 dʒokoe\t3 dʒ?okoe lika\t3 dʒoka\t1 hudiŋ\t1 huḏiŋ\t3 dʒoka\t3 dʒoka\t1 hudiŋ\t3 dʒoka\t1 uriŋ\t-\t1 hudiŋ\t-\t1 huduŋ, 2 kʌm\t1 huḏiŋ\t1 hudiŋ leka\t1 uriŋ\t9 hʌŋga\t1 hudi?\t1 hudiŋ\t8 aŋga\t-\t1 huduriŋ\t4 tʰora gan, 5 eka, 6 duka\t7 keṯi?\t2 kom")),
180:("many",f("1 pura·, 3 saŋgi, 4 esu\t1 pure\t1 pure\t1 pure\t1 pure\t1 pure\t1 pure\t1 pure\t1 pure\t1 pure\t1 pure?\t1 pura\t1 pure\t-\t1 pure?e\t1 pure\t1 purʌ?\t3 saŋgi\t1 pure, 10 bidʒen\t9 dʒaṯka\t5 ɖeher, 10 bedʒʌŋ\t10 bedʒaŋ\t-\t4 isu, 5 ɖʰer, 8 an hut\t6 aema, 7 adi\t5 ɖer\t2 bohut")),
181:("all",f("1 saben, soben, 3 sanam\t1 sʌben\t1 sʌben\t1 sʌben\t1 sʌbin\t1 sʌbin\t1 soben\t1 soben\t1 sʌben\t1 sʌben\t1 sʌbin\t1 saben\t1 sʌben\t-\t2 dʒʌṯʌ\t1 sʌbin\t2 dʒʌṯo\t1 sʌbin\t2 dʒeṯe\t2 dʒʌṯo\t2 dʒʌṯo\t2 dʒʌṯʌ\t-\t1 soben\t2 dʒoṯo, 3 sanam\t2 dʒʌṯʌ?\t1 sobu")),
182:("eat",f("1 dʒom\t1 dʒom|ida|dʒomʌme\t1 dʒom|idʌŋ|dʒomeme\t1 dʒʌ|eda|dʒʌmem\t1 dʒomkidʌŋ|dʒommem\t1 dʒʌm|ida|dʒʌmem\t1 dʒom|ida|dʒommeme\t1 dʒom|ida|dʒommeme\t1 dʒom|ida|dʒomem\t1 dʒʌm|ia|dʒʌmem\t1 dʒom|eḏeŋ|dʒommeme\t1 dʒom\t1 dʒom|i|a|dʒomem\t1 dʒome\t1 dʒʌm|ida|dʒʌmeme\t1 dʒom|ida|dʒommeme\t1 dʒomkeda|dʒomeme\t1 dʒom|eḏa|dʒomʌm\t1 dʒom|ija|dʒomeme\t1 dʒʌm|a|dʒʌmeme\t1 dʒom|ija|dʒʌmeme\t1 dʒʌmkeaj|dʒʌmem\t1 dʒom\t1 dʒom\t1 dʒom\t1 dʒʌm|ija|dʒʌmme\t2 kalba")),
183:("bite",f("1 huwa?a, 3 hab\t1 uwe|ida|uweme\t1 uwe|ida|uweme\t1 hue|ida|hueme\t1 uwekie|uweime\t1 huekie|hue?me\t1 uwe|ie|uwe|me\t1 uwe|ie|uwe|me\t1 uwe|ie|uwe|me\t1 hu|ida|hueme\t1 huwʌ?kie?|huweme\t1 hua, 3 hab\t1 hue|ie|hueme\t-\t1 hue|ida|hue?me\t1 huwe|ie|huwe|me\t1 uwakija|uwa?ime\t1 huwakʌ|huwʌgi?me\t1 hue?kidʒija?|hueeg?me\t1 hua|ia|huame\t1 hue|igi?ṯ|huegidʒi?me\t1 huekijae|huagijẽ\t-\t3 hab\t4 ger, 5 lasok\t4 ger|ija|gereme\t2 ṯsuba|la")),
184:("be hungry",f("1 reŋge?e\t1 reŋge|ida\t1 reŋge|ina\t1 reŋge ṯʌikina\t1 reŋge|inʌŋ\t1 reŋgeṯe kʌlkena\t1 reŋge|ina\t1 reŋge|ida\t1 reŋge|ina\t1 reŋge ṯaikina, reŋge|ida\t1 reŋge reŋṯaikina\t1 reŋge\t1 reŋgeṯe ṯaikena\t-\t1 reŋge ṯʌikena\t1 reŋgekie\t1 reŋge ṯaikʌnʌ\t1 reŋge?i ṯaikʌŋ\t1 reŋgekieŋ\t1 reŋgeṯe ṯaikena\t1 reŋge ṯaikena\t1 reŋgeṯege ṯaikena\t-\t1 reŋge\t1 reŋgetʃ\t1 reŋge ṯahekena\t2 bʰoko hela?")),
185:("drink",f("1 nu·\t1 nui|ida|nuime\t1 nui|ida|nuime\t1 nu|ida|nuime\t1 nui|idʌŋ|nuime\t1 nu|ida|nu|me\t1 nu|ida|nuime\t1 nui|ida|nuime\t1 nui|ida|nuime\t1 nu|ida|nuime\t1 nuleḏa|nujem\t1 nu\t1 nukida|nuime\t-\t1 nu|ida|nuima\t1 nuikida|nuime\t1 nuikinʌ?|nu|me\t1 nuleḏa|nu?ime\t1 nu|i|ʌŋ|nu?me\t1 nu·la|nu|ma\t1 nu|ija|nuiẽme\t1 nukija|nui?ne\t-\t1 nu\t1 nũ\t1 nulija|nũime\t2 pi·ba")),
186:("be thirsty",f("1 ṯeṯaŋ\t1 ṯiṯaŋ|ina\t1 ṯiṯaŋ|ida\t1 ṯiṯaŋ ṯaikina\t1 ṯiṯaŋ|inʌŋ\t1 ṯiṯaŋ ṯaikena\t1 ṯiṯaŋ|ida\t1 ṯiṯaŋ|ida\t1 ṯiṯaŋ|ina\t1 ṯiṯaŋ ṯaikina, ṯiṯaŋ|ida\t1 ṯeṯaŋ|ia\t1 ṯeṯaŋ\t1 ṯiṯaŋ ṯaikena\t-\t1 ṯiṯaŋ ṯaikena\t1 ṯiṯaŋkida\t1 ṯiṯʌŋ|ija\t1 ṯeṯaŋ ṯaikʌnʌ\t1 ṯiṯaŋ ṯadʒie\t1 ṯiṯaŋ ṯaiken\t1 ṯiṯaŋ ṯe ṯaikena\t1 ṯiṯaŋ ṯadʒi ṯaikena\t-\t1 ṯeṯaŋ\t1 ṯeṯaŋ\t1 ṯiṯaŋ ṯahekena\t2 soso hala?")),
187:("sleep",f("1 dʒapiṯ, 3 dum, 4 giṯi?i\t1 dʒʌpid|ina|dʒpidʌme\t1 dʒʌpid|ina|dʒʌpidʌme\t4 giṯi|ina|giṯime\t1 dʒʌpid|idʌŋ|dʒʌpidem\t4 giṯi|ina|giṯime\t1 dʒʌpid|ina|dʒʌpidem\t1 dʒʌpid|ina|dʒʌpidem\t1 dʒʌpid|ina|dʒʌpidʌme\t4 giṯi|ina|giṯime\t4 giṯiŋ|ina|giṯime\t4 giṯi\t1 dʒapi?d|ina, 4 giṯi|ina\t-\t4 giṯi|ina|giṯim\t1 dʒʌpidkida|dʒʌpideme\t4 giṯi|eŋʌ|giṯi?im\t3 durʌm|eŋa|durʌmme?\t4 giṯi|ina|giṯit?me\t4 giṯikena|gi?|me\t4 giṯi?ṯ|ina|giṯi?ṯme\t4 giṯi|ina|giṯit?me\t-\t4 giṯi\t1 dʒapiṯ, 4 giṯitʃ?\t4 giṯi|ina|giṯi?me\t2 nido")),
188:("lie down",f("1 giṯi?i, 3 baṯi\t1 giṯi|ina|giṯime\t1 giṯi|ina|giṯime\t3 baṯi|ina|baṯime\t1 giṯi|inaŋ|giṯime\t1 giṯi|ina|giṯime\t1 giṯi|ina|giṯime\t1 giṯi|ina|giṯime\t1 giṯi|ina|giṯime\t3 baṯi|ina baṯime\t1 giṯi|ina|giṯime\t1 giṯi, 4 burum\t1 giṯi|ina|giṯime\t-\t3 baṯi|balina|baṯi|balenʌm\t1 giṯi|ina|giṯime\t1 giṯi|baleŋʌ|giṯi|hapape\t1 giṯiakʌn ṯaikʌnʌ|giṯime\t1 giṯid|ina|giṯi?me?\t-\t1 giṯi?ṯ|ina|giṯi?ṯme\t1 giṯikene|giṯi?ne\t-\t1 giṯi, 3 baṯin, 4 burum\t1 giṯitʃ\t-\t2 porigola")),
189:("sit down",f("1 ɖub\t1 ɖub|ina|ɖubme\t1 ɖub|ina|ɖubme\t1 ɖup|ina|ɖupme\t1 ɖub|inʌŋ|ɖubme\t1 ɖup|ina|ɖupme\t1 ɖub|ina|ɖubme\t1 ɖub|ina|ɖubme\t1 ɖub|ina|ɖubme\t1 ɖup|ina|ɖupme\t1 ɖup|ina?i|ɖupme\t1 ɖub\t1 ɖub?|ina|ɖub?me\t1 ɖu?b\t1 ɖub|ina|ɖupme\t1 ɖub|ina|ɖubme\t1 ɖub|iŋʌ|ɖub?me\t1 ɖubakan ṯaikʌnʌ|ɖub?me?\t1 duɖup|ina|duɖupme\t1 ɖupkena|ɖupme\t1 ɖub|ina|ɖupme\t1 ɖupkene|ɖupme\t1 ɖu?b\t1 ɖub?\t1 ɖurup?\t1 ɖurup|in|ɖurupme\t2 bosiba")),
190:("give",f("1 em, 3 om\t1 imadiŋe|imaime\t1 imadiŋe|imaime\t1 em|ida|ememe\t1 imadiŋe|imaime\t1 em|ida|imaime\t1 imadiŋe|imaime\t1 imadiŋe|imaime\t1 imadiŋe|imaime\t1 em|ida|ememe, imaime\t3 um|ije|umame\t1 em\t1 em|ie|imaime\t1 eme\t1 em|ida|emem\t1 imadiŋe|imaime\t1 imʌŋiʌ|imeŋdʌ\t3 om|eḏa|umaŋme?\t3 omo?dʒiet|umaŋme\t1 ema|ina|imanme\t3 om|ia?a|omaŋme\t1 emddʒanae|imaene\t1 em\t1 em, 3 om\t1 em\t1 em|ija|imaŋme\t2 ɖeba")),
191:("burn (wood)",f("1 dʒul, 2 lo·, 3 atar\t1 dʒul|ina|dʒuleme\t1 dʒul|ina|dʒuleme\t1 dʒul|ida|dʒuleme\t1 dʒul|ina|dʒuleme\t1 dʒul|ida|dʒuleme\t1 dʒul|ina|dʒul|eme\t1 dʒul|ina|dʒul|eme\t1 dʒul|ina|dʒuleme\t1 dʒul|ida|dʒuleme\t1 dʒul|dʒina|dʒulṯina\t2 lo, 3 atar\t1 dʒol|ina|dʒolem\t-\t1 dʒul|ida|dʒuleme\t1 dʒul|ina|dʒuleme\t1 dʒul|iŋʌ|dʒulo?me\t2 lo|eŋa|loṯ?ŋme\t1 dʒul|ina|dʒulem\t5 sal goʌkida|sal gaeme\t1 dʒul|ia|dʒul|em\t1 dʒul|ijae|dʒul|ime\t-\t2 lo, 3 atar\t2 lo, 4 dʒeret\t1 dʒul|ina|dʒulme\t1 dʒoliba")),
192:("die",f("1 godʒo?o\t1 goe|ina|godʒome\t1 goe|ina|godʒome\t1 goe?jana|godʒʌme\t1 godʒe|jʌna|goeme\t1 go?e|ina|godʒo?me\t1 gojʌna|godʒome\t1 gojʌna|godʒome\t1 gojʌna|godʒome\t1 goe|jana|godʒʌme\t1 ge|eŋa|godʒome\t1 godʒo\t1 goe|ina|godʒo?me\t1 goidʒ\t1 go|in|godʒʌm\t1 gojʌna|godʒome\t1 goe|eŋʌ|godʒome\t1 goedʒanʌ|godʒome\t1 godʒ?dʒana|godʒu?me\t1 goe|jana|godʒʌ?me\t1 goe?|ina|godʒo?me\t1 godʒ?dʒanae|gudʒu?me\t1 goidʒ\t1 godʒo\t1 gudʒuk, gotʃ?\t1 gudʒu|ina|gudʒu?me\t2 moronõ")),
}

FIELDS="Item Gloss Site_Code PDF_Page Printed_Page Column Manual_Transcription Review_Status Confidence Uncertainty Reviewer_Method Reviewed_At Reviewer_Declaration".split()
AMB={(179,"HKA")}
def main():
    rows=[]
    for item in range(169,193):
        gloss,forms=DATA[item]; page=128+(item-169)//3
        for idx,(site,form) in enumerate(zip(SITES,forms)):
            amb=(item,site) in AMB; special=bool(form and any(c in form for c in "ʌʃʒŋɖṇɭ̱·ʰɔẽõ?"))
            row={"Item":item,"Gloss":gloss,"Site_Code":site,"PDF_Page":page,"Printed_Page":page-9,"Column":"left" if idx<14 else "right","Manual_Transcription":form or "","Review_Status":"ambiguous" if amb else ("attested" if form else "blank"),"Confidence":"low" if amb else ("medium" if special else "high"),"Uncertainty":"legacy affricate/glottal glyph sequence not fully resolvable; tentative 3 dʒ?okoe lika" if amb else ("diplomatic Unicode rendering of legacy survey glyphs" if special else ""),"Reviewer_Method":METHOD,"Reviewed_At":"2026-08-28","Reviewer_Declaration":DECL}
            rows.append({k:unicodedata.normalize("NFC",str(v)) for k,v in row.items()})
    assert len(rows)==648 and len({(r["Item"],r["Site_Code"]) for r in rows})==648
    with OUT.open("w",encoding="utf-8",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=FIELDS,delimiter="\t");w.writeheader();w.writerows(rows)
    print("wrote 648 explicit hand-keyed cells")
if __name__=="__main__": main()
