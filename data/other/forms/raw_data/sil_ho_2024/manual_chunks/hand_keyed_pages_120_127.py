#!/usr/bin/env python3
"""Emit OCR-blind, explicitly hand-keyed Ho decisions for PDF pp120--127."""
from __future__ import annotations
import csv
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "pages_120_127_hand_keyed.tsv"
SITES = "HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
METHOD = "manual-source-image; rendered-400dpi; OCR-not-accepted"
REVIEWED_AT = "2026-08-28"

def f(text: str) -> list[str | None]:
    out = [None if x == "-" else x for x in text.split("\t")]
    assert len(out) == 27
    return out

DATA = {
145:("light",f("1 la·r, 3 labar\t3 lʌbar\t3 lʌbar\t3 lʌbar\t1 lar\t3 lʌbar\t3 lʌbar\t3 lʌbar\t3 lʌbar\t1 lar, 3 lʌbar\t3 labar\t1 lar\t1 lar\t-\t1 lar\t3 lʌbar\t3 labar\t3 rʌbal\t3 lʌbar, rʌbal\t3 rʌbal\t3 rʌbal\t3 rʌbal\t-\t-\t3 rawal, 4 marsal\t3 rʌwal\t2 haluka")),
146:("above",f("1 tʃeṯan\t1 tʃiṯan\t1 tʃiṯan\t1 tʃiṯan\t1 tʃiṯan\t1 tʃiṯʌn\t1 tʃiṯan\t1 tʃiṯan\t1 tʃiṯan\t1 tʃiṯan\t1 tʃiṯʌn\t-\t1 tʃiṯan\t-\t1 tʃiṯan\t1 tʃiṯan\t1 tʃeṯan\t1 tʃiṯan\t1 tʃiṯan\t1 tʃeṯan\t1 tʃilʌn\t1 tʃigʌn\t-\t1 tʃeṯan\t1 tʃeṯan, 3 tʃot\t1 tʃiṯan, 3 tʃʌt\t2 uporo")),
147:("below",f("1 laṯar, 3 suba\t1 lʌṯar\t1 lʌṯar\t1 laṯar\t1 lʌṯar\t1 lʌṯʌr\t1 lʌṯar\t1 lʌṯar\t1 lʌṯar\t1 lʌṯʌr\t1 laṯar\t1 laṯare, 3 subare\t1 lʌṯar\t-\t1 lʌṯʌr\t1 lʌṯʌr\t1 laṯar\t1 laṯar\t1 lʌṯar\t1 lʌṯar\t1 lʌṯar\t1 lʌṯar\t-\t1 laṯar\t1 laṯar, 4 pʰed\t1 lʌṯar\t2 ṯolo")),
148:("white",f("1 puṇɖi\t1 puṇɖi\t1 puṇɖi\t1 puṇɖi\t1 puṇɖi\t1 puṇɖi\t1 puṇɖi\t1 puṇɖi\t1 puṇɖi\t1 puṇɖi\t1 puṇɖi\t1 puṇɖi\t1 puṇɖi\t-\t1 puṇɖi\t1 puṇɖi\t1 puṇɖi\t1 puṇɖi\t1 puṇɖi\t1 puṇɖi\t1 puṇɖi\t1 puṇɖi\t-\t1 puṇɖi\t1 poṇɖ\t1 pu·ṇɖ\t2 ɖʰola")),
149:("black",f("1 heṇɖe\t1 hʌṇɖe\t1 heṇɖe\t1 heṇɖe\t1 heṇɖe\t1 heṇɖe\t1 heṇɖe\t1 hʌɖe\t1 heṇɖe\t1 heṇɖe\t1 heṇɖe\t1 heṇɖe\t1 heṇɖe\t-\t1 heṇɖe\t1 heṇɖe\t1 heṇɖe\t1 heṇɖe\t1 heṇɖe\t1 heṇɖe\t1 heṇɖe\t1 heṇɖe\t-\t1 heṇɖe\t1 heṇɖe\t1 heṇɖe\t2 koɭa?")),
150:("red",f("1 ara?a, 3 dʒeŋga\t1 ʌra\t1 ʌra\t1 ara?a\t1 ʌra, 3 dʒiŋga\t1 ara?a\t1 ʌra?a\t1 ʌra\t1 ʌra?\t3 dʒiŋga\t1 ara\t-\t1 ara?a\t-\t1 ara?a\t1 ʌra?\t1 ara\t1 ara\t1 ʌra?a\t4 raŋga\t1 ara?a\t4 raŋga\t-\t1 ara\t1 arak\t1 ara?\t2 nali, 4 roŋgo")),
151:("one",f("1 mijad, 3 mid\t1 mijʌd?\t1 mijʌd\t1 mien\t1 mijʌd\t1 mien\t1 mijad\t1 mijd\t1 mijʌd\t1 mied\t1 mijʌd?\t1 miad, 3 mid\t1 mijen\t1 miad\t1, 3 miet\t1 mijad\t1 mijʌn\t1 mijʌṇɖ?\t4 mudʒed\t1 mien\t4 mudʒe?d\t5 mõe\t1 miad\t1 miaɖ\t3 mit\t3 miti\t2 eko")),
152:("two",f("1 bar, barija\t1 bʌrie\t1 bʌrie\t1 barie\t1 barie\t1 barie\t1 barie\t1 barie\t1 barie\t1 birie\t1 bari·\t1 baria, bar\t1 barie\t1 baria\t1 barie\t1 barie\t1 bari?ʌ\t1 bariʌ\t1 barie\t1 barie\t1 barie\t1 barie\t1 barie\t1 baria, bar\t1 bar\t1 baria\t2 du·i")),
153:("three",f("1 apija, ape·\t1 ʌpie\t1 ʌpie\t1 apie\t1 ʌpie\t1 apie\t1 ʌpie\t1 ʌpie\t1 ʌpie\t1 apie\t1 ʌpi·\t1 apea, ape\t1 apie\t1 apia\t1 apie\t1 ʌpie\t1 ʌpi?ʌ\t1 ʌpiʌ\t1 ʌpie\t1 apie\t1 ʌpie\t1 apie\t1 apie\t1 apie\t1 pea\t1 peja\t2 ṯini")),
154:("four",f("1 upunija, upun\t1 upunie\t1 upunie\t1 upunie\t1 upunie\t1 upunie\t1 upunie\t1 upunie\t1 upunie\t1 upunie\t1 upunie?\t1 upunia, upun\t1 upunie\t1 upunia\t1 upunie\t1 upunie\t1 upuni?ʌ\t1 upuniʌ\t1 upunie\t1 upunie\t1 upunie\t1 upunija\t1 upunia\t1 upun\t1 pon\t1 upunie\t2 tʃari")),
155:("five",f("1 moja\t1 moja\t1 moja\t1 mõja\t1 moja\t1 mõja\t1 moja, moŋeja\t1 moja\t1 moja\t1 mõŋea, mõja\t1 mõnie?\t1 moja\t1 mõja\t1 moŋja\t1 mõŋea\t1 moŋeja\t1 mõŋi?ʌ\t1 moniʌ\t1 moŋeja\t1 moŋeja\t1 mõŋẽa\t1 mõŋoja\t1 moŋea\t1 moŋrea\t1 mõrẽ\t1 mõŋe\t2 pantʃo")),
156:("six",f("1 ṯuruija, ṯurui\t1 ṯurie\t1 ṯurie\t1 ṯurie\t1 ṯurie\t1 ṯurue\t1 ṯurie\t1 ṯurie\t1 ṯurie\t1 ṯuruje\t1 ṯurie?\t1 ṯuruia, ṯurui\t1 ṯuruije\t1 ṯuruia\t2 tʃʌ\t1 ṯurie\t1 ṯuri?ʌ\t1 ṯuriʌ\t1 ṯurie\t2 tʃo\t1 ṯurie\t1 ṯuruja\t1 ṯuria\t1 ṯuria, ṯuruia\t1 ṯurui\t1 ṯurui\t2 tʃʰe")),
157:("seven",f("1 aija, ai\t1 aie\t1 aie\t1 aije\t1 aie\t1 aije\t1 aie\t1 aie\t1 aie\t1 aije\t1 ajija\t1 ae, aea\t1 aije\t1 aja\t2 saṯ\t2 saṯ\t2 saṯʰ\t1 aje\t2 saṯ\t2 saṯ\t2 sʌṯ\t2 saṯ\t1 ea\t1 ea\t1 eae\t1 ijaje\t2 saṯo")),
158:("eight",f("1 irleja, iril\t1 irilija\t1 irilie\t1 irlia\t1 irilija\t1 irlia\t1 irilija\t1 irilija\t1 irilija\t1 irlia\t1 ilija\t1 irilia, iril\t1 irel\t1 irilia\t2 aṯ\t2 aṯ\t2 aṯʰ\t1 ilijʌ\t2 aṯ\t2 aṯ\t2 aṯ\t2 aṯ\t1 iralia\t1 iralia, irilia\t1 iral\t1 irel\t2 aṯʰo")),
159:("nine",f("1 areja, are·\t1 arija\t1 arija\t1 aria\t1 arija\t1 aria\t1 arija\t1 arija\t1 arija\t1 aria\t1 arija\t1 area, are\t1 areja\t1 area\t2 nʌ\t2 no\t2 nʌ\t1 arijʌ\t2 no\t2 no\t2 nʌ\t2 no\t1 area\t1 area\t1 are\t1 are\t2 nao")),
160:("ten",f("1 geleja, gel\t1 gelija\t1 gelija\t1 gelia\t1 gelija\t1 gelia\t1 gelija\t1 gelija\t1 gelija\t1 gelia\t1 gelija\t1 gelea, gel\t1 gelea\t1 gel\t2 ɖʌs\t2 ɖos\t2 ɖʌs\t1 gelijʌ\t2 ɖos\t2 ɖʌs\t2 ɖʌs\t2 ɖʌs\t1 gelna\t1 gelea, gel\t1 gel\t1 gel\t2 ɖaso")),
161:("eleven",f("-\t1 gel mijʌd\t1 gel mijʌd\t1 gel mien\t1 gel mijʌd\t1 gel mien\t1 gel mijʌd\t1 gel mijʌd\t1 gel mijʌd\t1 gel mien\t1 gel mijʌd?\t1 gel miad, gel mi\t1 gel mijen\t-\t2 egar\t2 egijaro\t2 egaro\t1 gel mijʌd?\t2 eg garo\t2 egar\t2 egar\t2 gʌra\t-\t-\t1 gel mit\t1 gel mit?\t2 egaro")),
162:("twelve",f("1 gel bar\t1 gel barie\t1 gel barie\t1 gel birie\t1 gel barie\t1 gel barie\t1 gel barie\t1 gel barie\t1 gel barie\t1 gel barie\t1 gel bari·\t1 gel baria, gel bar\t1 gel barie\t-\t2 bar\t2 baro\t2 baro\t1 gel bariʌ\t2 baro\t2 bar\t2 bar\t2 bara\t-\t-\t1 gel barea\t1 gel bar\t2 baro")),
163:("twenty",f("1 hisi\t1 isi\t1 hisi\t1 hisi\t1 isi\t1 hisi\t1 hisi\t1 hisi\t1 isi\t1 hisi\t1 isi\t1 hisi\t1 hisi\t1 hisi\t2 koɖie\t2 koɖie\t2 kuri\t1 hesi\t1 hisi\t2 koɖie\t2 koɖie\t3 bis\t1 hisi\t1 hisi\t1 isi\t1 isi\t2 kori·e")),
164:("one hundred",f("-\t1 moi hisi\t2 mit so\t2 mi sʌ\t1 moja isi\t2 mi sʌ\t1 moja isi, moŋe isi, 2 mi so\t1 moja isi, 2 mi so\t2 mit so\t1 mõŋe hisi\t2 miṯ? so\t1 moi hisi\t2 mid so\t3 sau\t3 sʌe\t2 mit so\t4 moŋe kuri\t1 moŋ?e hesi\t2 mod so\t2 mit? so\t2 mod sʌ\t2 mot sʌ\t1 mone hisi, 3 saj\t2 mid sae\t3 sae\t2 mit? sae\t3 eko saho")),
165:("who?",f("1 okoe\t1 okoe?\t1 okoe\t1 okoe\t1 okoe\t1 okoe\t1 okoe\t1 okoe\t1 okoe\t1 okoe\t1 okoe\t1 okoe\t1 okoe\t1 okoi\t1 okoe\t1 okoe\t1 okoi\t1 okoe\t1 okoja\t1 okoe\t1 okoe\t1 okoe\t1 okoi\t1 okoe\t1 okoe\t1 okoe\t2 kie")),
166:("what?",f("1 tʃikana, 1, 3 tʃina\t1, 3 tʃina\t1, 3 tʃina, 1 tʃikʌna\t1, 3 tʃina\t1 tʃikʌna\t1, 3 tʃina?a\t1 tʃikʌna\t1 tʃikʌna\t1, 3 tʃina, 1 tʃikʌna\t1, 3 tʃina\t1, 3 tʃi·na?\t1 tʃikana\t1, 3 tʃina\t1, 3 tʃina\t1, 3 tʃina?a\t1, 3 tʃina\t1 tʃikena\t2 kaŋa\t3 tʃiem\t2 kana\t1, 3 tʃia\t1 tʃikana\t1, 3 tʃina\t1, 3 tʃina\t4 tʃet?\t4 tʃet?\t2 kɔno?")),
167:("where?",f("1 okonre, okonṯe, okonpa\t1 okonre\t1 okonre\t1 okonṯe\t1 okonpa\t1 okonṯe\t1 okonre\t1 okonre\t1 okonre\t1 okonṯe\t1 okonre\t1 okonre, okonṯe, okonpare\t1 okonpa\t-\t1 okonṯe, okonpa\t1 okonre\t1 okonpa\t1 okonre\t1 okosa\t1 okoṯe\t1 okosa\t1 oka\t-\t1 okonreko\t1 oka\t1 okaset?\t2 keuntare, kuade")),
168:("when?",f("1 tʃuila, 3 tʃimiṯa\t3 tʃimiṯe\t3 tʃimiṯe\t1 tʃuile\t1 tʃuile\t1 tʃuile\t1 tʃuile\t1 tʃuile\t3 tʃimiṯe\t1 tʃuile\t1 tʃuile\t1 tʃuila\t3 tʃimiṯeŋ\t-\t1 tʃuile\t1 tʃuile\t3 tʃimiṯam\t3 tʃimiṯʌŋ\t3 tʃimiṯe\t3 tʃimiṯem\t-\t3 tʃumṯa\t-\t3 tʃimiṯa\t4 tisre, 5 kʰan\t4 ṯis\t2 kebe")),
}

FIELDS="Item Gloss Site_Code PDF_Page Printed_Page Column Manual_Transcription Review_Status Confidence Uncertainty Reviewer_Method Reviewed_At Reviewer_Declaration".split()

def main():
    rows=[]
    for item in range(145,169):
        gloss,forms=DATA[item]
        page=120+(item-145)//3
        for index,(site,form) in enumerate(zip(SITES,forms)):
            ambiguous=item==167 and site=="HKE"
            special=bool(form and any(c in form for c in "ʌʃʒŋɖṇɭ̱·ʰɔẽõ?"))
            row={
                "Item":item,"Gloss":gloss,"Site_Code":site,"PDF_Page":page,
                "Printed_Page":page-9,"Column":"left" if index<14 else "right",
                "Manual_Transcription":form or "",
                "Review_Status":"ambiguous" if ambiguous else ("attested" if form else "blank"),
                "Confidence":"low" if ambiguous else ("medium" if special else "high"),
                "Uncertainty":("overwritten/struck source cell; tentative reading 1 okonṯe; independent re-review required" if ambiguous else ("diplomatic Unicode rendering of legacy survey glyphs" if special else "")),
                "Reviewer_Method":METHOD,"Reviewed_At":REVIEWED_AT,"Reviewer_Declaration":DECLARATION,
            }
            rows.append({key: unicodedata.normalize("NFC", str(value)) for key,value in row.items()})
    assert len(rows)==648 and len({(r["Item"],r["Site_Code"]) for r in rows})==648
    with OUT.open("w",encoding="utf-8",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=FIELDS,delimiter="\t");w.writeheader();w.writerows(rows)
    print(f"wrote {len(rows)} explicit hand-keyed cells")

if __name__=="__main__": main()
