"""Source abbreviation decisions, checked against pp. XIII–XIX and Jambu registry."""
import csv,re,unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
# A third field names a variety that must be represented by a registered dialect.
SPEC='''A A
Aw Aw
B B
B.chit B Chittagong
B.roh B Rohingya
Ap Ap
AMg jmag
As Asuri
Aś As
Ash Ash
Av Av
Bal Bal
Balt Balti
Bashg Bashg
Bau WPah Bauri
Bagh bagheli_lakshman
Bar unresolved
Barā WPah Barari
Bghṭ WPah Baghati
Bhad bhad
Bhal bhal
Bhaṭ bhatr
Bhaṭe bhat
Bhiḍ bhad Bhidlai
Bhil Bhili
Bhil.MBh Bhili
Bhil.wag Wagdi
BHS BHSk
Bhoj Bhoj
Bhum unresolved
Bi Bi
Bir Birhor
Bng WPah Bangani
Bon re
Brah Brahui
Brj Brj
Brj.-Aw unresolved
Br.-Aw unresolved
Bro bro
Bshk Bshk
Bun bfu
Bund bundeli_atarra
Bur Bur Hunza
Bur.ng Bur Nagar
Bur.ys Bur Yasin
Bush WPah Bushahari
Cam cam
Chak unresolved
Chant unresolved
Chatt NDu
Chep unresolved
Chil Chil
Chin cih
Chit unresolved
Chu unresolved
Ḍ D
D D
Dara unresolved
Dard Dard
Dari unresolved
Dash Bshk Dashwa
Deg Pas Degano
Deog WPah Deogari
Desh Pk Deshya
Desh.M M Deshya
Dhū-kaṛ L Dhundi-Kariali
Dir Bshk Dir
Ḍog dog
Dm Dm
Eur eur
G G
G.pars G Parsi
G.saur Saurashtra
Ga Gadaba
Gadd ga
Gan Dhp
Gar bfu
Garh Garh
Garh.nā Garh Nagpuriya
Garh.pau Garh Pauri
Garh.r Garh Rathi
Garh.ṭ Garh Tihriyali
Gau Gowro
Gaw Gaw
Glaṅ Gng
Gmb Gmb
Gtaˈ gt
Gu gu
Guj Goj
H H
H.bgh bagheli_lakshman
H.chg NDu
H.kau H Kauravi
Haṇḍ WPah Handuri
Him WPah Himachali
Hind awan
Hind.man awan Mansehra
Hind.pesh awan Peshawar
Ho ho
Ind Mai
Ind.dub Mai Duber
Ind.ky Mai Kanyawali
Ind.seo Mai Seo
Insir insir
Ishk Ishk
Jaḍ unresolved
Jaṭ jhang
Jaun jaun
Jav Av
JM jmh
JŚ Pk Jaina-Shauraseni
Ju ju
Jur Juray
K K
K.ḍoḍ dod
K.pog pog
K.rām ram
K.sir sir
Kal Kal
Kal.rumb Kal Rumbur
Kal.urt Kal Urtsun
Kalk Kalk
Kamd Kam
Kan Kannada
Kana unresolved
Kann unresolved
Kc kc
Kgr kgr
Kh kh
Khand Khandesi
Khas unresolved
Khaś khash
Khaśdh WPah Khashdhari
Khāśi WPah Khashi
Khet khet
Kho Kho
Kho.kiv Kho Kivi
Khot Khot
Khm Kjl
Khz Chvar
Kiś K Kishtwari
Kiũṭh kiuth
Klm Bshk
Ko Ko
Ko.chr Ko Christian
Ko.coch Ko Cochin
Ko.kaṇ Ko Kankon
Ko.kuṇ Ko Kunabi
Ko.S Ko South-Kanara
Koh unresolved
Kod Koda
Kol KolBangladesh
Kor unresolved
Ks Kund
Kt Kt
Kṭg ktg
Kt.g ktg
Kṭgu WPah Kotguru
Kṭkh WPah Kotkhai
Ku Ku
Kul kul
Kurd Kurd
Kurkh Kurux
Kva kvar
L L
Lad Lad
Lagh Pas Laghmani
Lāh unresolved
Lam Bshk Lamuti
Lhd L
L.poṭh poth
M M
M.ber M Berari
M.coch M Cochin
M.hal hal
M.kas M Kasargod
M.kuḍ M Kudali
M.nāg M Nagpuri
M.wār Varli
Ma Malayalam
Mag MagahiNepal
Mah jmh
Maha Mahali
Māl Malw
Malt Malto
Maṇḍ mand
Marw Marw
Marm khash Marmati
Md Md
Mg jmag
Mgr unresolved
Minj Mj
Mj Mj
Mth Mth
Mu mu
Mult srk Multani
N N
New New
Nih Ni
Nim Nimadi
Niṅg Ning
Niy NiDoc
NPers Pers
OAw OAw
OG OG
OMarw OMarw
OP OP
Or Or
Ora Kurux
Orm Orm
Osir WPah Outer-Siraji
Oss Oss
P P
P.poṭ poth
Pa Pa
Pāḍ WPah Padri
Pal Phal
Phal Phal
Palp N Palpa
Pang pan
Par Par
Paš Pas
Paš.ar Pas Areti
Paš.DN Pas Darrai-Nur
Paš.deg Pas Degano
Paš.lauṛ Pas Laurowan
Paš.weg Pas Wegali
Paṭṭ lae
Pers Pers
Pk Pk
Pog pog
Pr Pr
Psht Psht
Pur Prk
R Rosh
Ra unresolved
Rab MH
Rāj Bshk Rajkoti
Rāmb ram
Ramp ramp
Rem re
Rj Rj
Rj.ah mewati_jhambaus
Rj.ajm Rj Ajmeri
Rj.bik Rj Bikaneri
Rj.dng Rj Dang
Rj.har had
Rj.jaip dhundari_bamore
Rj.mal mewari_basad
Rj.mev mewati_akera
Rj.rāj Rj Rajawati
Rj.shekh dhundari_badagaon
Rj.torā Rj Torawati
Roh roh
Rom Gy
Rom.Arm Lom
Rom.As as
Rom.B RomVlax Burgenland
Rom.F Gy Finland
Rom.G RomBalk Greece
Rom.Germ RomSint
Rom.Eur eur
Rom.N Gy Norway
Rom.P as Palestine
Rom.Q unresolved
Rom.S Gy Sweden
Rom.W RomWel
Rp unresolved
Rudh khash Rudhari
S S
S.kcch kcch
Sad Sadri
Sain sai
Saṇ Ash
Sang Sang
Sant sa
Sar Sar
Sã̄s unresolved
Sh Sh
Sh.ast Sh Astor
Sh.chil Sh Chilas
Sh.dras Sh Dras
Sh.gil Sh Gilgit
Sh.gult Sh Gultar
Sh.gur Sh Gures
Sh.jij Sh Jijelut
Sh.koh Sh Kohistani
Sh.kōl Sh Kola
Sh.pales Sh Palesi
Sh.saz Sh Sazin
Sh.tang Sh Tangir
Šaṭ Mai Shatoti
Śeu khash Seuti
Shgh Shgh
Shor WPah Shoracholi
Shum Shum
Si Si
Sir.ḍoḍ sir
Sir.shim WPah Shimla-Siraji
Sir.Suk suk Siraji
Sira srk
Sirm sirm
So so
Sod sod
Sogd Sogd
Sp unresolved
Srk srk
Suk suk
Sun unresolved
Surkh surkh
Sv Sv
Tam Tamil
Td sbu
Tel Telugu
Thak Ths
Thal Bshk Thal
Thar Chitwan
Tib Tib
Tinau L Tinauli
Tir Tir
Tor Tor
Tor.cail Tor Chail
Treg Gmb
Tu Tulu
Tur Turi
Ur H
Ush Ush
Wām Ash Wama
Wan wne
Waz Psht Waziri
Werch Bur Yasin
Wg Wg
Wkh Wkh
Woṭ Wot
WPah WPah
YAv Av
Yazgh Yazgh
Yghn Yghn
Yid Yid
Zb Ishk Zebaki
Zaz unresolved
Zz unresolved'''
MAP={}
for line in SPEC.splitlines():
 code,lang,*dialect=line.split();MAP[code+'.']=(lang,' '.join(dialect))
# Full labels used in prose. External comparison languages stay in the record audit.
for code,lang in [('Sanskrit','Sk'),('OIA','Sk'),('BHS','BHSk'),('Bengali','B'),('Bangani','WPah'),('Baṅgāṇī','WPah'),('Banganī','WPah'),('Hindi','H'),('Mundari','mu'),('Santali','sa'),('Kurukh','Kurux'),('Malto','Malto'),('Kannada','Kannada'),('Tamil','Tamil'),('Telugu','Telugu'),('Tibetan','Tib'),('Bunan','bfu'),('Gondi','excluded'),('Kui','excluded'),('Khmer','excluded'),('Palaung','excluded'),('Sre','excluded'),('Mnong','excluded'),('Katu','excluded'),('Temiar','excluded'),('Nyah Kur','excluded'),('Sedang','excluded'),('Georgian','excluded'),('Greek','excluded'),('PIE','reconstruction'),('PMK','reconstruction'),('PTB','reconstruction'),('Common Kartvelian','reconstruction'),('Proto-Kherwarian','reconstruction')]:MAP[code]=(lang,'Bangani' if code in ('Bangani','Baṅgāṇī','Banganī') else '')
PAT=re.compile(r'(?<![\w.])('+ '|'.join(map(re.escape,sorted(MAP,key=len,reverse=True)))+r')(?=\s|[,;:)—]|$)')
for label,target in {'Baur.':'Bau.','Deś.':'Desh.','Kinn.':'Kann.','Sk.':'Sanskrit','Sor.':'So.','Ho':'Ho.','Tibet.':'Tib.','Dam.':'Dm.','Khāś.':'Khāśi.','Kh.':'Kh.','Korku':'Kor.','Chepang':'Chep.','Newar':'New.','Bodo-Gadaba':'Gu.','Munda':'Mu.','Shina':'Sh.','Khowar':'Kho.','Burushaski':'Bur.','Kalasha':'Kal.','Kham':'Khm.'}.items():MAP[label]=MAP[target]
for label in ['Sherpa','Tang.','LwAd.','Apa.','Kana.','Kann.','Chit.','Thakali','Apatani','Ao','Pochuri','Tangam','Kinnauri','Jad','Tamang']:
 MAP.setdefault(label,('unresolved',''))
for label in ['Lith.','Lat.','Latv.','Arm.','Ar.','Alb.','Fi.','Kv.','Germ.','Hitt.','English','German','Swedish','Gothic','Cymric','Danish','Norwegian','Hittite','Latin','Russian','Proto-Waic','Proto-Katuic','Proto-Palaungic','Proto-Pray-Pram','Proto-North-Bahnaric','Proto-South Bahnaric','Proto-Vietic','Proto-Monic','Proto-Khmuic','Proto-Bahnaric','Proto-Karenic','Proto-Khmer','Alak','Ngeq','Sapuan','Tampuan','Bahnar','Tarieng','Cuac','Kui','Kuy','Jahai','Car','Pnar','Pear','Mon','Bru','Phong','Semelai','Nyah Kur','Palaung','Katuic','Khmu','Khmer','Adyghe','Kabardian','Apkhazian','Ubykh','Svan','Laz','Megrelian','Abkhaz','Mok','Ksingmul','Arem','Maleng','Chrau','Temuan','Semnam','Semaq Beri','Semai','Lawa','Kensiw','Kentaq','Batek','Jah Hut','Kammu','Ksing Mul','Proto-Mon-Khmer','Common Sindy','Common Sind','Common Kartvelian']:
 MAP.setdefault(label,('comparison-control',''))
PAT=re.compile(r'(?<![\w.])('+ '|'.join(map(re.escape,sorted(MAP,key=len,reverse=True)))+r')(?=\s|[,;:)—]|$)')

MAP['Cur.']=('cur','')
MAP['Korku']=('ko','')
MAP['Korwa']=('kw','')
PAT=re.compile(r'(?<![\w.])('+ '|'.join(map(re.escape,sorted(MAP,key=len,reverse=True)))+r')(?=\s|[,;:)—]|$)')
MAP.update({
 'Surin Khmer':('unresolved',''),
 'Munda':('reconstruction',''),
 'West Pahāṛī':('WPah',''),'West Pahārī':('WPah',''),
 'Nepali':('N',''),'Old M.':('OM',''),'Old G.':('OG',''),'Old H.':('OH',''),
 'Jingpho':('unresolved',''),
 'Old B.':('OB',''),'Old Bengali':('OB',''),'Old Si.':('OSi',''),
 'Old Pers.':('OPers',''),'Old Persian':('OPers',''),
 'Middle Pers.':('MPrs',''),'Middle Persian':('MPrs',''),'Old Greek':('Gk',''),
 'Old Mon':('unresolved',''),'Old Georgian':('unresolved',''),'Old Khmer':('unresolved',''),
 'Old English':('unresolved',''),'Old Hungarian':('unresolved',''),'Old Welsh':('unresolved',''),
 'Gtaˈ':('gt',''),'Gtaʔ':('gt',''),'Gta':('gt',''), 'Bondo':('re',''),'Bonda':('re',''),
 'Juang':('ju',''),'Lyngngam':('Lyngngam',''),'Deshi Prakrit':('Pk','Deshya'),
 'D. og.':('dog',''),'D. .':('D',''),'D. hu-kaṛ.':('L','Dhundi-Kariali'),
 'Ho.':('ho',''),'Khas.':('unresolved',''),'Ar.':('Ar',''),'Arm.':('Arm',''),
 'Lat.':('Lat',''),'Lith.':('Lith',''),'Greek':('Gk',''),'Gothic':('Goth',''),
 'Hitt.':('Hit',''),'Russian':('Russ',''),'English':('Eng',''),'Swedish':('Swed',''),
 'Old Irish':('unresolved',''),'Old Nordic':('unresolved',''),
 'Kol.':('unresolved',''),'Kol':('unresolved',''),
 'Mon-Khmer':('reconstruction',''),'Mon- Khmer':('reconstruction',''),
})
PAT=re.compile(r'(?<![\w.])('+ '|'.join(map(re.escape,sorted(MAP,key=len,reverse=True)))+r')(?=\s|[,;:)—]|$)')

MAP['Gāndhārī']=('Dhp','')
