"""Generate reviewable source-label mappings against a pinned Glottolog snapshot."""
import csv,collections,json,re,unicodedata,gzip
from pathlib import Path
from languages import MAP
from west_pahari import promote_mapping
RAW=Path(__file__).parent;ROOT=RAW.parents[4]
def norm(s):return ''.join(c for c in unicodedata.normalize('NFD',s).lower() if c.isascii() and c.isalpha())
rs=list(csv.DictReader(gzip.open(RAW/'glottolog-languages.csv.gz','rt')));byid={r['ID']:r for r in rs};idx=collections.defaultdict(list)
for r in rs:idx[norm(r['Name'])].append(r)
existing=list(csv.DictReader((ROOT/'cldf/languages.csv').open()));ebyid={r['ID']:r for r in existing};eglot=collections.defaultdict(list)
for r in existing:eglot[r['Glottocode']].append(r)
ALIASES={'Cua':'Cua','Jeh':'Jeh','War':'War-Jaintia','Jirel':'Jirel','Sp.':'Spiti Bhoti','Chep.':'Chepang','Chepang':'Chepang','Rp.':'Rongpo','Kann.':'Kinnauri','Kinn.':'Kinnauri','Dari.':'Darai','Khas.':'Khasi','Jaḍ.':'Jad','Tang.':'Tangam','Apa.':'Apatani','Sherpa':'Solu-Khumbu Sherpa','Chit.':'Chitkuli Kinnauri','Kana.':'Kanashi','Ra.':'Raji','Sã̄s.':'Sansi','Chak.':'Chakma','Chant.':'Chantyal','Bhum.':'Bhumij','Khmu':'Khmu','Nyah Kur':'Nyahkur','Sre':'Koho','Gta':'Gtaʔ','Bondo':'Remo','Juang':'Juang','Kol':'Kol (Bangladesh)','Lith.':'Lithuanian','Lat.':'Latin','Latv.':'Latvian','Arm.':'Armenian','Ar.':'Arabic','Alb.':'Albanian','Fi.':'Finnish','Hitt.':'Hittite','Gothic':'Gothic','Tibetan':'Tibetic','German':'Standard German','Swedish':'Swedish','English':'English','French':'French','Russian':'Russian','Persian':'Western Farsi','Khmer':'Central Khmer','Mon':'Mon','Greek':'Ancient Greek','Adyghe':'Adyghe','Megrelian':'Mingrelian','Apkhazian':'Abkhaz','Abkhaz':'Abkhaz','Kui':'Kuay','Pacoh':'Pacoh','Sedang':'Sedang','Korku':'Korku','Burgenland Romani':'Burgenland Romani'}
cs=json.load(open(RAW/'candidates.json'));labels=set(c['label'] for c in cs if c['label'])|set(MAP)
ALIASES['Surin Khmer']='Northern Khmer'
ALIASES['Chang']='Chang Naga'
ALIASES['Old English']='Old English (ca. 450-1100)'
for c in cs:labels.update(c['labels'])
mapping={};new={};unresolved=[]
for label in sorted(labels):
 if label in ('Kol','Kol.','Kor.','Bar.','Brj.-Aw.','Br.-Aw.','Koh.','Semnan','Pahāṛī','Pahari','Pahārī'):
  unresolved.append((label,'Historically ambiguous label; no unique canonical mapping',[]));continue
 old=MAP.get(label,('unresolved',''))
 if old[0] in ebyid:
  mapping[label]={'language':old[0],'dialect':old[1],'basis':'Source abbreviation table and existing canonical registry'};continue
 if label.startswith(('Proto','Common')) or old[0]=='reconstruction':continue
 if label.endswith('.') and label not in ALIASES:
  unresolved.append((label,'Unresolved source abbreviation',[]));continue
 if len(label)<=3 and label not in ('Mon','Hu','En','Ir','Lak','Udi','Cua','Jeh') and label not in ALIASES:
  unresolved.append((label,'Short label requires manual resolution',[]));continue
 name=ALIASES.get(label,label.rstrip('.'))
 matches=idx.get(norm(name),[])
 matches=[r for r in matches if r['Level']!='family' and 'Eurasia' in r['Macroarea']]
 if len(matches)!=1:unresolved.append((label,name,[(r['ID'],r['Name'],r['Level']) for r in idx.get(norm(name),[])]));continue
 g=matches[0];base=byid[g['Language_ID']] if g['Level']=='dialect' else g
 oldbase=eglot.get(base['ID'],[])
 if len(oldbase)==1:ident=oldbase[0]['ID']
 elif len(oldbase)>1:unresolved.append((label,name,[(r['ID'],r['Name']) for r in oldbase]));continue
 else:
  ident=re.sub(r'[^A-Za-z0-9]','',unicodedata.normalize('NFKD',base['Name']))
  if ident in ebyid and ebyid[ident]['Glottocode']!=base['ID']:ident+='_'+base['ID']
  new[ident]={'ID':ident,'Name':base['Name'],'Glottocode':base['ID'],'Latitude':base['Latitude'],'Longitude':base['Longitude'],'Clade':'Other','Location':f"Glottolog language point ({base['Countries']}); modern approximation, not Zoller's field locality",'Quality':'C'}
 mapping[label]={'language':ident,'dialect':g['Name'] if g['Level']=='dialect' else '', 'glottolog':g['ID'],'basis':'Exact name/explicit source-label alias; Glottolog modern language point (quality C)'}
promote_mapping(mapping)
(RAW/'language-map-proposed.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n')
(RAW/'new-languages-proposed.json').write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n')
(RAW/'unresolved-languages.json').write_text(json.dumps(unresolved,ensure_ascii=False,indent=2)+'\n')
print('Mapped labels',len(mapping),'new languages',len(new));print('New:',[(k,v['Glottocode']) for k,v in new.items()]);print('Unresolved:',unresolved[:50])
