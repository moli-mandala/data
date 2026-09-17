"""Helpers for explicitly reviewed pending decisions; no accepted-data mutations."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
NAMES={'Malvi':'mewari_basad','Nimadi':'Nimadi','Bagheli':'bagheli_lakshman'}
class Batch:
 def __init__(self,number):
  self.number=number;self.parents=json.loads((P/'parents.json').read_text());self.inv={l:json.loads((P/f'{l}-inventory.json').read_text()) for l in NAMES};self.proposals={l:[] for l in NAMES};self.used=set();self.start={}
  for lang,lid in NAMES.items():
   old=[]
   for f in (P.parent/lid).glob('batch-*.json'):
    d=json.loads(f.read_text())
    if d.get('researchDirectory')==str(P):old+=d['proposals']
   self.used.update(i for x in old for i in x['formIds']);self.start[lang]=max([x['number'] for x in old],default=0)
 def add(self,lang,gloss,words,parent,evidence,tier='straightforward',locator=None,kind='reflex',source_glosses=None,citation=None,source_url=None):
  words=words.split('|');allowed=source_glosses if source_glosses is not None else [gloss]
  rows=[r for r in self.inv[lang] if r['Gloss'] in allowed and r['Form'] in words]
  assert rows,(lang,gloss,words);assert all(r['ID'] not in self.used for r in rows),(lang,gloss,words)
  assert self.parents[parent],parent
  requested_parent=parent
  resolved=self.parents[parent].get('id',parent)
  self.parents.setdefault(resolved,self.parents[parent])
  parent=resolved
  self.used.update(r['ID'] for r in rows);n=self.start[lang]+len(self.proposals[lang])+1
  cite=citation or 'CDIAL['+(locator or requested_parent.replace('-','.',1))+']'
  x={'number':n,'status':'pending-review','difficulty':tier,'formIds':[r['ID'] for r in rows],'forms':list(dict.fromkeys(r['Form'] for r in rows)),'gloss':gloss,'records':rows,'parentId':parent,'parentForm':self.parents[parent]['word'],'kind':kind,'citation':cite,'evidence':evidence,'assignments':[{'Form_ID':r['ID'],'Etymon_ID':parent,'Kind':kind,'Rank':'1','Status':'accepted','Source':cite,'Notes':evidence+f' Pending central-survey proposal {lang} {n}.','Pos':''} for r in rows]}
  if source_url:x['primarySourceURL']=source_url
  senses=list(dict.fromkeys(s for r in rows for s in r['Gloss'].split('; ')))
  x['sourceSenses']=senses
  if len(senses)==1 and gloss!=senses[0]:x['groupingLabel']=gloss;x['gloss']=senses[0]
  if kind=='borrowed':x['parentLanguage']=self.parents[parent].get('language',{}).get('name',self.parents[parent].get('language_id',''))
  self.proposals[lang].append(x)
 def save(self):
  paths={l:P.parent/lid/f'batch-{self.number:03d}.json' for l,lid in NAMES.items() if self.proposals[l]}
  assert all(not f.exists() for f in paths.values()),'Refusing to overwrite existing batch'
  for lang,f in paths.items():
   payload={'language':NAMES[lang],'survey':lang,'batch':self.number,'status':'pending-review','scope':json.loads((P/'inventory-summary.json').read_text())[lang],'proposals':self.proposals[lang],'deadline':'2026-09-11T15:30:00Z','researchDirectory':str(P)}
   f.write_text(json.dumps(payload,ensure_ascii=False,indent=2));print(lang,len(self.proposals[lang]),sum(len(x['formIds']) for x in self.proposals[lang]))
