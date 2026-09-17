"""Re-evaluate source-flag holds after checking the source-local audit.
The flag concerns duplicate locality codes, not lexical readings; all other
phonological/semantic exclusions remain active. No raw source tags are changed.
"""
import json,re,runpy
from pathlib import Path
P=Path(__file__).resolve().parent
specs=[(1,'decisions.json','decide.py','family-candidates.json'),(2,'second-decisions.json','second_decide.py','second-candidates.json'),(3,'third-decisions.json','third_decide.py','third-candidates.json'),(4,'fourth-decisions.json','fourth_decide.py','fourth-candidates.json'),(5,'fifth-decisions.json','fifth_decide.py','fifth-candidates.json')]
allacc=[];allheld=[]
for n,decision,script,candidate in specs:
 d=json.loads((P/decision).read_text());xs=[x for x in d['held'] if x['record']['Language_ID']=='Rana' and 'Tharu-RNS-' in x['record']['Tags'] and ('uncertainty marker' in x['reason'].lower() or 'uncertainty marker' in x['reason'])]
 if not xs:continue
 cn=f'resolve-rana-candidates-{n}.json';on=f'resolve-rana-decisions-{n}.json';(P/cn).write_text(json.dumps([{'record':x['record'],'families':x['families']} for x in xs],ensure_ascii=False,indent=1))
 s=(P/script).read_text().replace("P/'"+candidate+"'", "P/'"+cn+"'").replace("P/'"+decision+"'", "P/'"+on+"'")
 lines=s.splitlines()
 for i,line in enumerate(lines):
  if ('Source/dialect uncertainty marker:' in line or 'An uncertainty marker occurs in the source/dialect tags.' in line) and 'reason=' in line:
   prefix=line[:len(line)-len(line.lstrip())];cond='elif' if line.lstrip().startswith('elif') else 'if';lines[i]=prefix+cond+' False: reason=None  # Locality-only flag resolved below.'
 sf=P/f'resolve_rana_pass_{n}.py';sf.write_text('\n'.join(lines)+'\n');runpy.run_path(str(sf));out=json.loads((P/on).read_text())
 for x in out['accepted']:
  x['evidence']+=' The source-local Webster audit reports zero ambiguous lexical readings; the RNS uncertainty concerns assigning two occurrences to Sisaikhara versus Sisana, both Rana Tharu. That locality uncertainty is preserved in the source tags and does not affect this shared lexical analysis.'
  x['reviewPass']=n
 for x in out['held']:x['passNumber']=n
 allacc+=out['accepted'];allheld+=out['held']
(P/'rana-resolution-decisions.json').write_text(json.dumps({'accepted':allacc,'held':allheld},ensure_ascii=False,indent=1));print('Resolved supported',len(allacc),'still held for other reasons',len(allheld))
