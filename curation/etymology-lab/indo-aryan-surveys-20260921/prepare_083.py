import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
('13095',['hadzlo','hadzḷa','hājo','hādzo','hājɔ','hādza','hādzu','hāḍzla','hādzalo','hāḍzlo'],'Turner sajya/sajja explicitly gives Gujarati sājũ sound/healthy and regional good/in-good-order meanings. These neighbouring Bhili/Bareli/Noiri/Vasavi hāj-/hādz- good responses are qualified comparisons with that health/goodness family, retaining local s-to-h realization and adjectival extensions. The identification does not treat the similar right-hand meaning as the elicited sense.'),
('12637',['sūṇɔ','suṇā','ssuṇo'],'Turner śobhana explicitly documents Lahnda sōhṇā/sōṇhā, Khetrani sunnā, Awankari sōṇā and Punjabi suhuṇā beautiful/pleasing. These Bagri/Kaithal suṇ- good forms match that adjective family with the recorded reduced vowels and nasal retained.'),
('13460',['sūgʰar','suggʰar'],'Turner sughaṭa explicitly supplies Nepali sughar neat/clean, Hindi sughaṛ well formed and Old Maithili sughaḍa handsome. These Tharu good responses match that well-formed/pleasing adjective family; neither su-kara easy nor sugar is inferred from sound alone.'),
('7366',['niman','nīman','nimna'],'Turner nirmala explicitly records Bihari nīman beautiful, Bhojpuri nīman good and Hindi nīman good/healthy. These eastern survey good forms are compared with that clean/good family. The related-looking Hindi niman well-finished under nirmāṇa/nirmiṇoti is distinguished by sense; local contact transmission remains open.'),
('13541',['sotʰrā'],'Turner susthira explicitly supplies Punjabi suthrā good/clean/elegant and Lahnda suthrā excellent/clean. The Kaithal sotʰrā good form matches that northwestern family, with the survey’s vowel retained.'),
('9408',['bol','bɔl'],'Turner bhalla explicitly gives Oriya bhala/bhalā good and Bengali bhāla. These Adivasi Oriya bol/bɔl responses are qualified local deaspirated/contracted forms of that same eastern adjective family; the source spelling is retained.')]
a=[]
for i,(parent,forms,evidence) in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss']=='good' and r['Form'] in forms:a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}]',evidence=evidence,family=i))
h=[]
for r in rs:
 if r['ID'] not in done and r['Gloss']=='good' and r['Language_ID']=='Bishnupriya' and r['Form'] in ['hoba','hɔba']:
  h.append(dict(record=r,reason='Compare śubha 12532 and śobha 12635, but the retained b and vowel do not establish an inherited versus learned route or a unique branch. Hold pending specific Bishnupriya evidence.'))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass083-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass083-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g[0],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
