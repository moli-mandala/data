import json
from pathlib import Path
P=Path(__file__).resolve().parent;rs=json.loads((P/'remaining-initial.json').read_text());d={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())};a=[];h=[]
for r in rs:
 if r['ID'] in d:continue
 f,g=r['Form'],r['Gloss'];x=None
 if g=='bury' and f in ['gārla','gārlik','gār','gāḍ','gar']:
  x=dict(parent='3979',kind='reflex',citation='CDIAL[3979]',evidence='Turner *gaḍḍ dig/bury explicitly gives Nepali gāṛnu, Maithili gāṛab and Bhojpuri gāṛal. These simple regional survey verbs fit that family, including source r/ḍ notation and finite endings. Responses containing negative particles or a second verb are excluded.')
 if g=='to work' and f in ['kamkora','kam kɔranī','kamkor','kamkorna','kamkura']:
  x=dict(components=['2892','2814'],citation='CDIAL[2892];CDIAL[2814]',evidence='The response consists of kam work plus kor-/kar- do, compared with Turner karman and karoti, both with explicit eastern comparanda. The two lexical members are represented in order, including the Hajong fused spellings and Bishnupriya infinitive ending; no work-only link is substituted for the whole expression.')
 if g=='to work' and f=='kam':x=dict(parent='2892',kind='reflex',citation='CDIAL[2892]',evidence='Turner karman explicitly supplies eastern kām work. This single-word survey response to to work is the noun/root alone, unlike neighbouring kam-kora compounds; no absent do component is invented.')
 if g=='to work' and f in ['kaj kɔra','kadʒ kɔɾa']:
  x=dict(components=['3078','2814'],citation='CDIAL[3078];CDIAL[2814]',evidence='The Bengali expression is kāj work plus karā do. Turner explicitly derives these from kārya and karoti respectively; both components are represented in their surface order.')
 if g=='to steal' and f in ['curi kɔra','ʈʃuɾi kɔɾa']:
  x=dict(components=['4937','2814'],citation='CDIAL[4937];CDIAL[2814]',evidence='The Bengali expression is curi theft plus karā do. Turner caurikā explicitly includes Bengali cori/curi, distinct from the caurya branch, and karoti supplies the second component.')
 if g=='red pepper; chili' and f in ['muris','moric','moris']:
  x=dict(parent='9875-2',kind='reflex',citation='CDIAL[9875,2]',evidence='Turner *maricca, section 2 of marīca, gives Assamese maris, Bengali maric and eastern chilli senses. These Hajong muris/moris/moric forms are compared with that eastern retained-affricate/sibilant family, with vowel variation preserved. The separate *marucca northwestern branch is not inferred from muris alone.')
 if x:a.append(dict(record=r,**x))
 if g=='to steal' and f in ['corkora','cor koranī','curkura','corkor','cori']:
  h.append(dict(record=r,reason='The cor-/cur- component is clear, but cora thief, corayati steal and caurika theft give different exact parents. The unexpanded compound or bare cori does not yet establish which formation the Hajong/Bishnupriya response uses.'))
 if g=='to pull' and f in ['tana','taniva','tane','ʈana']:
  h.append(dict(record=r,reason='Compare dental tanayati 5762 and retroflex *ṭān 5443-2. Turner itself records phonetic dental t for Assamese written ṭāniba. The surveys cannot safely choose a branch from these broad transcriptions alone.'))
(P/'pass105-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n');print(len(a),len(h))
