import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
 dict(parent='8283',gloss=['old'],languages=['Kho','Phal','Mai','Chil','Gowro','Hajong','Or','Bhatri','AdivasiOriya','Bishnupriya','poth'],forms=['pāṛāṇū̃','pāṛāṇõ','parõṇõ','porõ','pūrõ','porā̃','porā','purun','poṛuṇa','puṛuṇa','puṛṇa','puṛuna','purna','punna','prāṇā̃','paṛāṇā̃'],citation='CDIAL[8283]',evidence='Turner explicitly compares Khowar paránu, Palula purōṇo, Kohistani Shina pōṇṷ, Punjabi purānā, Assamese purān and Oriya puruṇā under purāṇa ancient/old. These whole survey adjectives fit that regional series, including shortened Kohistani por-/pur- forms and eastern purun-/purn- forms. The family comparison does not claim a detailed derivation of every nasal vowel or consonant reduction; extended or additional-word responses are excluded.'),
 dict(parent='4850',gloss=['sour'],languages=['Hajong'],forms=['cuku','cukau','cuka'],citation='CDIAL[4850,1]',evidence='The adjective branch of cukra sour includes Bengali cuk/cukā acid, Assamese suksukiyā slightly acid and several northwestern cuk-/cūk- forms. Hajong cuku/cuka/cukau preserves this affricate-plus-velar sour stem with final-vowel variation. The adjective branch is used, rather than the separately defined drink or plant senses; the evidence establishes the regional family, not an independently demonstrated route for each ending.'),
 dict(parent='11175',gloss=['bamboo'],languages=['Hajong','Bishnupriya'],forms=['bas','baha'],citation='CDIAL[11175,1]',evidence='Turner lists Assamese bā̃h and Bengali bā̃s bamboo together under vaṃśa, distinguishing the later vaṃśaka formation. Hajong bas and Bishnupriya baha agree with these eastern s/h bamboo forms. The survey lacks a nasalization mark, which is preserved rather than silently repaired; the s/h contrast alone is not used to assert borrowing from either modern language.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and r['Language_ID'] in g['languages']:a.append(dict(record=r,parent=g['parent'],kind='reflex',citation=g['citation'],evidence=g['evidence'],family=i))
assert len(a)==len({x['record']['ID'] for x in a})
h=[]
for r in rs:
 if r['ID'] in done:continue
 if r['Gloss']=='door' and r['Language_ID'] in ['Gaw','Bshk','Tor','Mai','Chil','Gowro','Phal'] and r['Form'] in ['dar','dār','darā̃','dārā̃','darā̃ū','darā̃ū / darā̃']:
  h.append(dict(record=r,reason='CDIAL 6651 explicitly cautions that some northern dar door forms may instead be Persian loans. The matching dictionary comparandum therefore does not settle inherited *dvara versus borrowing for this survey record; hold rather than encode a categorical inherited parent.'))
 if r['Gloss']=='new' and r['Form'] in ['notun','nautun','nɔtun'] and r['Language_ID'] in ['B','Hajong']:
  h.append(dict(record=r,reason='Primary Bengali dictionaries disagree: Samsad derives natun/notun from Sanskrit navatana, whereas the Accessible Dictionary gives nūtana. CDIAL 7575 does not itself list a Bengali comparandum. Hold the competing parent analysis.'))
(P/'pass026-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass026-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g['parent'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
for x in a:print(x['record']['Language_ID'],x['record']['Form'],x['record']['Gloss'],x['parent'])
