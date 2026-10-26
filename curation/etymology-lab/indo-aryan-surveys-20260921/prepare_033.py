import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
 dict(parent='2462-2',gloss=['one'],forms=['ekʰ / īka','hik','ekʰ','yakʰ','eko','e̩k','uk','hekʰ','eke'],citation='CDIAL[2462,2]',evidence='Turner groups Lahnda hekk/hikk, Punjabi ikk/hekka, northwestern akh/yakh, Oriya eka and Gujarati ek/eko in the retained-k ēkka branch. These simple survey numerals, including both Awankari ekʰ/īka alternatives, match that family with regional vowels, aspiration and inflection. Turner leaves emphatic doubling versus early Sanskrit reborrowing unresolved; this link does not resolve that prehistoric alternative.'),
 dict(parent='6984',gloss=['nine'],forms=['naɔ̃ / nɔ̃'],citation='CDIAL[6984]',evidence='Turner gives Lahnda nõ and Awankari naɔ̃, Punjabi naũ and the northwestern nasal-vowel series under nava nine. Both Gujari slash alternatives are the same numeral family, with diphthong versus contracted vowel; neither contains an extra classifier.'),
 dict(parent='6227',gloss=['ten'],forms=['da / dā','daso','daç','dʰəs'],citation='CDIAL[6227]',evidence='Turner lists Awankari dā, Oriya dasa and daś throughout Western Pahari under daśa ten. The survey responses match those regional numeral stems, with local vowel, sibilant and aspiration variation; both Awankari alternatives are the same form. The separate dā/das response is excluded because Turner marks Punjabi das as a Hindi loan.'),
 dict(parent='12278',gloss=['one hundred'],forms=['saho','xɔv','hɔv','sɔ','ço','hov','haū'],citation='CDIAL[12278]',evidence='Turner records Oriya sa/sae, Western Pahari śau and Gujarati sɔ in the śata hundred family. These whole survey hundred responses fit the regional weakened-intervocalic-t vowel series; the western h/x onsets are compared with the s/h variation elsewhere in the same Bhili surveys. The mapping is comparative, without claiming that every onset and glide is independently established as a regular sound change.'),
 dict(components=['2462-2','12278'],gloss=['one hundred'],forms=['ek haū','ek hõ','ek-ho'],citation='CDIAL[2462,2];CDIAL[12278]',evidence='The complete response is one + hundred: the initial ek matches Turner’s ēkka family, and haū/hõ/ho matches the western śata hundred series (compare Gujarati sɔ, Marathi śẽ and regional h-onset survey forms). The hyphenated and spaced answers are represented with two ordered components; no single ancient compound or source-spelling change is asserted.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms']:
   a.append(dict(record=r,**({'components':g['components']} if 'components' in g else {'parent':g['parent'],'kind':'reflex'}),citation=g['citation'],evidence=g['evidence'],family=i))
assert len(a)==len({x['record']['ID'] for x in a})
h=[]
for r in rs:
 if r['ID'] not in done and r['Gloss']=='ten' and r['Language_ID'] in ['Bhili','Bhilali','PauriBareli','Noiri','DungraBhili','Vasavi']:
  h.append(dict(record=r,reason='The doh/ḍoho/ḍoh family plausibly continues daśa (6227), but daśaka (6228) explicitly has Sindhi ḍaho for the number ten as well as a group of ten. These western forms need closer regional evidence to distinguish the simple numeral from the old collective formation.'))
(P/'pass033-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass033-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g.get('parent',g.get('components')),sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
