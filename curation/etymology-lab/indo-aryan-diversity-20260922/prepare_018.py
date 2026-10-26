import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');short=json.loads((P/'kudali-reviewed-shortlist.json').read_text());assigned={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
('akrā','2485','ekādaśa','Turner explicitly gives Marathi akrā eleven under ekādaśa, after MIA egārasa. The reviewed source əkra matches this contracted numeral.'),
('aṭrā','946','aṣṭādaśa','Turner gives Marathi aṭhrā eighteen after MIA aṭṭhārasa/aṭṭhāra. Source əṭra has no aspiration marked; its notation is retained.'),
('aḷśi','198','atasī','Turner section 1 gives Marathi aḷśī/āḷśī linseed via the alasī series. This is distinct from the tissi section and from a similar adjective meaning lazy.'),
('āŋṭi','138-2x','*aṅguṣṭhiya','Turner section 2 of aṅguṣṭhya gives Marathi ãgṭhī thumb- or toe-ring. The feminine source aŋṭi selects the existing reconstructed i-stem node 138-2x, not the bare thumb etymon.'),
('raḍ','10590','raṭati','Turner section 1 explicitly gives Marathi raḍṇẽ and Konkani raḍtā to cry under raṭati. The verbal source rəḍ belongs here rather than the noun raṭana 10591.'),
('sa','12803-3','*kṣaṭ','Turner section 3 gives Marathi sahā/sā and Konkani sa six through MIA cha. Use the existing *kṣaṭ subsection, not the bare ṣaṣ head or the Romani *ṣuvaṭ branch.'),
('hār','14059','hāra','Turner explicitly gives Marathi hār garland beside Prakrit hāra. The source hār matches this noun, distinct from verbs meaning lose.'),
('hiruvo','14114','*hiru','Turner gives Marathi hirvā green/raw/unripe under *hiru. Source hiruvo retains the v-bearing extension and the green sense; the differently formed hiriyo is held.'),
('hivāḷo','14098','*himakāla','Turner gives Marathi hĩvāḷā/hĩvāḷī winter season under *himakāla. Source hivaḷo matches the extended seasonal stem, with its unmarked nasalisation and final o preserved.'),
('hun','2389','uṣṇa','Turner gives Konkani hūna hot and Marathi ūnh/ūn hot/heat of sun under uṣṇa. The source hun hot fits this family; it is a separate record from homonymous hun to say.'),
('humro','2341','umbara','Turner gives Marathi umrā/ũbrā threshold beside Prakrit ummara. Source humro retains mr with initial h; use the unextended umbara node, not the *umbarastha branch for the ground under a threshold.')]
a=[]
for form,parent,label,ev in groups:
 for x in short:
  r=x['record']
  if r['Form']==form and r['ID'] not in assigned:a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}];'+r['Source'],evidence=ev+' Existing source audit explicitly marks the reading image-reviewed. Published transcription and the Kudali (Vengurla) dialect label remain unchanged.'))
reasons={'amvāśā':'Resolve retained mv/ś and learned or contact-mediated history separately from ordinary āvas reflexes of amāvāsyā.','āŋṭo':'The ring search reaches aṅguṣṭhiya, but the thumb requires its own exact parent; not assigned to the ring etymon.','gharto':'The nest form needs an analysis of its extension; not assigned wholesale to ghara house.','laḍ':'Search returns laḍṇẽ meaning cry under raṭati, which does not justify the source fight sense. Hold for its own parent.','hiriyo':'The -iyo form must be distinguished from hiruvo/*hiru and possible harita-family morphology.','hununit':'Extended hot form requires morphology beyond simple hun; no suffix ancestry invented.','hundir':'Mouse candidate discovery did not resolve a sufficiently supported exact parent; do not select by superficial resemblance.'}
held=[dict(record=x['record'],reason=reasons[x['record']['Form']]) for x in short if x['record']['Form'] in reasons and x['record']['ID'] not in assigned]
(P/'pass018-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n');(P/'pass018-source-audit.json').write_text(json.dumps([x for x in short if x['record']['ID'] in {z['record']['ID'] for z in a}],ensure_ascii=False,indent=2)+'\n')
lines=['# Kudali: first diversity batch','',f'{len(a)} records / {len(a)} reflex links, each with an image-reviewed source audit. No rebuild.','','| # | Kudali | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(f,pa,la,ev) in enumerate(groups,1):
 r=next(x['record'] for x in short if x['record']['Form']==f);la=la.replace('*',r'\*');lines.append(f'| {i} | **{f}** ‘{r["Gloss"]}’ | **{la}**, CDIAL **{pa}** | {ev} |')
lines+=['',f'{len(held)} records held:']+[f'- **{x["record"]["Form"]}** ‘{x["record"]["Gloss"]}’: {x["reason"]}' for x in held]+['','Validation and save status: pending.'];(P/'pass018-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(held),'held')
