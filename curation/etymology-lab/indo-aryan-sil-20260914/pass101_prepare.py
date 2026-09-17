import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass101';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
def rule(parent,citation,evidence):return dict(parent=parent,citation=citation,evidence=evidence)
rules=[
rule('986','CDIAL[986]','CDIAL 986 asmad gives regional ham we, Oriya ami/āme, Marathi āmhī and Gujarati ame. Selected hamu and ami first-person plural forms fit this family; the final u in hamu is retained as a regional extension. Source inclusive/exclusive labels are preserved without claiming the historical root encoded that distinction. Local Indo-Aryan transmission remains open.'),
rule('10511','CDIAL[10511]','CDIAL 10511 yuṣmad explicitly gives Hindi tum, Assamese/Bengali tumi, Gujarati tamε and Prakrit tumhē with initial t remodelled after tuvam. Selected tum/tumi/tumu/tame second-person forms fit this plural-family stem, with final u and source singular-honorific/plural distinctions retained as regional developments. This does not equate the tum stem with bare tū under 5889; local transmission remains open.'),
rule('986','CDIAL[986];CDIAL[13276]','The full response is first-person pronoun plus all: CDIAL 986 gives eastern ham and Oriya ami; CDIAL 13276 gives eastern sab and Oriya sabu. Two ordered components preserve hamsab or ami səbu, without asserting an inherited Sanskrit compound. The collective element supplies plurality/emphasis; source inclusive labels and local transmission remain open.'),
rule('5889','CDIAL[5889];CDIAL[13276]','The full response is tū/tū̃ plus sab all. CDIAL 5889 explicitly gives eastern tū/tu and nasal second-person variants; 13276 gives sab/sabh all. Two ordered components retain the collective expression and source nasalisation. This analyses the actual tū constituent, not a hypothetical tum stem; the elicited you gloss does not independently specify number or politeness.'),
rule('10511','CDIAL[10511];CDIAL[11119]','The full response is tum/tūm plus log people. CDIAL 10511 gives Hindi tum; 11119 gives Prakrit lōga people and an Old Bengali plural-affix use. Two ordered components preserve this pluralising expression, including source vowel length and word spacing. This is not a single inherited Sanskrit compound and does not settle local Indo-Aryan transmission.'),
rule('986','CDIAL[986];CDIAL[11119]','The full hemlog response is a first-person pronoun plus log people. CDIAL 986 gives regional ham we and 11119 Prakrit lōga people. Two ordered components retain source hem vowel variation and the inclusive elicitation label without claiming an inherited Sanskrit compound or a settled transmission route.')]
acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 f=r['Form'];g=r['Gloss'].lower();i=None;cs=None
 if f in {'hamu','ami'} and g.startswith('we'):i=0
 if f in {'tum','tumi','tumu','tame'} and g.startswith('you'):i=1
 if f in {'hamsab','ami səbu'} and g.startswith('we'):i=2;cs=['986','13276']
 if f in {'tūsab','tū̃sab'} and g.startswith('you'):i=3;cs=['5889','13276']
 if f in {'tūmlog','tumlog','tum log'} and g.startswith('you'):i=4;cs=['10511','11119']
 if f=='hemlog' and g.startswith('we'):i=5;cs=['986','11119']
 if i is not None:
  q=rules[i];x=dict(record=r,parent=q['parent'],family=i,kind='component' if cs else 'reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+f+'.')
  if cs:x['components']=cs
  acc.append(x)
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'pass101_save.py').write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem))
from collections import Counter
print(len(acc),sum(len(x.get('components',[x['parent']])) for x in acc),Counter(x['family'] for x in acc))
