import json
from pathlib import Path
P=Path(__file__).resolve().parent
done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
held={}
for f in sorted(P.glob('pass*-decisions.json')):
 if f.name=='pass118-decisions.json':continue
 for x in json.loads(f.read_text()).get('held',[]):held[x['record']['ID']]=x
rules=[
 ('Compare śītala','12487','CDIAL[12487 and addenda];CDIAL[12613]','User adjudication selects śītala (12487) for these regional cold forms. Turner gives Gujarati śīḷũ and Marathi śiḷā under śītala; *śaitala (12613), with e-vowel comparanda, remains an explicit alternative. The survey e vowel alone does not resolve historical vṛddhi versus regional lowering. This is the approved family assignment, retaining that uncertainty and the recorded regional lateral variants.'),
 ('11366 explicitly','11366','CDIAL[11366 and addenda];CDIAL[11363]','User adjudication selects vartman (11366) for these bāṭ/vāṭ-family path responses. Turner explicitly lists the relevant Himalayan, eastern and western path families here, while allowing some feminine forms to derive equally from vartis (11363). That alternative remains in this note: the survey normally does not supply gender evidence that would exclude it.'),
 ('retained seed','9261','CDIAL[9261 and addenda]','User adjudication selects bījya (9261) for these Hajong, Bishnupriya and Bengali seed responses with retained affricate or sibilant. Turner documents retained-j seed comparanda under bījya, including Torwali biž and Himalayan bīj. Applying that family to these eastern survey forms is the reviewed analysis, not a claim that Turner explicitly prints each eastern form.'),
 ('Hajong/Bishnupriya jui','5300','CDIAL[5300];CDIAL[6606,addendum]','User adjudication selects jyotis (5300) for Hajong/Bishnupriya jui/jī fire, compared with Turner’s Assamese zūi and Bengali jũi fire. The alternative dyuti (6606), whose addendum also gives Assamese jui fire, remains explicitly noted. The chosen parent does not imply that this competing analysis has been disproved.')]
a=[];counts={}
for needle,parent,citation,evidence in rules:
 group=[x for i,x in held.items() if i not in done and needle in x['reason']];counts[parent]=len(group)
 for x in group:a.append(dict(record=x['record'],parent=parent,kind='reflex',citation=citation,evidence=evidence))
assert counts=={'12487':39,'11366':30,'9261':23,'5300':16},counts
assert len({x['record']['ID'] for x in a})==108
(P/'pass118-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=1)+'\n');print(counts)
