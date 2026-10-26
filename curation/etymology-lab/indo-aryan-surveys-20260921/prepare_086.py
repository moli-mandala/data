import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
(dict(parent='8330',gloss='whole',forms=['pūrā','puṛī','puṛa','pʰura','pūrū̃','pūrū','purā','puru','pūḷo'],evidence='Turner pūra explicitly records Punjabi/Lahnda pūrā full, Nepali puro complete, Oriya purā and Gujarati pūrũ full. These whole/unbroken survey responses match that adjective family, preserving local aspiration and r/ṛ/ḷ spellings. Nasal-final pūrṇ-/puron-like responses are excluded because their learned versus inherited history and possible old/whole sense confusion need separate review.')),
(dict(parent='13355',gloss='whole',forms=['sārā'],evidence='Turner sāra explicitly records Hindi/Punjabi sārā whole and Old Marwari sāro all/complete/whole. The Kaithal sārā answer matches this family. It is distinguished from sarva, despite the overlapping all/whole senses.')),
(dict(parent='136',gloss='ring',forms=['āŋgūlī','āŋgūli'],languages=['Mai'],evidence='Turner aṅgulīya finger-ring explicitly cites Maiya aṅgulī. These Maiya survey ring forms match that exact comparandum, with vowel lengths preserved. They are not placed under aṅguli finger merely because the modern spellings are alike.')),
(dict(parent='12045',gloss='ring',forms=['vīṭṭī','vītī'],evidence='Within Turner’s vīṭā entry, the *vīṭṭa family explicitly gives Gujarati vīṭī ring. These neighbouring Bhili/Bareli vīṭṭī/vītī ring forms are compared with that precise subfamily, retaining the recorded consonant spelling. Turner describes a wider expressive/contact family of rounded objects; no unique ultimate origin is asserted.')),
(dict(parent='f_dbuiao2qkdgv4',kind='borrowed',gloss='whole',forms=['hābūt','śyābūt','sābūtā'],citation='platts1884[s.v. s̤ubūt, s̤abūt, p. 368]',evidence='Platts explicitly gives vernacular sabūt entire. These u-vowel whole responses match the previously reviewed regional loan family, with local initial h/ś and adjective endings retained. The existing Hindi sabūt donor is used provisionally, leaving regional intermediaries open; a/i-vowel sābat/sābit responses remain separate.'))]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss']==g['gloss'] and r['Form'] in g['forms'] and ('languages' not in g or r['Language_ID'] in g['languages']):a.append(dict(record=r,parent=g['parent'],kind=g.get('kind','reflex'),citation=g.get('citation',f"CDIAL[{g['parent']}]"),evidence=g['evidence'],family=i))
h=[]
for r in rs:
 if r['ID'] not in done and r['Gloss']=='whole' and r['Form'] in ['puron','pūrūn','pūrɔn','pūraṇ']:
  h.append(dict(record=r,reason='Whole: nasal-final puron/pūraṇ may involve learned pūrṇa rather than pūra, and eastern puron also raises a possible old/whole confusion with purāṇa. The n-final forms are not automatically folded into the transparent pūrā full family.'))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass086-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass086-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g['parent'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
