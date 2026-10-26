import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
forms=['aŋgūtʰī','ā̃ŋgūtʰī','yaŋgūtī','aŋgūṭʰī / āŋgūṭʰī','aŋgūṭī','aŋtʰi','āũtʰi','aŋṭᵗi','āᵘnṭʰi','ā̃ᵘṭiyā','aũtʰi','gũṭʰī','gutʰi','guttʰi','agutʰi','gũntʰi','aᵘŋgoṭʰi','aᵘŋguṭʰi']
a=[]
for r in rs:
 if r['ID'] not in done and r['Gloss']=='ring' and r['Form'] in forms:
  a.append(dict(record=r,parent='138-2x',kind='reflex',citation='CDIAL[138, 2]',evidence='Turner’s second aṅguṣṭhya branch, *aṅguṣṭhiya, explicitly includes Punjabi aṅgūṭhī, Nepali/Bihari aũṭhi, Bengali āṅṭi and the Gujarati/Marathi ring forms. These survey items fit that finger-ring family, including contraction or loss of initial a- and local nasal/aspiration variation. Responses with Persian-type -starī or extra unidentified words are excluded.'))
h=[]
for r in rs:
 if r['ID'] not in done and r['Gloss']=='palm' and r['Language_ID']=='Kho' and r['Form']=='pʰān':h.append(dict(record=r,reason='Turner 8045 explicitly considers both pāṇi hand and an Iranian comparison (Wakhi pun), with possible influence of sphāna. Hold the six Khowar phān records rather than silently choosing among these histories.'))
(P/'pass082-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held')
