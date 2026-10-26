import json
from pathlib import Path
P=Path(__file__).resolve().parent;rs=json.loads((P/'remaining-initial.json').read_text());d={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())};a=[];h=[]
plain={'yā','yū','yah','ya','yyā','yyɛ','yā̃','yɜ','yū̃','jyo','jyā','yu','yau','yeh','yya','īya','īā','ī','e','e̤','æ','yo','ye','ye / ya','yo / yā','yā / yo','ye / e'}
for r in rs:
 if r['ID'] in d:continue
 f,g=r['Form'],r['Gloss'];x=None
 if all(v.strip().split(' ')[0] in ['this','these'] for v in g.split(';')) and f in plain:
  x=dict(parent='2530',kind='reflex',citation='CDIAL[2530]',evidence='Turner eṣa/eta explicitly gives Hindi yah, Kumauni/Nepali yo, Maithili ī, Bengali/Oriya e and Punjabi/Lahnda e/eh. The selected proximal survey pronouns are compared with this regional paradigm, retaining local gender/number vowels, y/j notation and breathiness. Distal-looking o/vo responses, hā-/ā- stems and classifier compounds are excluded.')
 if g.startswith('these') and f in ['isab','e̩sʌbu','eisabu','yī sab']:
  x=dict(components=['2530','13276'],citation='CDIAL[2530];CDIAL[13276]',evidence='This plural proximal expression consists of i/e/yi this plus sab/sabu all. Turner eṣa gives the eastern i/e pronouns and sarva the eastern all-forms. Both members are represented in order, including fused source spellings.')
 if x:a.append(dict(record=r,**x))
 if g=='with' and f=='nāl':h.append(dict(record=r,reason='Turner CDIAL 103 tentatively compares Punjabi nāḷ with aṅkapāli embrace, explicitly marking the addendum with a question. The Pothwari cognate is clear, but that remote etymology is not promoted to a firm accepted link.'))
(P/'pass109-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n');print(len(a),len(h))
