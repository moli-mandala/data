import json
from pathlib import Path
P=Path(__file__).resolve().parent;rs=json.loads((P/'remaining-initial.json').read_text());d={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())};a=[]
for r in rs:
 if r['ID'] in d:continue
 x=None
 if r['Gloss']=='heart' and r['Form'] in ['man','mon','mɔn']:
  x=dict(parent='9822',kind='reflex',citation='CDIAL[9822];platts1884[s.v. man]',evidence='Turner manas supplies the regional man mind family. Platts explicitly derives Hindi man from Prakrit maṇo and Sanskrit manas and includes heart among its senses. The surveyed man/mon forms are compared with this family; their elicited heart sense is preserved, without claiming that all varieties distinguish anatomical and emotional senses identically.')
 if r['Gloss']=='mouth' and r['Form'] in ['ṭõḍ','t oṇḍ','ṭoṇḍ','ṭonḍ','tonḍə','tõd','tõḍ','toṇḍ','tɔṇḍɨ','tɔṇḍ']:
  x=dict(parent='5853',kind='reflex',citation='CDIAL[5853,3,7]',evidence='Turner tuṇḍa section 3 *tōṇḍa explicitly gives Marathi tõḍ and Konkani toṃḍa mouth; section 7 records initial retroflex variants, and the introduction discusses internal dental-to-retroflex assimilation. These regional o-vowel mouth forms are compared with that family. The existing canonical node is the unsplit 5853; no nonexistent section node is invented, and source-internal spacing is retained.')
 if x:a.append(dict(record=r,**x))
(P/'pass103-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=1)+'\n');print(len(a))
p=P/'lexical-reference-links.json';x=json.loads(p.read_text());x['platts1884[s.v. man]']='https://www.rekhta.org/urdudictionary?keyword=man';p.write_text(json.dumps(x,ensure_ascii=False,indent=1)+'\n')
