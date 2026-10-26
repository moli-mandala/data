import json
from pathlib import Path
P=Path(__file__).resolve().parent;rs=json.loads((P/'remaining-initial.json').read_text());d={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())};a=[]
for r in rs:
 if r['ID'] in d:continue
 f,g=r['Form'],r['Gloss'];x=None
 if all(v.strip().startswith('we ') or v.strip().startswith('we_') for v in g.split(';')) and f in ['āmū','amū','āmu','āmũ','amɛ','am','amu','ame','amo','hemu','assī','assī̃','as','spā','ispā','īspā']:
  x=dict(parent='986',kind='reflex',citation='CDIAL[986]',evidence='Turner asmad explicitly groups the western ame/āmhī, central ham, northwestern as/assī and Khowar ispā pronouns, deriving their modern paradigms from Middle Indo-Aryan nominative, oblique and instrumental forms. The selected survey pronouns fit these regional families, including contracted spā and western am(u) variants. Inclusive/exclusive labels are retained as elicited, not projected onto the ancestor.')
 if g.startswith('long') and f in ['lāmo','lāma','lām','lamo','lamā','lammā','lama','lamba','lāmlo','lāmbalɔ','lāmblū̃','lāmblo','lamlo']:
  x=dict(parent='10951',kind='reflex',citation='CDIAL[10951]',evidence='Turner lamba first branch explicitly includes Kumauni lāmo, Nepali lāmu, Bhojpuri lām and western lamb-/lamm- long/tall forms. These simple survey adjectives match that family, including local l-adjectival extensions. Comparative and longer compound forms are excluded.')
 if g=='breast' and f in ['dūdʰ','dud','dudə','dūdū','duṭ','ḍūḍʰ','dudʰ','dudʰi']:
  x=dict(parent='6391',kind='reflex',citation='CDIAL[6391 and addenda]',evidence='Turner dugdha explicitly includes Oriya dud breast, western Pahari dudhu female breast/breast-milk and Jaunsari dudhū teat. These regional dud(h) breast responses therefore fit an attested milk-to-breast extension, with local deaspiration, devoicing and nominal endings preserved.')
 if x:a.append(dict(record=r,**x))
(P/'pass106-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=1)+'\n');print(len(a))
