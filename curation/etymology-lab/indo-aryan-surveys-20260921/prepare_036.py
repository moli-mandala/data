import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
 dict(parent='10178',gloss=['many'],forms=['muc','mūc'],languages=['awan','Goj'],citation='CDIAL[10178];shackle-auto[p. 243]',evidence='Shackle’s printed muccu entry (p. 243, directly inspected) gives abundant/plenty, big/broad and very, and explicitly prints CDIAL 10178 mucca as its etymology. This supplies the semantic bridge missing from Turner’s shorter lump/heap and Punjabi piece entries. The Awankari/Gujari muc/mūc adjectives are qualified regional members of that family; the comparison does not establish the extent of local historical diffusion.'),
 dict(parent='9498',gloss=['different'],forms=['bʰinā̃i','bʰinde'],citation='CDIAL[9498]',evidence='Turner gives Punjabi bhinn difference, Assamese/Bengali bhin separate and Maithili bhīn under bhinna different. Dotyali bhinā̃i and Majhi bhinde match the bhin- adjective with local adverbial/inflectional endings. The d-bearing Majhi form is a qualified comparison rather than a claim of a regular sound law.'),
 dict(components=['9498','9498'],gloss=['different'],forms=['bin.e bin.e','bʰin.e bʰin.e'],citation='CDIAL[9498]',evidence='The Bhatri response repeats the bhin-/bin- adjective different, with its local -e ending. Turner explicitly attests Punjabi bhinn-bhinn separate/apart, beside eastern bhin. Both occurrences are represented as ordered components of a distributive repetition; no half of the source response is discarded.'),
 dict(parent='700',gloss=['different'],forms=['ɔlag'],citation='CDIAL[700]',evidence='Turner gives Old Marwari alago, Gujarati aḷgũ and Hindi alag separate under alagna. Rathwi ɔlag matches that regional unconnected/separate adjective; no extra component is present.'),
 dict(components=['700','700'],gloss=['different'],forms=['alge alge','ɔlɔg ɔlɔg','agalagal','agālagāl'],citation='CDIAL[700]',evidence='Both parts repeat the regional alag/agal adjective separate, from alagna (Turner: Hindi alag/algā, Oriya alaga, Old Marwari alago and Gujarati aḷgũ). The metathesized Mewari agal forms are compared with the same regional adjective and retain their source spelling. The full distributive repetition is recorded with two ordered components.'),
 dict(parent='f_gtfibpl3lt2qo',kind='borrowed',gloss=['near'],forms=['neṣḍīk','nesḍīk','nazīq / nazdīk'],citation='platts[s.v. nazdīk];kannauji[p. 89]',evidence='Platts identifies Hindi/Urdu nazdīk near as Persian in origin, while the installed Hindi attestation supplies an immediate donor node. These regional n-z-d-k/contracted variants are recorded as borrowings through a provisional Hindi/Urdu donor proxy; the precise local transmission route is unresolved. Both Awankari alternatives match the same loan family, not two inherited stems.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and ('languages' not in g or r['Language_ID'] in g['languages']):
   a.append(dict(record=r,**({'components':g['components']} if 'components' in g else {'parent':g['parent'],'kind':g.get('kind','reflex')}),citation=g['citation'],evidence=g['evidence'],family=i))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass036-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=1)+'\n')
(P/'pass036-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
l=json.loads((P/'lexical-reference-links.json').read_text());l['s.v. nazdīk']='https://www.rekhta.org/urdudictionary?keyword=nazdiik';(P/'lexical-reference-links.json').write_text(json.dumps(l,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',[(g.get('parent',g.get('components')),sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
