import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text())
done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
 dict(parent='9408',gloss=['good'],forms=['bala','bʰolo','bhalo','bʰɔl'],languages=['Hajong','Or','Bhatri','B','AdivasiOriya'],citation='CDIAL[9408]',evidence='Full bhalla explicitly lists Bengali bhāla, Oriya bhala/bhalā and Assamese bhāl “good”, with regional short and long vowels. These eastern survey forms continue that good-adjective family. Turner calls the shortening irregular and compares bhadra through *bhadla; the family link retains that qualification and leaves regional borrowing unresolved.'),
 dict(parent='5654',gloss=['few'],forms=['tānīk','t anak','tanak'],languages=['SSr','bundeli_atarra'],citation='CDIAL[5654]',evidence='Full tanu/tanuka explicitly includes Hindi tanuk/tanak/tanik “small”. The selected few/quantity responses preserve that whole adjective, with source vowel length and internal transcription space retained; the ordinary small-to-few use is compatible with the entry’s Pali thin/small senses. Longer -yākā and nīkanā forms are not reduced to this stem.'),
 dict(parent='7150',gloss=['small'],forms=['nikā / niqaṛā','nikā / nika'],languages=['awan'],citation='CDIAL[7150]',evidence='Full nikta gives its Middle Indo-Aryan replacement *nikka, Lahnda nikkā “small, young”, Khetrani nikṛā and Western Pahari nikṛo “small”. Both members of each selected Awankari slash response belong to this same nikkā/nikṛā family, so the complete response is covered; mixed nikkā plus choṭā/nanḍā responses remain excluded.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and r['Language_ID'] in g['languages']:
   a.append(dict(record=r,parent=g['parent'],kind='reflex',citation=g['citation'],evidence=g['evidence'],family=i))
held=[]
for r in rs:
 if r['Gloss']=='mother' and any(s in r['Form'] for s in ['pirā','pīrā','birā','bʰāī','bʰirā','pʰirā','pʰāī','baṛo','vada','vadā','badā','vaḍā','hālī','hāli','mhāli','mhālī','mhāḷi']):
  held.append(dict(record=r,candidates=[],reason='This mother response contains what appears to be additional kinship vocabulary or a longer phrase (often big/older brother). It cannot be treated as a simple mother reflex without checking source row/column alignment. Preserve the source-defined record and hold; source transcription and identity repairs are outside this no-rebuild etymology pass.'))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass015-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
(P/'pass015-decisions.json').write_text(json.dumps({'accepted':a,'held':held},ensure_ascii=False,indent=1)+'\n')
print(len(a),[(g['parent'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)],'held',len(held))
