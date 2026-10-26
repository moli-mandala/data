import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
 dict(parent='1658',gloss=['few'],forms=['ūcūt','ūcūṭʰ','ūṣūt','ūʦ̣ūṭ'],languages=['Bshk'],citation='CDIAL[1658];gawri',evidence='Turner explicitly gives Bashkarik účuṭ, feminine ičiṭ, very small under ucchōṭṭa; the separately installed Gawri dictionary attests úcuṭ few/some. The survey forms preserve the diagnostic initial u plus palatal segment and final retroflex stop. The local sibilant/affricate and aspiration variants are treated as a qualified lexical family, not as separate roots.'),
 dict(parent='248-2',gloss=['below'],forms=['hiṭʰ / hɛṭʰ','eṭh','eṭhā̃','iṭhã','ɛṭhã','eṭā','ethā'],citation='CDIAL[248,2]',evidence='Turner specifically derives Punjabi/Lahnda heṭh, Bhojpuri heṭhā̃ and Gujarati heṭhe below from adhiṣṭāt through MIA heṭṭha/hiṭṭha. The Gujari and Vasavi survey forms preserve the retroflex-t adverb family with variable initial h and final nasal/vowel endings; the slash alternatives share this history. The distinct adhastāt first branch is not selected.'),
 dict(parent='9280',gloss=['below'],forms=['bun','ˈbunne','bəṇṇə','ˈbūne','bōnne','boṇṇe','buṇe','būṇ','būn'],languages=['Goj','kul','poth'],citation='CDIAL[9280,1]',evidence='Turner gives Kangri bunnh/bunh and Western Pahari Bhadrawahi bunh, Bhalesi bùn below under bundha. The Kullui, Pothwari and Gujari bVn adverbs match this neighbouring Indic series, including nasal gemination/retroflexion and locative endings. This is a regional comparison: Turner separately reports Persian-origin analyses for Pashai and Wotapuri bun, and those languages are not included here.'),
 dict(parent='28',gloss=['whole'],forms=['akʰoi','ākhuvo'],citation='CDIAL[28]',evidence='Turner explicitly gives Gujarati ākhũ whole and Marathi akhā/ākhā whole, undivided under akṣata unhurt/unbroken. The Mewari and Noiri survey adjectives match this contracted kh-bearing western family, with inflectional endings; l-expanded whole forms remain excluded because they raise an additional akhila comparison.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and ('languages' not in g or r['Language_ID'] in g['languages']):
   a.append(dict(record=r,parent=g['parent'],kind='reflex',citation=g['citation'],evidence=g['evidence'],family=i))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass035-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=1)+'\n')
(P/'pass035-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',[(g['parent'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
