import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text())
done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
 dict(parent='986',gloss=['I'],forms=['ham'],languages=['H','Kannauji-Central','MagahiNepal','bundeli_atarra','KochilaTharu'],citation='CDIAL[986];platts1884[s.v. ham, personal pronoun]',evidence='Turner’s asmad article explicitly lists Maithili, Bhojpuri, Awadhi and Hindi ham as the direct and oblique continuation of the plural stem. Platts separately attests the same pronoun’s frequent singular “I” use, resolving the apparent number mismatch in these elicited responses. The link is to asmad, not aham or the homonymous Persian ham prefix; longer hāmra/hame forms are excluded.'),
 dict(parent='13271',gloss=['snake'],forms=['śyæp','śʸæp','sʸæp'],languages=['Dotyali'],citation='CDIAL[13271]',evidence='Full sarpa explicitly includes Kumaoni syāp “snake”, alongside Nepali sāp. The Dotyali syæp variants match this geographically specific palatal-glide continuation; source ś/s notation and fronted vowel are retained.'),
 dict(parent='13271',gloss=['snake'],forms=['hāp','hap'],languages=['Noiri','PauriBareli','Bhili'],citation='CDIAL[13271]',evidence='Turner lists Gujarati/Marathi sāp and the wider Prakrit sappa snake family. These short western Bhil hāp/hap forms retain that stem with the regional s-to-h correspondence also seen in the saved dry/needle series. Compound responses containing another snake word, and -ḍ/-l-expanded stems, are excluded.'),
 dict(parent='13271',gloss=['snake'],forms=['sab','sapo','sā̃pʰ'],languages=['AdivasiOriya','Or','Goj'],citation='CDIAL[13271]',evidence='Full sarpa gives Oriya sāpa/sā̃pa and the neighboring plains sāp/sapp family. The selected simple responses retain that snake stem, with source final b/aspiration and Oriya nominal vowel preserved. They have no unexplained r retention or additional diminutive stem.'),
 dict(parent='7039',gloss=['snake'],forms=['nāŋg'],languages=['Bshk'],citation='CDIAL[7039,1]',evidence='Turner explicitly records Bashkarik nāṅg “snake” as an early nasalized *nāṅga continuation within nāga section 1. The survey form is the same named-language comparator; the elephant subsection and uncertain palatalized/devoiced survey variants are excluded.'),
 dict(parent='5994-3',gloss=['three'],forms=['ṭin','ṭīn','ṭiṇe'],languages=['Dang','Vasavi'],citation='CDIAL[5994,3]',evidence='Section 3 trīṇi supplies the widespread tin/tīn and western triṇ/traṇ numeral family. These complete Tharu/Vasavi numeral responses retain its diagnostic final nasal, with source retroflex and final vowel notation preserved; classifier phrases are excluded.'),
 dict(parent='5994-3',gloss=['three'],forms=['cin','cīn','cĩn','coṇ'],languages=['kul'],citation='CDIAL[5994,3,Addenda]',evidence='The addendum specifically treats Western Pahari Kiunthali cīn and Kotgarhi cɔ̈̄n under the trīṇi branch, explaining the cī oblique through *trihi and the rounded forms with Middle Indo-Aryan *trao influence. These Kullui cīn/coṇ survey forms match that regional palatalized nasal-final series; the r-retaining forms are kept separate.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and r['Language_ID'] in g['languages']:
   a.append(dict(record=r,parent=g['parent'],kind='reflex',citation=g['citation'],evidence=g['evidence'],family=i))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass017-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
(P/'pass017-decisions.json').write_text(json.dumps({'accepted':a,'held':[]},ensure_ascii=False,indent=1)+'\n')
f=P/'lexical-reference-links.json';links=json.loads(f.read_text());links['s.v. ham, personal pronoun']='https://www.rekhta.org/urdudictionary?keyword=ham';f.write_text(json.dumps(links,ensure_ascii=False,indent=1)+'\n')
print(len(a),[(g['parent'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
