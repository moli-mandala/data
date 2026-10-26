import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
(dict(parent='3197',glosses=['what kind?','what kind'],forms=['kīsok','kīhɔ','kīśõk','kaśyā','kaśyān','kaśyɔk','kaśyek','kɛīsɛ','keīsɛ','kæse','kæsan','kæīsīn','kʌsʌn','kɔslā','kɔsɔ','kɔsalo','kɔhalɔ','koslā','kaslõ','kɔslo','kehlū','kisā','kisak','kɔśṇu','koço','kaśṇe','kośnoi','kaçā'],evidence='Turner kīdṛśa explicitly gives Old Marwari kiso, Hindi kaisā, Punjabi kihā/kihṛā, Marathi kasā and related kis-/kas-/keh- type interrogatives. These survey forms fit that what-kind family, including local inflection and extensions. The extended -l-/-n- forms are qualified regional comparisons, not independent reconstructions of each suffix.')),
(dict(parent='3197',glosses=['what kind?','what kind'],forms=['kīyā','kiyā'],languages=['bagri_fatehabad'],evidence='Turner kīdṛśa explicitly records Awankari kìā and Punjabi kihā of what sort, alongside Sindhi kĩa. The Bagri kiyā/kīyā responses with the same what-kind sense are compared with this northwestern qualitative interrogative; they are not automatically equated with Hindi kyā what under kim.')),
(dict(parent='3197',glosses=['what','what?'],forms=['hū̃','hũ','sū'],evidence='Turner kīdṛśa explicitly derives Gujarati śũ what through *ksiũ with loss of the vowel. These neighbouring Bhili/Bareli/Gojri hū̃/sū forms are compared with that Gujarati-area interrogative, with local s/h alternation and loss of written nasalization retained. The exact regional contact route remains open.')),
(dict(parent='3197',glosses=['what','what?'],forms=['kahā','kāhā'],citation='CDIAL[3197];platts1884[kahā, interrogative]',evidence='Platts’s separate interrogative kahā entry (not the past of say) explicitly connects Old Hindi/Braj what/which to Prakrit keddaha/Apabhramsha keho and Sanskrit kīdṛśa. These central-western survey kahā/kāhā what responses fit that documented qualitative-interrogative family; the geographical where word kahā̃ is excluded.')),
(dict(parent='2575',glosses=['what','what?'],forms=['kun'],languages=['KochilaTharu'],evidence='Turner kaḥ punar supplies Nepali kun who, Oriya dialect kun which/what and the wider kaun/kon family. The Tharu kun what response is a qualified regional use of that interrogative, preserving its elicited sense.'))]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['glosses'] and r['Form'] in g['forms'] and ('languages' not in g or r['Language_ID'] in g['languages']):
   a.append(dict(record=r,parent=g['parent'],kind='reflex',citation=g.get('citation',f"CDIAL[{g['parent']}]"),evidence=g['evidence'],family=i))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass080-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=1)+'\n')
(P/'pass080-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),[(g['parent'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
p=P/'lexical-reference-links.json';links=json.loads(p.read_text());links['platts1884[kahā, interrogative]']='https://www.rekhta.org/urdudictionary?keyword=%DA%A9%DB%81%DB%81&lang=ur&reftype=rweb';p.write_text(json.dumps(links,ensure_ascii=False,indent=2)+'\n')
