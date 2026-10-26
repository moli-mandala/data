import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
 dict(parent='3244',gloss=['axe'],languages=['Rana','Sunha','Kathoriya','awan','srk','Goj','Hajong','KochilaTharu','MagahiNepal','Majhi','hal','Bhatri','RathwiBareli','Bhilali','PalyaBareli','AdivasiOriya','kaithal','kul','poth','Noiri','DungraBhili'],forms=['kuṛhārī','kohārī','kvāṛā / kvāṛī','kvārī','gūāṛḍī','gūāḍī','kūāṛī / kūārī','kʰuāṛḍā','kurul','kuḍʰār','kuher','kurhair','kulāri','kuṛʰar','kuṛar','kuṛaṛi','kuṛaṛ','kurār','kurāḍu','kūrāḍū','kūradī','kūrdī','kūraīḍ','kūdarī','kūadrī','kʷāḍā','kʷāḍi','koṛʰāḍi','kəˈrərə','kurəḍu','kəṛaḍi','kəṛaḍə','ˈkəṛaḍi','kəṛʰãḍi','kəṛāḍi','kəṛā̃ḍi','kəṛəṛu','kovāṛī','kūuāṛi','kuvāṛɔ','kuvāḍe'],citation='CDIAL[3244]',evidence='Turner documents the kuṭhāra axe family with Prakrit kuḍhāra/kuhāḍa and explicitly discusses the ṭh–r versus h–ḍ metathesis. Comparanda include Punjabi kuhāṛā/kuhāṛī, Bengali kuṛul, Oriya kurhāṛi/kuṛāri, Gujarati kuvāṛī, Marathi kurhāḍ and Kotgarhi kəṛari. These survey forms preserve those regional axe stems, including masculine/feminine pairs and local initial voicing or reduced vowels; the family link does not assert a regular sound law for every local variant. Other axe names and extended or uncertain responses are excluded.'),
 dict(parent='12461',gloss=['mortar'],languages=['MagahiNepal'],forms=['silaut','silauti'],citation='CDIAL[12461]',evidence='Turner gives Bihari and Maithili silauṭ beside Hindi silauṭ/silwaṭ and Nepali silauṭo, a stone for grinding spices, from śilāpaṭṭa stone slab. The Magahi survey silaut/silauti names that grinding implement under the broader elicitation gloss mortar. The feminine final vowel is ordinary local nominal variation, not a separate added component.'),
 dict(parent='2360-4',gloss=['mortar'],languages=['kaithal','RathwiBareli','PauriBareli','Khandesi','Noiri','Vasavi'],forms=['vokʰli','ūkhɔv','ūkhõ','ūkhaī','ūkhiyo','ukhõ','ukhiõ'],citation='CDIAL[2360,4]',evidence='The fourth *udukkhala branch (Prakrit ukkhala/okkhala) explicitly includes Punjabi ukkhalī, Hindi ukhlī/okhlī, Gujarati ukhaḷ/ukhḷũ/ukhḷī and Marathi ukhaḷ. These survey mortar forms fit that reduced-initial ukh-/okh- series. The western forms with weakened or lost final l are qualified family comparisons, not claims that every local ending has an independently demonstrated historical derivation; pestle-glossed homophones and unrelated mortar names are excluded.'),
 dict(parent='10223',gloss=['pestle'],languages=['AdivasiOriya','jaun'],forms=['mvsūl','musar'],citation='CDIAL[10223,1]',evidence='The first musala pestle branch gives Oriya musaḷa and the Bihari/Maithili mūsar/musarā series, alongside western Himalayan musal. Adivasi Oriya mvsūl and Jaunsari musar preserve these s-bearing pestle stems. Turner separates ś-bearing western Himalayan forms into a second branch; those and h-bearing forms with unresolved branch membership are not included here.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and r['Language_ID'] in g['languages']:a.append(dict(record=r,parent=g['parent'],kind='reflex',citation=g['citation'],evidence=g['evidence'],family=i))
assert len(a)==len({x['record']['ID'] for x in a})
h=[]
for r in rs:
 if r['ID'] not in done and r['Gloss']=='head' and r['Form'].startswith(('mū','mu','mũ','moṇ','maṇ','ˈmaṇ')):
  h.append(dict(record=r,reason='CDIAL 10247 explicitly says the unaspirated head forms may derive from, or be contaminated by, muṇḍa shaved (10191), rather than solely from mūrdhan. These survey forms do not resolve the competing ancient source; apparent matching comparanda are not sufficient to choose a categorical parent.'))
(P/'pass028-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass028-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g['parent'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
