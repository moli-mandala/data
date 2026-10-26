"""Research-only Census Tamil residual DEDR article proposals and explicit holds."""
import csv,gzip,json
from pathlib import Path
P=Path(__file__).resolve().parent;D=P.parents[2]
records=json.loads((P/'survey-inventory.json').read_text());assert len(records)==349
search=[]
for start in range(0,349,35):
 end=min(start+35,349);search+=json.loads((P/f'discovery-{start+1:03}-{end:03}-results.json').read_text())
assert len(search)==349
assert [(i+1,q['opts']['params']['form']) for i,q in enumerate(search) if q['count']>len(q['rows'])]==[(183,'vā')]
# DEDR IDs denote whole articles. This catalogue is deliberately restricted to
# directly checked Tamil article families; loans and opaque elicitation phrases
# receive per-record research holds below.
groups=[
('d2552',[1],'DEDR 2552 explicitly lists Tamil iruṭṭu “darkness”; the Census form is identical.'),
('d5503',[7],'DEDR 5503 lists Tamil veḷḷam “flood, deluge”; the Census flood noun is identical.'),
('d4742',[9],'DEDR 4742 gives Tamil malai “hill, mountain”; the Census source has the same form despite a split gloss.'),
('d131',[20],'DEDR 131 gives Tamil aṇṇaṉ “elder brother”; Census annan keeps the kin stem while its gloss merges elder/younger.'),
('d3588',[21],'DEDR 3588 lists Tamil naṇpaṉ “friend, companion”; Census ṇanpan is the same friend noun with source nasal notation.'),
('d1594',[22],'DEDR 1594 includes Tamil ciṟumi “girl” under the ciṟu small/young family; Census cirumi keeps that feminine form.'),
('d4813',[25],'DEDR 4813 gives Tamil māmaṉ/māmā “maternal uncle”; Census māma is the matching kin term.'),
('d4715',[27,28],'DEDR 4715 explicitly gives Tamil maru-makaṉ “cross nephew” and feminine maru-makaḷ “cross niece”; Census preserves both source-defined kin forms.'),
('d474',[29],'DEDR 474 lists Tamil iraṭṭaiyar “twins” in the two/double family; Census iraṭṭaiyar is exact.'),
('d4355',[37],'DEDR 4355 gives Tamil pūṉai “cat”; Census pūnai is the same animal name.'),
('d2901',[40],'DEDR 2901 lists Tamil naṇṭu/ñaṇṭu “crab”; Census ṇanṭu preserves the nasal-dental animal name.'),
('d2591',[41,52],'DEDR 2591 gives Tamil iṟaku “feather” and iṟakkai “wing”; Census iraku/irakkai are source spellings of the two attested senses.'),
('d3606',[43],'DEDR 3606 gives Tamil nari “jackal” and Kannada nari “fox”; Census nari “fox” identifies that canid family without imposing a narrower species.'),
('d4642',[45],'DEDR 4642 gives Tamil mayil “peacock”; Census mayil is an exact form-and-sense match.'),
('d1711',[47],'DEDR 1711 lists Tamil kutirai and Kannada kudire “horse”; Census kutirai is direct.'),
('d1769',[49],'DEDR 1769 explicitly gives Tamil kuraṅku “monkey, ape”; Census kuraȠku is the same animal name in source notation.'),
('d4307',[50],'DEDR 4307 gives Tamil puli “tiger”; Census puli is exact.'),
('d5155',[51],'DEDR 5155 gives Tamil āmai “turtle, tortoise”; Census āmai keeps the tortoise sense.'),
('d1366',[60],'DEDR 1366 gives Tamil kar̤uttu “neck, throat”; Census kaLuttu preserves the retroflex-lateral neck form in its source notation.'),
('d3976',[67],'DEDR 3976 gives Tamil parutti “cotton plant, cotton”; Census paruti retains the cotton noun.'),
('d5028',[75],'DEDR 5028 is the Tamil mūṅkil “bamboo” article; Census mūȠkil is its source-transcribed form.'),
('d3549',[81],'DEDR 3549 includes Tamil tōṭṭam “garden, plantation”; Census toṭṭam keeps the garden sense with a shorter source vowel.'),
('d3163',[84],'DEDR 3163 lists Tamil tāmarai “lotus”; Census tāmarai is exact.'),
('d1304',[85],'DEDR 1304 gives Tamil kalappai “plough, ploughshare”; Census kalappai is exact.'),
('d530',[86],'DEDR 530 includes Tamil iṉippu “sweetness”; Census inippu is the same taste-quality stem.'),
('d1762',[95],'DEDR 1762 gives Tamil kuyavaṉ “potter”; Census kuyava is the occupational stem with the ending omitted.'),
('d1249',[99],'DEDR 1249 explicitly gives Tamil kacappu “bitterness”; Census kacappu is identical.'),
('d1670',[100],'DEDR 1670 gives Tamil kuṭṭai “shortness, dwarfishness”; Census kuṭṭai keeps the short adjective stem.'),
('d1787',[101],'DEDR 1787 lists Tamil kuruṭu “blindness”; Census kuruṭu is the same blind stem.'),
('d4322',[107],'DEDR 4322 gives Tamil puḷippu “sourness”; Census pulippu preserves the sourness base.'),
('d5496',[109],'DEDR 5496 gives Tamil veḷḷai “white”; Census veḷḷai is exact.'),
('d3825',[110],'DEDR 3825 gives Tamil paci “hunger, be hungry”; Census paci is the same hunger stem.'),
('d5281',[112],'DEDR 5281 lists Tamil vali “pain, ache”; Census vali is exact.'),
('d697',[113],'DEDR 697 explicitly gives Tamil uṇmai “existence, reality, truth”; Census unmai “TRUE” preserves that truth stem.'),
('d3808',[118],'DEDR 3808 includes Tamil pakaivaṉ “enemy”; Census pakaivan is the matching antagonist noun.'),
('d3250',[121],'DEDR 3250 gives Tamil tiruṭṭu “theft, robbery”; Census tiruṭṭu is exact.'),
('d486',[130],'DEDR 486 explicitly gives Tamil irumpu “iron”; Census irumpu is exact.'),
('d474',[134,136],'DEDR 474 gives Tamil iraṇṭu “two” and iraṇṭām “second”; Census irenṭu/irenṭām preserve the numeral and its ordinary ordinal form.'),
('d5052',[135],'DEDR 5052 gives Tamil mūṉṟu “three” and mūṉṟām “third”; Census mūnrām keeps the ordinary ordinal.'),
('d1147',[137],'DEDR 1147 gives Tamil kaṭṭu “tie, fasten, bind”; Census kaṭṭu is the matching binding verb.'),
('d1957',[142],'DEDR 1957 gives Tamil cey “do, make”; Census cey is exact.'),
('d77',[145,176],'DEDR 77 gives Tamil aṭi “beat, strike”; Census enters the same verb for hitting and beating a drum.'),
('d3570',[148],'DEDR 3570 gives Tamil nakku “lick”; Census naku keeps the licking stem.'),
('d851',[152],'DEDR 851 includes Tamil eṭu “take, pick up”; Census eṭu keeps this taking stem.'),
('d3135',[155],'DEDR 3135 gives Tamil taḷḷu “push, shove”; Census taḷḷu is exact.'),
('d5549',[156],'DEDR 5549 gives Tamil vai “put, place”; Census vai is exact.'),
('d4993',[160],'DEDR 4993 includes Tamil mūḻku “sink, submerge”; Census mūLku keeps the source lateral notation.'),
('d3435',[161],'DEDR 3435 gives Tamil teḷi “sprinkle, scatter”; Census teḷi is exact.'),
('d3323',[163],'DEDR 3323 gives Tamil tuppu “spit”; Census tuppu is exact.'),
('d3549',[168],'DEDR 3549 also includes Tamil tōṇṭu “dig”; Census tōṇṭu is the digging verb, separate from its garden noun.'),
('d4194',[172],'DEDR 4194 lists Tamil piḷa “split, cleave”; Census piḷa matches the splitting verb.'),
('d4065',[177],'DEDR 4065 gives Tamil pāṭu “sing”; Census pāṭu is exact.'),
('d79',[178],'DEDR 79 explicitly gives Tamil aṭai “reach, arrive at”; Census aṭai keeps the arrival sense.'),
('d916',[182],'DEDR 916 gives Tamil ēṟu “climb, ascend”; Census ēru is the same climbing verb in source transcription.'),
('d5270',[183],'DEDR 5270 gives Tamil varu with imperative vā “come”; Census vā is the imperative stem. Its broad exact search was capped, so the full article resolves the family.'),
('d504',[185],'DEDR 504 gives Tamil ir̤u “draw, pull, drag”; Census īLu retains the source lateral transcription for dragging.'),
('d4020',[188],'DEDR 4020 gives Tamil paṟa “fly”; Census para is the source spelling of the flying verb.'),
('d4422',[189],'DEDR 4422 gives Tamil peṟu “get, obtain”; Census peru is the same obtaining verb.'),
('d4572',[191],'DEDR 4572 gives Tamil pō “go”; Census pō is direct.'),
('d2342',[209],'DEDR 2342 gives Tamil camai “cook”; Census camai is exact.'),
('d600',[211],'DEDR 600 includes Tamil ūṭṭu “feed a child or sick person”; Census ūṭṭu retains the causative feeding stem.'),
('d2017',[215],'DEDR 2017 gives Tamil kēḷ “hear, ask, inquire”; Census keLu is the asking stem.'),
('d3630',[231],'DEDR 3630 gives Tamil naṉai “become wet, moisten”; Census ṇanai preserves the wet stem with source nasal notation.'),
('d1562',[239],'DEDR 1562 gives Tamil ciri “laugh”; Census ciri is exact.'),
('d3376',[241],'DEDR 3376 gives Tamil tūṅku “sleep, slumber”; Census tūȠku keeps this sleep stem.'),
('d4886',[242],'DEDR 4886 gives Tamil mukar “smell”; Census mukar is exact.'),
('d1443',[253],'DEDR 1443 gives Tamil kāṭṭu “show”; Census kāṭṭu is exact.'),
('d3682',[260],'DEDR 3682 gives Tamil niṟaiya “abundant, full”; Census ṇiraiya keeps the fullness stem.'),
('d5078',[272,273,277],'DEDR 5078 gives Tamil mel/men “soft” and meli “become lean, thin”; Census menmaiyāna and melliya keep the soft/thin adjective family with derived endings.'),
('d311',[280],'DEDR 311 gives Tamil aṟam “virtue, moral duty”; Census aram is the source spelling of that noun.'),
('d1619',[286,307],'DEDR 1619 gives Tamil kir̤akku “east” and kīr̤ “below, down”; Census kiLakku/kīLe retain these lateral forms.'),
('d5218',[290],'DEDR 5218 gives Tamil vaṭakku “north”; Census vaṭakku is exact.'),
('d3449',[293],'DEDR 3449 includes Tamil teṟku “south”; Census terkku retains the directional stem.'),
('d5151',[295],'DEDR 5151 gives Tamil eppōtum “always”; Census eppotum is the temporal interrogative/indefinite form.'),
('d5020',[296],'DEDR 5020 gives Tamil muṉ and muṉṉar “before, in front”; Census mun retains that temporal/spatial stem.'),
('d1',[309,310,324,325,326,327,328],'DEDR 1 contains Tamil avaṉ/avaḷ/avar and distal anta/aṅku/appōtu; Census distal pronouns and adverbs stay within that whole paradigm, with source case morphology retained.'),
('d410',[313,317,318,319],'DEDR 410 contains Tamil itu “this” and proximal iṅku “here”; Census proximal pronouns and locatives retain that i- base with source case morphology.'),
('d5151',[316,331,332,333,334,335,336],'DEDR 5151 contains Tamil eppaṭi “how”, eṉṉa “what”, eppōtu “when”, eṅku “where” and yār “who”; Census forms retain the corresponding interrogative bases and ordinary case forms.'),
('d5160',[320,321],'DEDR 5160 gives Tamil nāṉ with en- oblique “I/my”; Census enakku/en retain that first-person singular stem.'),
('d5154',[322,330],'DEDR 5154 includes Tamil eṅkaḷ exclusive “we” with oblique/possessive extensions; Census enkaḷuṭaiya/eȠkaḷukku preserve that plural stem.'),
('d3684',[337,338],'DEDR 3684 gives Tamil nī with uṉ- oblique “you” singular; Census unakku/unnuṭaiya retain the oblique and ordinary case endings.'),
('d698',[342],'DEDR 698 gives Tamil uḷ “inside, interior” and its locative use; Census uḷḷe “in” preserves the interior base with locative morphology.'),
('d833',[346],'DEDR 833 gives Tamil eli “rat” and Kannada eli/ili “rat, mouse”; Census eli “mouse” belongs to the same small-rodent family.'),
('d4879',[348],'DEDR 4879 gives Tamil mīcai “moustache”; Census mīcai is exact despite the split source gloss.'),
]
proposals={}
for parent,idxs,evidence in groups:
 for n in idxs:
  assert n not in proposals,n
  proposals[n]=([parent],evidence)
proposals[26]=(['d4813','d4616'],'Census māman makaḷ “maternal uncle’s daughter” joins Tamil māmaṉ “maternal uncle” in DEDR 4813 and makaḷ “daughter” in DEDR 4616 in source order.')

ids={p for ps,_ in proposals.values() for p in ps};articles={}
with gzip.open(D/'data/dedr/entry-texts-audit.csv.gz','rt') as f:
 for r in csv.DictReader(f):
  if r['Entry_ID'] in ids:articles[r['Entry_ID']]=r
assert set(articles)==ids,ids-set(articles)
(P/'primary-evidence.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2)+'\n')

accepted=[];held=[]
special_holds={
 6:'Tīvu is normally “island” in Tamil; the Census “pond” gloss conflicts with that primary sense.',
 11:'Kaṭal is “sea” in Tamil, while Census glosses it “valley”; this semantic conflict needs source inspection.',
 44:'Kiṭa in DEDR 1123 is specifically a male goat or ram; the unqualified Census goat gloss does not fix sex.',
 64:'Arici denotes uncooked rice in Tamil, but the source gloss joins uncooked and cooked responses; clarify the elicitation before linking.',
 65:'Cōru denotes cooked rice in Tamil, but the source gloss joins uncooked and cooked responses; clarify the elicitation before linking.',
 83:'Palā paLam is a jackfruit expression, while the Census gloss says cucumber; the source sense must be checked.',
 119:'Kaṭan “debt” conflicts with the Census “victory” gloss; source transcription or gloss needs inspection.',
 329:'Avarkaḷukku is dative “to them”, but the Census gloss says “their”; do not choose case-sensitive analysis until verified.',
 78:'The Census mīḷakai “chilli” resembles the miḷaku pepper family (DEDR 4867), but its fruit ending and the vowel/transcription require a component check before assigning this specific form.',
}
recent={'bus','student','teacher','bullet','divorce','heaven','hell','brass','gold','money','banian','sari','shoe','mason','cabbage','potato','tomato','ink'}
for n,(r,q) in enumerate(zip(records,search),1):
 if n in proposals:
  ps,evidence=proposals[n]
  x={'record':r,'parent':ps[0],'kind':'component' if len(ps)>1 else 'reflex',
     'citation':r['Source']+';'+','.join('dedr['+p[1:]+']' for p in ps),
     'evidence':evidence,'comparison':'Homepage exact search, full DEDR article and exact Census record checked; whole-article family identification.',
     'status':'pending-review'}
  if len(ps)>1:x['components']=ps
  accepted.append(x)
 else:
  form=r['Form'];gloss=r['Gloss'];g=gloss.lower()
  if n in special_holds:reason=special_holds[n]
  elif any(k in g for k in recent):reason='A likely contact or modern cultural term needs its immediate donor verified.'
  elif ' ' in form:reason='The elicited response is multi-part; its constituent stems and construction need separate verification.'
  elif '/' in gloss or '?' in gloss:reason='The source gloss is disjunctive or qualified; a single parent sense cannot be fixed yet.'
  else:reason='No same-sense primary DEDR article has been verified for this exact source transcription in this pass.'
  held.append({'record':r,'candidates':[],'reason':f'{form} “{gloss}”: {reason} Exact-form homepage search returned {q["count"]} row(s); further comparison remains open.'})
(P/'proposed-decisions.json').write_text(json.dumps({'status':'pending-review','language':'Tamil','source':'census2023tamilnadu','accepted':accepted,'held':held},ensure_ascii=False,indent=2)+'\n')
(P/'research-holds.json').write_text(json.dumps(held,ensure_ascii=False,indent=2)+'\n')
print('proposed',len(accepted),'rows',sum(len(x.get('components',[x['parent']])) for x in accepted),'held',len(held),'articles',len(articles))
