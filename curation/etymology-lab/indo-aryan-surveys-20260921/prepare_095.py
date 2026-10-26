import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
('8045','palm',['pɛ̣̃'],237,'Trail and Cooper explicitly derive pẹ̃ palm from pāṇi. Turner also gives the Kalasha palm reflex, keeping it distinct from the uncertain Khowar phan comparison.'),
('6914','fingernail',['nāŋgūźɛk','nāgūźɛk'],212,'Trail and Cooper derive naŋguś fingernail, with variant nawžik, from nakha (first branch). The survey forms are compared with that locally extended nail family; Turner discusses possible interaction with *aṅguṣ in this same branch, so a wholly regular unextended derivation is not asserted.'),
('6757','roof',['drāmī','ḍrāmī'],82,'Trail and Cooper explicitly derive drami roof, Urtsun ḍrami, from dharman support. Turner directly cites the Kalasha roof form.'),
('1576','lightning',['īndocik','indocik'],128,'Trail and Cooper and Turner explicitly give Kalasha indocik under *indradyotya lightning.'),
('13249','wind',['sī̃ra','sīra'],272,'Trail and Cooper explicitly assign sirã wind to saraṇyu. Turner gives the same Kalasha comparison with a question mark; this link preserves that tentative historical comparison, not an unqualified sound-law claim.'),
('13784','root',['isnõs','iznos','īsnos'],130,'Trail and Cooper derive isnos root from snasā tendon. Turner explicitly gives Kalasha Rumbur istnos and Urtsun iznos root, supporting the survey dialect forms.'),
('5323','thorn',['cũk'],63,'Trail and Cooper explicitly derive coŋk thorn from *jhaṅk clump/cluster. The nasalized survey cũk is compared with that local thorn family; this is their etymological proposal, not a claim that Turner independently lists the Kalasha form.'),
('12528','dog',['śọ̃ā','ṣụ̄̃ā'],286,'Trail and Cooper explicitly derive śọ̃a dog from śuna; Turner lists the Kalasha reflexes in its first branch. The local sibilant and nasal vowel spellings are retained.'),
('9467','husband',['bɛrū','bārū'],36,'Trail and Cooper explicitly derive beru husband, Urtsun baru, from *bhāriyāpa. Turner gives the corresponding Kalasha and neighboring reflexes under this reconstruction.'),
('3517','who',['kūrā'],177,'Trail and Cooper and Turner explicitly derive Kalasha kura who from the historically lexicalized ko ’paraḥ who else. The existing compound entry is used as the parent.'),
('3384','where',['kavā','kavāī','kam'],153,'Trail and Cooper explicitly derive kawa where, with Urtsun kam, from kuha and analyze kaway to where as kawa-ay. The survey forms are assigned to that local interrogative family, retaining the directional extension.')]
a=[];held=[]
for i,(parent,gl,forms,page,ev) in enumerate(groups):
 for r in rs:
  if r['Language_ID']=='Kal' and r['ID'] not in done and r['Gloss']==gl and r['Form'] in forms:a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}];trail-cooper1999[printed p. {page}, PDF p. {page+25}]',evidence=ev,family=i))
for r in rs:
 if r['Language_ID']=='Kal' and r['ID'] not in done and r['Gloss']=='heard':held.append(dict(record=r,reason='The local saŋga- hear match is clear, but Turner’s 12842 addendum proposes alternative ancestry from *saṃhati/*saṅghati or śaṃsati; do not silently assign the unrevised saṃkhyāti head.'))
(P/'pass095-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=1)+'\n');print(len(a))
