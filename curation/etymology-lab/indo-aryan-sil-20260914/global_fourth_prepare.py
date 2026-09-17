import json,csv,re
from pathlib import Path
P=Path(__file__).resolve().parent
exec((P/'sixth_prepare.py').read_text().split('qs=[]')[0])
qs=[]
def add(parent,gloss,words,ev):qs.append(dict(parent=parent,gloss=gloss.split('|'),words=words.split('|'),citation='CDIAL['+parent.replace('-','.',1)+']',evidence=ev))
add('10188','heart','muṭu|mutu|muṭṭu|muṭ̚ṭu','CDIAL 10188 *muṭṭa² explicitly gives Nepali muṭu “heart, courage, feelings”. The simple survey muṭu/mutu heart forms fit this family; dental versus retroflex transcription is retained as a phonetic qualification. The reconstruction’s proposed connection with the lump family remains uncertain.')
add('10223','pestle','musul|musala|musar|musara|musura|musaṛi|muzal|muzel|muhlo|mola|mula|muhḷu','CDIAL 10223.1 musala gives Khowar mvsul, Shina muzvl, Bihari musar, Hindi musal and Punjabi muhlā/mohlā “pestle”. The selected variants follow these documented s/z/h and l/r developments; possible non-Aryan remote origin and intra-IA diffusion remain open.')
add('10223-2','pestle','muśaḷ|muśoḷ|muśal|muśol','CDIAL 10223.2 explicitly separates *muṣala/*muśala, with Western Pahari muśḷ and Jaunsari mūśṛī “pestle”. The retained palatal sibilant selects this subsection rather than the unqualified musala head.')
add('10915','tail','lej|lenj|lenja|laj|lanj|lanz|lez|nej|neja','CDIAL 10915 lañja² explicitly gives Bengali lej/nej/nejā, Assamese lā̃z/lẽz/nez and Oriya lāñja “tail”. The selected eastern simple nouns preserve the documented nasal and nonnasal variants; longer opaque tail compounds are excluded.')
add('3607-2','near','kol|kole','CDIAL 3607.2 kōla explicitly includes Punjabi kol/kole “near” and Lahnda kol “with”, following the breast/lap noun and its spatial grammaticalization. This selects the kōla subsection, not the unsplit krōḍa head.')
add('13211','same','saman|soman|samaṇ','CDIAL 13211 samāna “same, alike” gives Prakrit samāṇa and Kashmiri samān/Gujarati samāṇu. The simple survey saman/soman forms preserve the whole adjective. Learned renewal or transfer across Indo-Aryan remains possible; this link identifies the lexical family.')
add('11378','broom','baṛni|baṛhani|baḍni|barhni|barhan|barni|baṛhni|barhani|baḍhahani|baḍahani|badhani|badni','CDIAL 11378, section 2, explicitly gives Assamese bārni, Bengali bāṛhan, Bihari baṛhanī, Bhojpuri bāṛhanī and Hindi baṛhnī “broom”, from the vardhanī sweeping formation. No distinct stored subsection node exists, so the citation selects section 2 on the existing head; the euphemistic motivation is not treated as settled.')
qs[-1]['citation']='CDIAL[11378.2]'
add('5523','path|road','ḍagar|ḍagra|ḍagra|ḍagaro','CDIAL 5523.1 *ḍag² “step” explicitly gives Oriya ḍagara “footstep, road”, Maithili/Hindi ḍagar and Hindi ḍagrā “road”. The retroflex a-vowel path forms select this branch, distinct from the dental *dag and i-vowel *ḍig subsections.')
raw={r['ID']:r for r in json.loads((P/'inventory.json').read_text())};acc=[];held=[]
for rr in csv.DictReader((P/'unresearched-records.csv').open()):
 r=raw[rr['ID']];w=norm(r['Form']);gs={g.strip().lower() for g in r['Gloss'].split(';')}
 for i,q in enumerate(qs):
  if w not in {norm(z) for z in q['words']} or not gs<=set(q['gloss']):continue
  reason=None
  if q['parent']=='10223-2':reason='The central/western survey palatal sibilant could continue musala through local phonetic development; the distinct *muṣala/*muśala subsection is attested explicitly in northwestern Pahari. Local evidence is needed to choose the branch.'
  if q['parent']=='10223' and rr['clade']=='Pahari' and w.startswith('muh'):reason='Pahari h may continue the separately numbered sibilant musala variant; choose the branch with local evidence.'
  if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason=reason or 'Source reading requires local review.'
  if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=21));continue
  ev=q['evidence']+' Any unresolved borrowing between Indo-Aryan languages is retained under the user’s preference for a provisional supported family link.'
  if r['Language_ID']=='Rana' and 'uncertain' in r['Tags']:ev+=' RNS uncertainty concerns source locality, not lexical reading, as established in the source audit.'
  acc.append(dict(record=r,family=i,parent=q['parent'],citation=q['citation'],evidence=ev))
(P/'global-fourth-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'global-fourth-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1))
(P/'global_fourth_save.py').write_text((P/'sixth_save.py').read_text().replace('sixth-save-decisions.json','global-fourth-decisions.json').replace('sixth','global-fourth'))
for i,q in enumerate(qs):
 a=[x['record'] for x in acc if x['family']==i];print(q['parent'],len(a),'; '.join(l+': '+', '.join(sorted({x['Form'] for x in a if x['Language_ID']==l})) for l in sorted({x['Language_ID'] for x in a})))
print('accepted',len(acc),'held',len(held))
