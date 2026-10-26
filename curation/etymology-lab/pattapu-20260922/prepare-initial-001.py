import json,csv,gzip,sys
from pathlib import Path
root=Path.cwd();p=root/'data/curation/etymology-lab/pattapu-20260922';old=p.parent/'das-yerava-20260922';sys.path.insert(0,str(root/'data'))
from etymology_assignments import read_assignments
inv=json.load(open(p/'inventory.json'));assigned={r['Form_ID'] for r in read_assignments()}
rules={
'kanu':('eye','d1159','DEDR1159(a) gives Tamil kaṇ, Malayalam kaṇṇu and Telugu kanu/kannu eye. Pattapu kanu preserves the same k-n eye stem with a final vowel; the source consonant quantity and place are retained rather than rewritten to the Tamil form.'),
'kayi':('arm or hand','d2023','DEDR2023 gives Tamil kai, Malayalam kayyi and Kannada kayi hand/arm. Pattapu kayi (source IPA kaji) matches the velar-plus-glide hand stem and combined arm/hand sense; j is the source phonetic glide, rendered y in the existing Jambu form.'),
'kālu':('foot or leg','d1479','DEDR1479 gives Tamil/Malayalam kāl, Kodagu kālï and Telugu kālu leg/foot. Pattapu kālu preserves the long vowel and lateral, with the source foot-or-leg gloss left intact.'),
'nāi':('dog','d3650','DEDR3650 explicitly gives Tamil nāy/nāi and Malayalam nāy dog. Pattapu nāi preserves the full simple dog stem. The explicitly analysed -kuṛi element of other languages in this article is absent here and is not supplied.'),
'mīnu':('fish','d4885','DEDR4885 gives Tamil/Malayalam mīn, Kodagu mīnï and Telugu mīnu fish. Pattapu mīnu matches the long-vowel nasal stem. The Sanskrit mīna comparison is retained as a comparison, not an immediate donor assertion.'),
'puvu':('flower','d4345','DEDR4345 gives Malayalam pūvu, Kannada pūvu and Telugu pūvu/puvvu flower. Pattapu puvu preserves the p-v shape but marks no long vowel or geminate; this is a family comparison, not a claim of identical phonetic realization. The questioned Sanskrit puṣpa comparison is not treated as a donor route.'),
'vāi':('mouth','d5352','DEDR5352 gives Tamil vāy, Malayalam vāy/vā and Telugu vāyi mouth. Pattapu vāi preserves the long vowel and diphthong/glide termination; it is the mouth noun, not an added compound component.'),
'muku':('nose','d5024','DEDR5024 gives Tamil/Malayalam mūkku, Kodagu mūkï and Telugu mukku nose. Pattapu muku has the same m-k stem with short-vowel/single-stop transcription, which is retained without normalizing quantity or selecting a Telugu donor solely from resemblance.'),
'nāku':('tongue','d3633','DEDR3633 gives Tamil/Malayalam nākku alongside nā/nāvu tongue. Pattapu nāku agrees in the long vowel and velar formation; the source single k remains unaltered.'),
'polu':('tooth','d3986','DEDR3986(a) gives Tamil pal, Malayalam pallu and Telugu palu/pallu tooth. Pattapu polu is a proposed rounded-vowel member of this p-l family. The vowel correspondence remains a comparative hypothesis and is not repaired to palu.'),
'upu':('salt','d2674','DEDR2674(a) gives Tamil/Malayalam/Kannada/Telugu uppu salt. Pattapu upu has the corresponding salt stem, with the source single-stop transcription preserved. No extra salt-pan constituent is inferred.'),
'pāmu':('snake','d4085','DEDR4085 gives Telugu pāmu and Tamil/Malayalam pāmpu snake, with Kodagu pāmbï. Pattapu pāmu preserves a cluster-reduced form. Telugu contact versus inheritance is not decided merely by the matching shape.'),
'pambu':('snake','d4085','DEDR4085 gives Tamil/Malayalam pāmpu and Kodagu pāmbï snake. Pattapu pambu preserves the nasal-stop cluster but differs in marked vowel quantity; it remains a separate source record from pāmu, without imposing a synonym or variant edge between them.')}
props=[]
for r in inv:
 if r['ID'] in assigned:continue
 if r['Form'] in rules:
  g,d,ev=rules[r['Form']];assert r['Gloss']==g
  props.append(dict(record=r,parent=d,kind='reflex',citation=f'dedr[entry {d[1:]}];'+r['Source'],evidence=ev+' The proposal uses the full DEDR article; inheritance versus local transmission remains subject to review. Upstream cognacy/alignment labels are discovery aids, not independent evidence.',status='pending-review'))
(p/'proposed-decisions.json').write_text(json.dumps(dict(status='pending-review',accepted=props,held=[]),ensure_ascii=False,indent=2)+'\n')
(p/'survey-inventory.json').write_text(json.dumps(inv,ensure_ascii=False,indent=2)+'\n');(p/'externally-assigned-records.json').write_text(json.dumps([r for r in inv if r['ID'] in assigned],ensure_ascii=False,indent=2)+'\n');(p/'inventory-counts.json').write_text(json.dumps({'language':'Pattapu','source':'lindgren2023dravidian','initialUnlinked':101},indent=2)+'\n')
used={r['parent'] for r in props};articles={r['Entry_ID']:r for r in csv.DictReader(gzip.open(root/'data/data/dedr/entry-texts-audit.csv.gz','rt')) if r['Entry_ID'] in used and r['Status']!='duplicate-excluded'};(p/'primary-evidence.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2)+'\n')
s=(old/'render.py').read_text().replace('Das (1987) Yerava','Pattapu (Lindgren 2023)').replace('das-yerava-20260922','pattapu-20260922');(p/'render.py').write_text(s);(p/'verify-checkpoint.py').write_text((old/'verify-checkpoint.py').read_text());print('proposals',len(props))
