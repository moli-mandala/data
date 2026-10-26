import csv,gzip,json,sys
from pathlib import Path
root=Path.cwd();p=root/'data/curation/etymology-lab/pattapu-20260922';sys.path.insert(0,str(root/'data'))
from etymology_assignments import read_assignments
z=json.load(open(p/'proposed-decisions.json'));seen={a['record']['ID'] for a in z['accepted']+z['held']}|{a['Form_ID'] for a in read_assignments()}
rules={
'one':('d990','DEDR990(d) gives Kannada ondu, Kodagu ondï and Tamil oṉṟu/oṇṇu one. Pattapu voⁿdu retains the nasal-stop formation with an initial v absent from these comparanda; this prothetic correspondence is proposed explicitly rather than silently dropping it.'),
'two':('d474','DEDR474 gives Tamil colloquial reṇṭu, Malayalam raṇṭu and Irula reṇḍu/raṇḍu two. Pattapu ɹoⁿdu preserves the rhotic and nasal-stop skeleton, with a rounded vowel retained as a comparison requiring review.'),
'three':('d5052','DEDR5052 gives Tamil mūṉṟu, Kota mūnḏ, Kodagu mūndï and Telugu mūḍu three. Pattapu mūdu preserves the long vowel and final voiced stop; the lack of a nasal in its transcription is not repaired.'),
'four':('d3655','DEDR3655 explicitly gives Tamil/Malayalam nālu, Kannada nāl(u) and Kodagu nālï four. Pattapu nālu matches the long-vowel lateral stem.'),
'five':('d2826','DEDR2826 gives Kodagu añji, Kota anj and Tamil/Malayalam añcu five. Pattapu aⁿji preserves the nasal-affricate form, as opposed to automatically choosing the Telugu ayidu formation.'),
'six':('d2485','DEDR2485 gives Tamil/Malayalam āṟu, Kannada āṟu and Kodagu ārï six. Pattapu āɹu matches the long-vowel rhotic numeral. Its homographic river record is a separate lexeme and is not included.'),
'seven':('d910','DEDR910 gives Kodagu ëḷï and Tulu ēḷů seven alongside Tamil/Malayalam ēḻu. Pattapu ēlu preserves the long-vowel lateral outcome; the recorded non-retroflex l remains unchanged.'),
'eight':('d784','DEDR784 gives Tamil/Malayalam eṭṭu and Kodagu ëṭṭï eight. Pattapu etu preserves the vowel-stop sequence without the printed gemination/retroflexion of the comparanda. This is a family comparison, not transcription correction.'),
'nine':('d1025','DEDR1025 gives Kannada ombattu, Tamil ompatu and Kodagu oymbadï nine. Pattapu vombid̪u retains the mb cluster and later stop with initial v and a different second vowel. Those differences are explicit; no novel subtraction or compound decomposition is imposed.'),
'ten':('d3918','DEDR3918 gives Tamil/Malayalam pattu, Toda pot and Kodagu pattï ten. Pattapu poθu preserves the p-initial numeral with rounded vowel and dental fricative in place of the stop-series comparanda. This correspondence is proposed with the exact θ retained, not normalized to t.')}
for r in json.load(open(p/'inventory.json')):
 if r['ID'] in seen or r['Gloss'] not in rules:continue
 d,ev=rules[r['Gloss']];z['accepted'].append(dict(record=r,parent=d,kind='reflex',citation=f'dedr[entry {d[1:]}];'+r['Source'],evidence=ev+' Whole-article family membership is proposed; local contact and exact sound history remain subject to review. No higher numeral components or upstream cognacy labels are treated as independent proof.',status='pending-review'))
(p/'proposed-decisions.json').write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n');used={a['parent'] for a in z['accepted']};a={r['Entry_ID']:r for r in csv.DictReader(gzip.open(root/'data/data/dedr/entry-texts-audit.csv.gz','rt')) if r['Entry_ID'] in used and r['Status']!='duplicate-excluded'};(p/'primary-evidence.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
