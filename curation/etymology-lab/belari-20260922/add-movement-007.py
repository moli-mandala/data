import json,csv,gzip,sys
from pathlib import Path
root=Path.cwd();p=root/'data/curation/etymology-lab/belari-20260922';sys.path.insert(0,str(root/'data'))
from etymology_assignments import read_assignments
z=json.load(open(p/'proposed-decisions.json'));assigned={r['Form_ID'] for r in read_assignments()};seen={x['record']['ID'] for x in z['accepted']+z['held']}
rules={
'koḷu':('d2053','DEDR2053 gives Tulu (B-K.) koḷu give/grant; several other article members explicitly mean give to a third person. Belari koḷu matches the full form. Bhat1971 p.119 distinguishes its third-person recipient use from itta, retained as a separate unresolved source record; the installed generic give gloss is preserved.'),
'simbɨ':('d2677','DEDR2677 explicitly records Belari (LSB2.1) simbi carry. Survey simbɨ agrees in the s-mb stem with final ɨ versus i. It is a verb of carrying, not an analyst-derived carrying-pad noun; both occur within the whole family.'),
'ogi':('d984','DEDR984 explicitly records Belari (LSB2.1) ogi pull. The full source form matches; other article members express carrying or giving, but those meanings do not replace the Belari pull gloss.'),
'bī':('d5517','DEDR5517 includes the cooking/heat family with Tulu bēyuni/bēpini and Kannada bē, plus Kurux bi’inā and Malto bice/bite. Belari bī cook (something) is a qualified long-ī member of this family; its vowel differs from neighbouring Tulu ē. The article contains transitive cooking as well as intransitive heating, so no unexpressed causative component is supplied.')}
holds={
'itta':([],'Bhat1971 p.119 describes giving to first/second person, contrasting koḷu. Form search has not established an article explaining the complete itta stem. Recipient semantics alone cannot force it under the tar- give family or turn it into a variant of koḷu.'),
'tari':(['d3122'],'DEDR3122 has Kui tāṛa/tāṛi- dig and other scratching/digging forms, but Belari tari differs in marked vowel quantity and rhotic articulation. No close southern or same-language attestation found by current searches; retain this as a candidate needing correspondence evidence.'),
'toḷaga':([],'Relaxed form search returns shining and departing families, not rubbing. Tulu rubbing verbs have different full forms. Do not link to a phonetic match with an unrelated meaning or discard the Belari ending.'),
'baḷi':(['d4321','d3464'],'Full DEDR4321 gives Koraga boḷa float and Tulu pulāvuni, a plausible family lead, but Belari baḷi differs in both vowels. DEDR3464 tēl- is semantically apt but less close in consonants. Keep the source form unresolved pending stronger comparison.'),
'uḷu':(['d3319'],'Tulu wearing search supplies tuttuni and DEDR3319 contains tutt-/tūnd-/tund- formations. None directly explains Belari uḷu. Relaxed form search only matches unrelated words or English swear; no supported full-form wearing analysis yet.')}
for r in json.load(open(p/'inventory.json')):
 if r['ID'] in assigned or r['ID'] in seen:continue
 if r['Form'] in rules:
  d,e=rules[r['Form']];z['accepted'].append(dict(record=r,parent=d,kind='reflex',citation=f'dedr[entry {d[1:]}];'+r['Source'],evidence=e+' Inheritance versus local transmission remains subject to review.',status='pending-review'))
 elif r['Form'] in holds:
  c,e=holds[r['Form']];z['held'].append(dict(record=r,reason=e,candidates=c))
(p/'proposed-decisions.json').write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n');a=json.load(open(p/'primary-evidence.json'));ids={x['parent'] for x in z['accepted']};a.update({r['Entry_ID']:r for r in csv.DictReader(gzip.open(root/'data/data/dedr/entry-texts-audit.csv.gz','rt')) if r['Entry_ID'] in ids and r['Status']!='duplicate-excluded'});(p/'primary-evidence.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n');print(len(z['accepted']),len(z['held']))
