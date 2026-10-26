# Original six-record research proposal only. Do not rerun: final decisions hold both foot records after existing ancestry-cycle rejection.
import csv,json,unicodedata
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
linked={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'};linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
rs=[r for r in json.loads((P/'schmidt-poguli-siraji-remaining057.json').read_text()) if r['ID'] not in linked]
groups=[('foot', ['kʰur', 'kʰor'], '3906', 'khura', 'Turner3906 section1 explicitly gives Poguli khur and Kashmiri khor foot, matching the two source forms. This is the r-retaining khura branch, not section2 khuḍa. Turner proposes an ancient Dravidian origin for the word family; that does not by itself make these modern regional forms direct Dravidian loans.'), ('nose', ['naht'], '7031', 'nasta', 'Turner7031 section1 explicitly gives Poguli nāht beside Kishtawari nasth. The table naht has the same diagnostic consonant order; its unmarked vowel length is preserved. Do not assign this ht form to the separate nakka family.'), ('nose', ['nak'], '6909', '*nakka', 'Turner6909 gives neighbouring Rambani/Ḍoḍi and Bhalessi nakk, Kashmiri nākh and wider nak forms. Siraji nak matches this k-family; Turner leaves its deeper derivation from nas-ka or nast-ka uncertain.'), ('moustache', ['mucʰ'], '10180', '*mucchā', 'Turner10180 section1 gives Lahnda/Punjabi mucch and nearby much/mūch moustache, supporting Siraji mucʰ rather than the o-vowel section2. Turner explicitly leaves the formation open between gōcchā influenced by mukha and MIA massu from śmaśru.'), ('moustache', ['gucʰe'], '4269', 'gōcchā', 'Turner4269 gives Prakrit guṁchā moustache, Kashmiri gõch and Savi gōčē. Poguli gucʰe fits this regional g-initial moustache family; its unmarked nasalisation and u vowel are retained. The proposed connection with guccha cluster is uncertain, so no direct link to that broader bunch-word entry is substituted.')]
a=[];audit=[];source=list(csv.DictReader(open('data/other/forms/raw_data/schmidt_kashmiri.csv')));raw=list(csv.reader(open('data/other/forms/20260725-schmidt-kashmiri.csv')))
lines=['# Schmidt and Kaul: Poguli and Siraji further body vocabulary','','Six reflex records across five families; standing autonomous authorization. Table 3, printed pp. 283–284 (PDF pp. 53–54), visually checked.','','| # | Poguli / Siraji forms and meaning | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(g,fs,parent,label,ev) in enumerate(groups,1):
 hits=[r for r in rs if r['Gloss']==g and r['Form'] in fs];assert len(hits)==len(fs),(g,len(hits))
 for r in hits:
  rr=[z for z in raw if z[0]==r['Language_ID'] and z[2]==r['Original'] and z[3]==g and z[6]==r['Description']];assert len(rr)==1,(r,rr)
  item=r['Description'].split()[-1];sr=next(z for z in source if z['Item']==item)
  col={'pog':'Poguli','sir':'Siraji'}[r['Language_ID']];assert unicodedata.normalize('NFC',sr[col])==r['Original']
  page=283 if int(item)<=13 else 284
  citation=f'schmidt[p. {page}, Table 3, item {item}];CDIAL[{parent} and addenda]'
  a.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=parent,kind='reflex',citation=citation,evidence=ev+' Source image and reviewed import row checked; no source transcription or identity changes.'))
  audit.append(dict(record_id=r['ID'],reviewed_source_row=sr,import_row=rr[0],printed_page=page,image=f'../tmp/pdfs/schmidt_kaul_2008/boundary150/page-{sr["Page"]}.png',visual_check='confirmed'))
 label=label.replace('*',r'\*');lines.append(f'| {i} | **{" / ".join(fs)}** ‘{g}’ ({len(hits)}) | **{label}**, CDIAL **{parent}** | {ev} |')
reasons={'hōṭ':'Discriminate 245a adhaōṣṭha from 2563 oṣṭha; the h-initial form alone does not settle the competing histories.','kʸāliz':'Turner3103 marks Kashmiri kalēja as borrowed from Lahnda. Resolve the immediate history of this Poguli palatalised form before choosing an inherited or borrowed edge.','kālzó':'The kālēyaka family is plausible, but its distribution includes loans and questioned Pahari forms; establish the Siraji route before linking.','masti':'The mastika/head comparison does not yet explain the source hair sense and formation.','zuṭó':'Jūṭa5258 explains knot-of-hair forms with intervocalic ṛ/ḍ; the source stop and broad hair sense need further evidence.'}
h=[dict(record=r,reason=reasons[r['Form']]) for r in rs if r['Form'] in reasons and r['Gloss'] in ['lip','liver','hair']]
assert len(a)==6 and len(h)==5

for name,val in [('decisions',dict(accepted=a,held=h)),('source-audit',audit)]: (P/f'pass058-{name}.json').write_text(json.dumps(val,ensure_ascii=False,indent=2)+'\n')
lines+=['','Five records held:']+[f'- **{x["record"]["Form"]}** ‘{x["record"]["Gloss"]}’: {x["reason"]}' for x in h]+['','Discovery: schmidt-search003–006. Full primary articles/addenda: schmidt-primary004–008. All source forms rechecked against the already inspected printed p.284 and reviewed import rows. No rebuild.','','Save and validation pending.'];(P/'pass058-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(h),'held')
