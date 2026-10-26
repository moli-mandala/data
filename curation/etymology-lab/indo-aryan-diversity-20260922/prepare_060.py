import csv,json,unicodedata
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
linked={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'};linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
rs=[r for r in json.loads((P/'schmidt-poguli-siraji-remaining057.json').read_text()) if r['ID'] not in linked]
groups=[('house', ['gar'], '4428', 'ghara', 'Turner4428 gives Kashmiri gara, Poguli gar and neighbouring Pahari ghar house. Siraji gar fits this regional ghara family with source unaspirated g retained; Turner notes the phonetic difficulty of deriving ghara from gṛha, so no direct gṛha link is substituted.'), ('milk', ['dud', 'dudi'], '6391', 'dugdha', 'Turner6391 gives Kashmiri dod, Ḍoḍi duddh and Bhadrawahi/Bhalessi dùd milk through MIA duddha. The source Poguli dud and Siraji dudi share this regional milk stem; retain the Siraji final i as printed, without assigning an unsupported grammatical category.'), ('plow', ['al', 'hali'], '14000', 'hala', 'Turner14000 includes hala and hali plough and regional h-less al/āl beside Pahari hal. The source Poguli al and Siraji hali fit this family, preserving their different initial consonants and final vowels; these denote the plough, not only its share.'), ('salt', ['nūn', 'lūn'], '10978', 'lavaṇa', 'Turner10978 gives Kashmiri nūn/lun and regional lūṇ/nūṇ salt through MIA loṇa/lūṇa. Poguli nūn and Siraji lūn fit these established nasal/lateral initial variants. Source dental n and vowel length remain unchanged.'), ('smoke', ['dūom', 'dū̃m'], '6849', 'dhūma', 'Turner6849 section1 gives regional dūm/dum smoke and nasal-vowel Pahari dhũ forms. Poguli dūom retains its printed vowel sequence and Siraji dū̃m its nasalisation; both retain final m and select the dhūma branch rather than the i-extended dhūmikā branch.'), ('snow', ['śīin'], '12492', 'śīna', 'Turner12492 explicitly gives Kashmiri śīn snow/ice under śīna frozen, and discusses neighbouring hīn forms. Poguli śīin fits this regional ś-initial snow family; its source īi notation is preserved, not silently contracted to the comparator.'), ('snow', ['hīm'], '14096', 'hima', 'Turner14096 gives regional hīm/him snow, including Gawri, Pashai and Shina. Siraji hīm preserves m and fits this family, distinct from śīna12492 with final n; no direct borrowing is asserted without donor evidence.')]
a=[];audit=[];source=list(csv.DictReader(open('data/other/forms/raw_data/schmidt_kashmiri.csv')));raw=list(csv.reader(open('data/other/forms/20260725-schmidt-kashmiri.csv')))
lines=['# Schmidt and Kaul: Poguli and Siraji household and weather vocabulary','','Eleven reflex records across seven families; standing autonomous authorization. Table 3, printed pp. 287–288 (PDF pp. 57–58), visually checked.','','| # | Poguli / Siraji forms and meaning | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(g,fs,parent,label,ev) in enumerate(groups,1):
 hits=[r for r in rs if r['Gloss']==g and r['Form'] in fs];assert len(hits)==len(fs),(g,len(hits))
 for r in hits:
  rr=[z for z in raw if z[0]==r['Language_ID'] and z[2]==r['Original'] and z[3]==g and z[6]==r['Description']];assert len(rr)==1,(r,rr)
  item=r['Description'].split()[-1];sr=next(z for z in source if z['Item']==item)
  col={'pog':'Poguli','sir':'Siraji'}[r['Language_ID']];assert unicodedata.normalize('NFC',sr[col])==r['Original']
  page=int(sr['Page'])+230
  citation=f'schmidt[p. {page}, Table 3, item {item}];CDIAL[{parent} and addenda]'
  a.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=parent,kind='reflex',citation=citation,evidence=ev+' Source image and reviewed import row checked; no source transcription or identity changes.'))
  audit.append(dict(record_id=r['ID'],reviewed_source_row=sr,import_row=rr[0],printed_page=page,image=f'../tmp/pdfs/schmidt_kaul_2008/etymology/page-{sr["Page"]}.png',visual_check='confirmed'))
 label=label.replace('*',r'\*');lines.append(f'| {i} | **{" / ".join(fs)}** ‘{g}’ ({len(hits)}) | **{label}**, CDIAL **{parent}** | {ev} |')
h=[]
assert len(a)==11

for name,val in [('decisions',dict(accepted=a,held=h)),('source-audit',audit)]: (P/f'pass060-{name}.json').write_text(json.dumps(val,ensure_ascii=False,indent=2)+'\n')
lines+=['','All selected source forms were matched to the reviewed import rows and the previously inspected source images. No source transcription, grammatical tags or identities were changed.','','Discovery: schmidt-search010–012; full primary articles/addenda: schmidt-primary012–013. No rebuild.','','Save and validation pending.'];(P/'pass060-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted')
