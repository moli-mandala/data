import csv,json,unicodedata
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
linked={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
rs=[r for r in json.loads((P/'schmidt-poguli-siraji-remaining057.json').read_text()) if r['ID'] not in linked]
groups=[
('eight',['aht','atʰ'],'941','aṣṭā','Turner941 explicitly gives Poguli āhṭ eight and regional aṭh/āṭh. The source Poguli aht and Siraji atʰ fit this branch; their consonant order, aspiration and unmarked length remain as recorded.'),
('five',['pā̃ʦ'],'7655','pañca','Turner7655 explicitly gives Poguli pā̃ċ five beside Kashmiri panċ and Pahari panċ. Poguli pā̃ʦ preserves the nasal vowel and affricate spelling; the more complex Siraji form is held.'),
('nine',['nãŭ','nao'],'6984','nava²','Turner6984 explicitly gives Poguli nāu and Pahari nao nine. Poguli nãŭ and Siraji nao fit these contracted regional forms; nasalisation and diphthong notation are retained.'),
('one',['ekə'],'2462-2','*ēkka','Turner2462 section2 gives Pahari and regional ek/ēk one from the reinforced *ēkka stem. Siraji ekə retains k and final ə, fitting that branch; the alternative unreinforced section1 is not selected from the surface spelling alone.'),
('seven',['sat'],'13139','sapta','Turner13139 gives Poguli satt seven alongside regional sat/satt. The source Poguli sat has the diagnostic s-t frame; its single t is preserved. The Siraji spelling with an additional r is held.'),
('six',['śeh','śah'],'12803','ṣaṣ','Turner12803 section1 gives Kashmiri śeh and Ḍoḍi śah, and also lists Poguli śe. The source Poguli śeh and Siraji śah belong to this ś-initial branch, rather than the separate Indo-Aryan ch/cha branch in section3; the h is retained.'),
('ten',['dah','daś'],'6227','daśa','Turner6227 explicitly gives Kashmiri dah and neighbouring Pahari daś ten from daśa. Poguli dah and Siraji daś retain their distinct final consonants; no modern donor is inferred.'),
('three',['trey'],'5994','trayaḥ','Turner5994 section1 gives NiDoc treya/tre and neighbouring Ḍoḍi trēī three. Siraji trey fits the tr-initial masculine series; its written final y is retained. Poguli cayi instead needs the c-initial section2 correspondence checked.'),
('two',['dih','duí'],'6648','dva','Turner6648 explicitly gives Poguli dīh and regional dui/dūī two. The source Poguli dih and Siraji duí fit these two branches of the same dva entry; their vowel length and final stress are preserved.')]
a=[];audit=[];source=list(csv.DictReader(open('data/other/forms/raw_data/schmidt_kashmiri.csv')));raw=list(csv.reader(open('data/other/forms/20260725-schmidt-kashmiri.csv')))
lines=['# Schmidt and Kaul: Poguli and Siraji basic numerals','','Fourteen reflex records across nine numeral families; standing autonomous authorization. Table 3, printed pp. 298–300 (PDF pp. 68–70), visually checked.','','| # | Poguli / Siraji forms and meaning | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(g,fs,parent,label,ev) in enumerate(groups,1):
 hits=[r for r in rs if r['Gloss']==g and r['Form'] in fs];assert len(hits)==len(fs),(g,len(hits))
 for r in hits:
  rr=[z for z in raw if z[0]==r['Language_ID'] and z[2]==r['Original'] and z[3]==g and z[6]==r['Description']];assert len(rr)==1,(r,rr)
  item=r['Description'].split()[-1];sr=next(z for z in source if z['Item']==item)
  col={'pog':'Poguli','sir':'Siraji'}[r['Language_ID']];assert unicodedata.normalize('NFC',sr[col])==r['Original']
  page=int(sr['Page'])+230
  citation=f'schmidt[p. {page}, Table 3, item {item}];CDIAL[{parent} and addenda]'
  a.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=parent,kind='reflex',citation=citation,evidence=ev+' Printed source image and reviewed import row checked; no source transcription or identity changes.'))
  audit.append(dict(record_id=r['ID'],reviewed_source_row=sr,import_row=rr[0],printed_page=page,image=f'../tmp/pdfs/schmidt_kaul_2008/etymology/page-{sr["Page"]}.png',visual_check='confirmed'))
 label=label.replace('*',r'\*');lines.append(f'| {i} | **{" / ".join(fs)}** ‘{g}’ ({len(hits)}) | **{label}**, CDIAL **{parent}** | {ev} |')
reasons={
('pog','yah','one'):'Turner gives Poguli yakh under *ēkka; the source yah lacks k. Establish whether the difference is regular loss or a transcription/lexical distinction before linking.',
('sir','păʦʰ','five'):'The printed form has a distinct vowel and aspirated affricate. Determine whether this is the expected local pañca outcome before copying Poguli pā̃ʦ.',
('pog','ʦarvar','four'):'Printed form appears tsawar; compiled ʦarvar needs transcription-profile review and the whole formation needs a specific catvāri branch.',
('sir','ʦut','four'):'Printed form appears tsur; compiled ʦut differs at the final consonant. Source/profile review is needed before selecting a branch.',
('sir','sartʰ','seven'):'Printed form appears satʰ; compiled sartʰ contains an additional r. Resolve source/profile discrepancy before assigning sapta.',
('pog','cayi','three'):'Turner5994 section2 explicitly gives Poguli cāē, but the reviewed source writes caji and compiled cayi. Resolve that vowel/glide correspondence before assigning a section.'}
h=[dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},reason=reasons[(r['Language_ID'],r['Form'],r['Gloss'])]) for r in rs if (r['Language_ID'],r['Form'],r['Gloss']) in reasons]
assert len(a)==14 and len(h)==6,(len(a),len(h))
for name,val in [('decisions',dict(accepted=a,held=h)),('source-audit',audit)]: (P/f'pass062-{name}.json').write_text(json.dumps(val,ensure_ascii=False,indent=2)+'\n')
lines+=['','Six records held:']+[f'- **{x["record"]["Form"]}** ‘{x["record"]["Gloss"]}’: {x["reason"]}' for x in h]+['','Discovery: schmidt-search017; full primary articles and separate addenda: schmidt-primary020–021. Written-source numeral notation and the source-specific uncertainty are retained. No rebuild.','','Save and validation pending.']
(P/'pass062-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(h),'held')
