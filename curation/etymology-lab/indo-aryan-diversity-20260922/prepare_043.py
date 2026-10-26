import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
linked={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
rs=[r for r in csv.DictReader(open('cldf/forms.csv')) if 'andersen1990[' in r['Source'] and r['ID'] not in linked]
groups=[
('ati','over','200','ati',134,'The scan explicitly gives RV ati and Pali ati under ati over. Turner200 is the matching Vedic preposition beyond/over; its discussion of later di forms is not used to reinterpret this unchanged glossary head. The separate atiyo excessively record is excluded.'),
('kʰo','indeed, really, surely','3846','khalu',146,'Andersen explicitly compares RV khalu and Pali kho. Turner3846 directly lists Ashokan kho and Pali kho under khalu certainly. The primary image confirms the particle and its three-part gloss.'),
('esa','this (following)','2530','eṣa',142,'Andersen heads esa, eta together and compares RV etad, Pali etad. Turner2530 explicitly unites nominative eṣa with the eta oblique/neuter paradigm and lists Ashokan esa, esā, etaṁ and eta. Preserve the imported cataphoric gloss and source locality tags; no claim that all cited alternative readings are identical.'),
('kal','to perform, to do','2814','karoti',144,'The scan gives kal- (kalati), RV kṛṇoti and Pali karoti, with kalaṃtaṃ as a present participle. Turner2814 directly includes Ashokan kaleti, kalāmi and kalaṁta. This is the r/l present-stem family, not a past-participle assignment.'),
('kāraṇa','deed, action, performance','3057','kāraṇa',144,'Andersen compares GB kāraṇa and Pali kāraṇa. Turner3057 lists Pali deed/cause and Ashokan kāraṇa/kālana purpose. The image confirms the noun independently of the following uncertain kāranaka scribe entry; no whole-compound analysis is asserted.')]
a=[];lines=['# Andersen Ashokan glossary: particles, pronoun and lexical heads','','Five proposed reflex records; standing autonomous authorization.','','| # | Ashokan Prakrit | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(f,g,parent,label,page,ev) in enumerate(groups,1):
 hits=[r for r in rs if r['Form']==f and r['Gloss']==g];assert len(hits)==1,(f,hits)
 r=hits[0];a.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=parent,kind='reflex',citation=r['Source']+';CDIAL['+parent+' and addenda]',evidence=ev+' Existing source spelling, citation and tags retained; source headword and comparison visually checked.'))
 lines.append(f'| {i} | **{f}** ‘{g}’ | **{label}**, CDIAL **{parent}** | {ev} |')
h=[]
for r in rs:
 if r['Form']=='adʰa' and r['Gloss']=='half':reason='Primary image p136 appears to print retroflex aḍha while installed form has dental adʰa; source transcription review required before assigning. No repair performed.'
 elif r['Form']=='gam' and r['Gloss']=='to go':reason='Source is a grammatical root head gam-, cross-referencing adhigam and upagata. Turner3955 is the specific gacchati present formation; resolve an appropriate existing root/family target rather than assuming that root and present are interchangeable.'
 else:continue
 h.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},reason=reason))
(P/'pass043-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=2)+'\n')
lines+=['','Two records held: adʰa half for a possible source transcription mismatch; gam go for root versus present-formation targeting.','Research: homepage andersen-search003–004; complete shortlisted CDIAL prose/addenda andersen-primary003. Existing images PDF134 (printed136), PDF142 (printed144), PDF144 (printed146), PDF146 (printed148) inspected.','Validation: pending.']
(P/'pass043-review.md').write_text('\n'.join(lines)+'\n')
audit=[r for r in csv.DictReader(open('.cache/ocr/andersen-mre/output/andersen_mre_entries.csv')) if r['pdf_page'] in ['134','142','144','146'] and r['headword'].rstrip('-') in ['ati','kho','esa','kal','kāraṇa','adha','gam']]
(P/'pass043-source-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(len(a),'accepted',len(h),'held',len(audit),'audit entries')
