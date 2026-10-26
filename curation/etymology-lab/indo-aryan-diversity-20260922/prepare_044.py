# Original four-record research generator. Final decisions hold sāvāna after source audit revealed merged sāvāpiteṇe provenance; do not rerun over final decisions.
import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
linked={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'};linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
rs=[r for r in csv.DictReader(open('cldf/forms.csv')) if 'andersen1990[' in r['Source'] and r['ID'] not in linked]
groups=[('porāna','old, ancient, former','8420','reflex','Turner8420 paurāṇa explicitly lists Ashokan porāṇā feminine singular and Pali/Prakrit pōrāṇa ancient. This specific o-vowel family is preferred to purāṇa8283; Andersen compares RV purāṇa at a broader historical level, retained here as an explicit source comparison. Image PDF158 confirms porāna and its regional retroflex variant porāṇa.'),('porāṇa','','f_oker3etbf3tqo','variant','Andersen PDF158 explicitly prints porāṇa, 2R [Br,Sd], see porāna. Preserve this source-defined retroflex spelling as a variant pointing to the dental-n glossary head, rather than manufacturing a gloss or merging the records.'),('sāvana','announcement, edict','12698','reflex','Andersen PDF173 explicitly compares Sanskrit śrāvaṇa and Pali sāvana announcing. Turner12698 śrāvaṇa¹ announcing gives Pali sāvana and Prakrit sāvaṇa/sāvaṇā. This is the action noun, not the month-name śrāvaṇa12699 or the shorter hearing noun śravaṇa12688. Alternative inscription readings in the source locality list remain preserved.'),('sāvāna','','f_dgnyetyrlbn2m','variant','Andersen PDF173 explicitly prints sāvāna, 1J [Sa], see sāvana. Preserve the source-defined long medial-vowel variant and its separate existing record, with the normal glossary head as its target.')]
a=[]
for f,g,parent,kind,ev in groups:
 hits=[r for r in rs if r['Form']==f and r['Gloss']==g];assert len(hits)==1,(f,hits)
 r=hits[0];a.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=parent,kind=kind,citation=r['Source']+((';CDIAL['+parent+' and addenda]') if kind=='reflex' else ''),evidence=ev+' Primary headword and comparison/cross-reference visually verified; no source transcription or identities changed.'))
(P/'pass044-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=2)+'\n')
audit=[r for r in csv.DictReader(open('.cache/ocr/andersen-mre/output/andersen_mre_entries.csv')) if r['pdf_page'] in ['158','173'] and r['headword'].rstrip('-') in ['porāna','porāṇa','sāvana','sāvāna']]
(P/'pass044-source-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
lines=['# Andersen: old and announcement, with source variants','','Four proposed records: two reflex and two variant rows, under standing autonomous authorization.','','| # | Ashokan Prakrit | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(f,g,parent,kind,ev) in enumerate(groups,1):
 label={'8420':'paurāṇa, CDIAL 8420','12698':'śrāvaṇa¹, CDIAL 12698','f_oker3etbf3tqo':'Variant of porāna','f_dgnyetyrlbn2m':'Variant of sāvana'}[parent]
 lines.append(f'| {i} | **{f}** ‘{g or "cross-referenced spelling variant"}’ | **{label}** | {ev} |')
lines+=['','The two variant records retain their empty stored glosses; meanings are supplied by their explicit source cross-references. Other candidates su and stone-pillar are not assigned in this pass.','Research: homepage andersen-search005–007, complete CDIAL articles andersen-primary004–005, existing source images PDF158/173 (printed160/175).','Validation: pending.']
(P/'pass044-review.md').write_text('\n'.join(lines)+'\n')
print(len(a),'accepted',len(audit),'cached audit entries')
