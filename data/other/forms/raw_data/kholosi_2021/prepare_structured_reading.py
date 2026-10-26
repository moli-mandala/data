"""Lossless author-CSV review units including explicit paradigms, not installed output."""
from pathlib import Path
import csv,json,re,collections
P=Path(__file__).resolve().parent
pos={'noun':['noun'],'verb':['verb'],'adjective':['adj'],'numeral':['num'],'pronoun':['pron'],'adverb':['adv'],'postposition':['postp'],'adposition':[],'noun, adjective':['noun','adj'],'conjunction':['conj'],'determiner':['determiner'],'preposition':['prep'],'particle':['part'],'?':['uncertain']}
records=[]
for name in ('lexicon.csv','sentences.csv'):
 raw=list(csv.reader((P/'upstream'/name).open()))
 for n,fields in enumerate(raw[1:],2):
  d=dict(zip(raw[0][:14] if name=='lexicon.csv' else raw[0],fields));key=f'kholosi:github:18b43f982efa:{name}:row{n}'
  r=dict(source_unit_key=key,source_file=name,physical_csv_row=n,raw_fields=fields,raw_column_names=raw[0],review_status='pending independent complete-output review')
  if not fields[0]:r.update(status='blank',reason='Entire source row blank, no missing form inferred');assert not any(fields)
  elif name=='lexicon.csv':
   r.update(status='target',form=d['Transcription'],phonetic=d['Phonetic IPA'],gloss=d['Meaning'],tags=pos[d['Part of Speech']],etymology=d['Etymology'],comparison=d['Cognates'],notes=d['Notes'],source_qualifier=d['Citations'],paradigm_raw=d['Form'],recorded=d['Date'])
   r['pronunciation_evidence']={k:d[k] for k in ['IPA 2','IPA (OLD)','Syllables'] if d[k]};r['persian_prompt']=d['Persian Meaning'];r['nonlexical_trailing_columns']=fields[14:]
   if d['Form']=='feminine':r['tags'].append('f')
   if d['Part of Speech']=='adposition':r['grammar_uncertainty']='Source says adposition without a preposition/postposition subtype.'
   if any(c in r['form'] for c in '()'):r['alternate_segmentation_pending']=True
   if r['paradigm_raw'] and r['paradigm_raw']!='feminine':
    r['paradigm_candidates']=[]
    for j,m in enumerate(re.finditer(r'<([^>]+)>\s*([^<]+)',r['paradigm_raw']),1):
     label,form=m.groups();form=form.strip().rstrip(',').strip();known={'f':['f'],'pl':['pl'],'pl.obl':['pl','obl'],'comp':['degree']}
     assert label in known,label
     r['paradigm_candidates'].append(dict(source_unit_key=f'{key}:paradigm:{j}',form=form,source_grammar=label,tags=r['tags']+known[label],gloss=r['gloss'],notes='Source gives comparative degree.' if label=='comp' else '',relationship='inflected form under source head; exact relation awaiting review'))
  else:
   r.update(status='target',form=d['Transcription'],phonetic=d['Phonetic IPA'],gloss=d['Meaning'],interlinear_analysis=d['Gloss'],notes=d['Notes'],recorded=d['Date'],tags=['sentential'])
   if r['form'].endswith('?'):r['tags'].append('interr')
   if not r['gloss']:r['tags'].append('uncertain');r['gloss_uncertainty']='Attested response has no translation in the source; no gloss inferred.'
   if r['form']=='xôdafez':r['tags']=['interj']
   if any(c in r['form'] for c in '/()'):r['alternate_segmentation_pending']=True
   if '?' in r['interlinear_analysis'] and r['interlinear_analysis']!='?':r['analysis_uncertainty']='Source interlinear analysis itself includes question marks; retain without resolving.'
  records.append(r)
(P/'structured-first-reading.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
print('all',len(records),'targets',sum(r['status']=='target' for r in records),'blanks',sum(r['status']=='blank' for r in records),'paradigm_forms',sum(len(r.get('paradigm_candidates',[])) for r in records))
