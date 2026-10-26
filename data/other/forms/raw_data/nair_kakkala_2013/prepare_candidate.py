"""Source-local whole-chapter proposal; no canonical writer or inferred IPA."""
import csv
import hashlib
import json
from pathlib import Path
import unicodedata
import re

HERE=Path(__file__).resolve().parent
LECT='dialect:Kakkala:nair2013-kakkala:Kakkala%20%28Kerala%29'
FIELDS=['Language_ID','Parameter_ID','Form','Gloss','Native','Phonemic','Notes','Source','Cognateset_ID','Etymology','Entry_Key','Variant_Of_Key','Borrowed_From_Key','Derivation_Parent_Keys','Tags']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def tags(section,w,index):
 a=w['source_analysis'];f=w['form'];g=w['gloss'];ts=[]
 # Source categories are deliberately separated from conditioning prose.
 if 'suffix' in a or a.endswith(' marker') and section not in {'participle-zero-context'}:
  ts.append('suffix')
 if 'conditioning vowel' in a: ts=['stem']
 if section=='pronoun-paradigm':
  ts=['pron']
  person=a.split()[0];ts += [person+('sg' if 'singular' in a else 'pl')]
  if 'proximal' in a:ts+=['demonstrative','prox']
  if 'distal' in a:ts+=['demonstrative','dist']
  if 'masculine' in a:ts.append('m')
  if 'feminine' in a:ts.append('f')
  if 'neuter' in a:ts.append('n')
 elif section=='malayalam-pronoun-comparison':ts=['pron','demonstrative','n','pl']
 elif section in {'numerals','fractionals'}:ts=['num']
 elif section=='identity-context':ts=['proper-noun','multiword-expression','uncertain']
 elif section=='personal-suffixes':
  ts=['suffix','personal']
  person={'first':'1','second':'2','third':'3'}[a.split()[0]]
  if 'singular' in a:ts.append(person+'sg')
  elif 'plural' in a:ts.append(person+'pl')
  else:ts.append('third-person')
  if 'masculine' in a:ts.append('m')
  if 'feminine' in a:ts.append('f')
  if 'neuter' in a:ts.append('n')
 elif section=='personal-examples':
  ts=['verb']
  ts += [['1sg'],['1pl'],['2sg'],['2pl'],['3sg','m'],['3sg','f'],['3pl','f'],['3sg','n'],['3sg','n'],['3sg','n'],['3sg','n'],['3sg','n'],['3sg','n']][index-1]
  # The neuter examples have explicit past/present/future table scopes.
  if index in (8,9):ts.append('pret')
  if index in (10,11):ts.append('pres')
  if index in (12,13):ts.append('fut')
 elif section=='lexical-items':
  if g.startswith('to '):ts=['verb']
  if g in {'husband','wife'}:ts+=['kinship']
  if g=='you':ts+=['pron','second-person']
  if g=='he':ts+=['pron','3sg','m']
  if g=='that':ts+=['pron','demonstrative','dist']
 else:
  is_marker=not g
  if not is_marker:ts+=['verb']
  if section.startswith('past-'):ts+=['pret']
  if section.startswith('present-'):ts+=['pres']
  if section.startswith('future-'):ts+=['fut']
  if section.startswith('negative-'):ts+=['neg','participle']
  if section=='negative-markers' and index==1:ts.append('relative')
  if section=='negative-examples' and index<=2:ts.append('relative')
  if section.startswith('prohibitive-'):ts+=['neg','prohibitive']
  if section.startswith('purposive-'):ts+=['purposive']
  if section.startswith('imperative-') and 'conditioning' not in a:
   ts+=['impv'];ts+=['sg' if 'singular' in section else 'pl']
   if not is_marker:ts+=['2sg' if 'singular' in section else '2pl']
  if section.startswith('potential-'):ts+=['possibility']
  if section.startswith('optative-'):ts+=['opt']
  if section.startswith('participle-') and 'conditioning' not in a:ts+=['participle']
  if section=='participle-other-examples' and index<=2:ts.append('neg')
  if section=='participle-other-markers' and index==2:ts.append('neg')
  if section.startswith('relative-'):ts+=['relative','participle']
  if not is_marker:
   for needle,t in [('(I)','1sg'),('(she)','3sg'),('(he)','3sg'),('(He)','3sg'),('(it)','3sg'),('(they)','3pl')]:
    if needle in g:ts.append(t)
  if section=='past-neuter-marker':ts=['suffix','personal','third-person','n']
  if section=='past-neuter-examples':ts+=['3sg','n']
 if w['language_label']=='Tamil':ts=['suffix','honorific']
 if w['language_label']=='Malayalam' and section=='imperative-singular-markers':ts=['suffix','impv','sg','pl']
 if w['language_label']=='Kakkala':ts.append(LECT)
 if '´' in f:ts.append('uncertain')
 return list(dict.fromkeys(ts))

def main():
 units=[json.loads(x) for x in (HERE/'first-reading.jsonl').read_text().splitlines()]
 rows=[];audit=[];variants=[]
 for u in units:
  section=u['unit_key'].split(':',2)[2];outkeys=[];physical=[]
  for i,w in enumerate(u['witnesses'],1):
   key=w['witness_key'];f=w['form'];note=w['note'];lang=w['language_label']
   held=lang not in {'Kakkala','Malayalam','Tamil'} or not f or f=='-ø'
   disposition='documentary_or_identity_hold' if held else 'emitted'
   parts=f.split('/') if section=='pronoun-paradigm' and '/' in f else [f]
   keys=[]
   if not held:
    for v,form in enumerate(parts,1):
     ek=key+(f':v{v}' if len(parts)>1 else '')
     n=note
     if w['source_analysis']:n=(n+' Source analysis: '+w['source_analysis']+'.').strip()
     if lang!='Kakkala':n=f'Nair attributes this comparison to {lang}. '+n
     if section=='pronoun-paradigm' and i>=5:
      n+=' The table distinguishes masculine/feminine rows although adjacent prose describes common human gender; retain the source disagreement.'
     if '´' in form:n+=' Source prints a separate acute-like mark after s; attachment and phonemic interpretation remain unresolved.'
     if section=='past-analysis-table':n+=' The source prints separate decomposition and surface witnesses joined by an arrow.'
     if section=='lexical-items' and form=='ummaṇḍi':n+=' The detached following Tea, water line has unresolved gloss scope; preserved in audit.'
     if section=='personal-examples':n+=' Source spellings, including literal macrons and ň, remain unharmonized with other examples.'
     # Source grammatical labels stay verbatim in the physical audit; the
     # corresponding person/number tags already encode them in display.
     gloss=re.sub(r'\s*\((?:sg|pl)\.?\)', '', w['gloss'], flags=re.I)
     row=[lang,'',unicodedata.normalize('NFC',form),gloss,'','',n.strip(),
          f'nair2013kakkala[p. {u["printed_page"]}, {section}, witness {i}]','', '',ek,'','','',' '.join(tags(section,w,i))]
     rows.append(row);keys.append(ek);outkeys.append(ek)
    if len(parts)>1:
     variants.append(dict(child_key=keys[1],variant_of_key=keys[0],printed_page=u['printed_page'],source_cell_key=key,forms=parts,status='applied source slash variants; root original-reviewed'))
   physical.append(dict(physical_key=key,witness=w,disposition=disposition,entry_keys=keys,
                        hold_reason=('printed zero exponent; not a lexical form' if f=='-ø' else 'source identity or unpaired gloss unresolved') if held else ''))
  audit.append(dict(source_unit_key=u['unit_key'],printed_page=u['printed_page'],source_claim=u['source_claim'],note=u['note'],physical_witnesses=physical,entry_keys=outkeys))
 # One printed iva/ava token has both explicitly named language scopes.
 # The Kakkala readings already have exact table endpoints; preserve the
 # repeated source occurrence through citation reuse instead of a new head.
 bykey={r[10]:r for r in rows}
 for edge in variants:
  assert edge['child_key'] in bykey and edge['variant_of_key'] in bykey
  bykey[edge['child_key']][11]=edge['variant_of_key']
 comparison=next(a for a in audit if a['source_unit_key'].endswith(':malayalam-pronoun-comparison'))
 for physical,endpoint in zip(comparison['physical_witnesses'],['nair2013kakkala:p28:pronoun-paradigm:w9:v2','nair2013kakkala:p28:pronoun-paradigm:w11:v2']):
  assert bykey[endpoint][2]==physical['witness']['form']
  bykey[endpoint][7]+='; nair2013kakkala[p. 28, Malayalam comparison statement]'
  physical['explicit_target_scope_reuse']=endpoint
  comparison['entry_keys'].append(endpoint)
 assert len(rows)==202 and len(units)==41
 assert len({r[10] for r in rows})==len(rows)
 with (HERE/'proposed.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
 (HERE/'proposal-audit.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audit))
 (HERE/'graph-proposal.json').write_text(json.dumps(dict(source_explicit_variants=variants,held_relations=['past decomposition/surface arrows are analysis, not variants','prose/table spelling differences do not establish variants','cross-language resemblances do not establish donor or derivation edges']),ensure_ascii=False,indent=2)+'\n')
 status=dict(status='whole chapter source-local candidate; owner notation/metadata review and independent acceptance pending',source_pdf_sha256=units[0]['source_pdf_sha256'],
  source_pages=[28,29,30,31],physical_units=len(units),physical_witnesses=sum(len(u['witnesses']) for u in units),
  emitted_rows=len(rows),target_rows=sum(r[0]=='Kakkala' for r in rows),comparison_rows=sum(r[0]!='Kakkala' for r in rows),
  audit_only_witnesses=sum(not p['entry_keys'] for a in audit for p in a['physical_witnesses']),
  first_reading_sha256=sha(HERE/'first-reading.jsonl'),csv_sha256=sha(HERE/'proposed.csv'),audit_sha256=sha(HERE/'proposal-audit.jsonl'),
  graph_edges_applied=2,variant_candidates=2,canonical_changes=False)
 (HERE/'proposal-status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n');print(json.dumps(status))
if __name__=='__main__':main()
