"""Emit fifteen-column source preview and full inventory audit; never install."""
import csv,hashlib,io,json,re
from collections import Counter
from pathlib import Path
from assembly_regeneration import ROOT, CANONICAL, build, sha

def prepare():
 inventory,status=build();rows=[];audit=[];excluded=Counter();bykey={r['key']:r for r in inventory}
 def citation(r):
  p=r['provenance'];bits=[]
  if p.get('printed_page') is not None:bits.append('p. '+str(p['printed_page']))
  if p.get('pdf_page') is not None:bits.append('PDF '+str(p['pdf_page']))
  for field in ['block','column','row','physical_row','printed_group','group_cell']:
   if field in p:bits.append(field.replace('_',' ')+' '+str(p[field]))
  if not bits:
   lines=p.get('source_line_keys',[]) or ([p['source_line_key']] if p.get('source_line_key') else [])
   pages=sorted({int(m.group(1)) for x in lines if (m:=re.search(r':pdf(\d+):',x))})
   if pages:bits.append('PDF '+','.join(map(str,pages)))
  # Exact stable unit is a locator, not an inferred printed number.
  bits.append('unit '+r['key'])
  return 'niranjan2010chakma['+', '.join(bits)+']'
 def reason(r):
  d=r['disposition']
  if r.get('lexical_emission_permitted') is False:return 'unresolved damaged glyph; explicit source audit exclusion'
  if r['kind']=='source_claim':return 'source analysis/headless commentary retained as evidence'
  if r['section'] in ['abbreviations','documentary','frontmatter']:return 'documentary/control metadata, not lexical attestation'
  if r['section']=='thematic' and r['kind']=='unit':return 'physical cell accounted by child rows' if r.get('child_keys') else 'unresolved source-cell transcription/segmentation'
  if d in ['paired comparison source row','numbered physical group','headless commentary group']:return 'physical container accounted by children or headless source claims'
  if d in ['control','blank','unresolved notation']:return d
  if 'outside requested' in d:return 'outside requested language families; comparison retained'
  if 'segmentation' in d:return 'unresolved label/form segmentation'
  if r.get('root_identity_decision') and r['root_identity_decision']['accepted_language_id'] is None and not r.get('language_registry_route_proposal'):return 'unresolved comparator language identity'
  if not r.get('accepted_language_id',r.get('language_proposal')):return 'unresolved source language identity'
  if not r['preview_reading'].strip():return 'blank source form'
  return None
 for r in inventory:
  why=reason(r);a={'source_unit_key':r['key'],'section':r['section'],'kind':r['kind'],'disposition':'audit-only' if why else 'lexical-preview','reason':why,'assembly_record_key':r['key'],'emitted_entry_keys':[],'original_provenance':r['provenance'],'original_evidence':r['evidence'],'literal_source_form':r['source_form'],'literal_source_gloss':r['source_gloss'],'source_analysis':r.get('source_claim'),'source_identity_decision':r.get('root_identity_decision')}
  if why:excluded[why]+=1;audit.append(a);continue
  if 'canonical_fields_unchanged' in r:
   row=list(r['canonical_fields_unchanged']);assert row[10]==r['key'];row[2]=row[4]=r['preview_reading']
   tags=row[14].split();notes=row[6]
  else:
   lang=r.get('accepted_language_id') or r['language_proposal'];form=r['preview_reading'];meta=r.get('accepted_root_metadata_fields',{})
   gloss=r.get('preview_gloss',meta.get('gloss_bn',r.get('source_gloss','')))
   if r['section']=='grammar' and r.get('source_gloss_en'):gloss=r['source_gloss_en']
   tags=list(meta.get('candidate_tags',r.get('candidate_tags',[])));notes=meta.get('source_note',r.get('source_note',''))
   row=[lang,'',form,gloss,form,'',notes,citation(r),'','',r['key'],'','','','']
  notesparts=[notes] if notes else []
  if r.get('source_group_label'):notesparts.append('Source group attribution: '+r.get('preview_source_group_label',r['source_group_label'])+'.')
  if r.get('source_language_label'):notesparts.append('Source language label: '+r.get('preview_source_language_label',r['source_language_label'])+'.')
  if r.get('source_qualifiers'):notesparts.append('Source qualifiers: '+'; '.join(r['source_qualifiers'])+'.')
  if r.get('required_identity_source_note'):notesparts.append(r['required_identity_source_note'])
  if r.get('language_registry_route_proposal'):notesparts.append(r['language_registry_route_proposal']['required_source_note'])
  if r.get('bengali_control'):notesparts.append('Printed Bengali control: '+r['bengali_control']+'.')
  if r.get('shared_definition_not_automatically_assigned'):notesparts.append('Printed head definition (not independently assigned to this comparator): '+r['shared_definition_not_automatically_assigned']['literal'])
  if r.get('secondary_work_provenance'):
   for w in r['secondary_work_provenance']:notesparts.append('Secondary quotation through Niranjan: '+w['title']+'; no independently consulted passage or edition.')
  uncertainties=[]
  for field in ['uncertainties','typed_independent_uncertainties','transcription_uncertainty','typed_root_uncertainties','typed_remaining_uncertainty']:
   if r.get(field):uncertainties.append({'field':field,'evidence':r[field]})
  if r.get('font_control_evidence'):uncertainties.append({'field':'marked-ha-encoding','evidence':r['font_control_evidence']});notesparts.append('Visible marked হ is preserved with provisional Unicode encoding; no phonological interpretation is asserted.')
  pending=[d for d in r.get('additional_root_decisions',[]) if d['accepted_value'] is None]
  pending += [d for d in r.get('root_comparison_decisions',[]) if d['accepted_value'] is None]
  pending += [d for d in r.get('root_grammar_decisions',[]) if d['accepted_value'] is None]
  if pending:uncertainties.append({'field':'held source reading proposal','evidence':pending})
  tags+=r.get('required_tags',[])
  if uncertainties:tags.append('uncertain');notesparts.append('Source transcription/scope uncertainty is retained; exact alternatives and page evidence are recorded in the keyed source audit.')
  if ' ' in row[2]:tags.append('multiword-expression')
  row[6]=' '.join(dict.fromkeys(notesparts));row[14]=' '.join(dict.fromkeys(tags))
  assert len(row)==15 and row[2] and row[10]
  rows.append(row);a.update(emitted_entry_keys=[row[10]],source_form=r['source_form'],preview_form=row[2],literal_root_changes=r.get('accepted_root_glyph_delta'),typed_uncertainties=uncertainties,root_metadata_fields=r.get('accepted_root_metadata_fields',{}),source_controls={'bengali':r.get('bengali_control'),'english':r.get('source_gloss_en'),'source_claims':r.get('source_claim')})
  audit.append(a)
 assert len(audit)==len(inventory)==5232
 assert len({r[10] for r in rows})==len(rows)
 old=list(csv.reader(CANONICAL.open()));assert {r[10] for r in old}<={r[10] for r in rows}
 # Keep all 742 historical rows first, in original order.
 index={r[10]:r for r in rows};legacy={r[10] for r in old};rows=[index[r[10]] for r in old]+[r for r in rows if r[10] not in legacy]
 summary={'status':'complete section-accounted exporter preview; not accepted for installation','rows':len(rows),'audit_units':len(audit),'legacy_keys_preserved':742,'language_counts':dict(Counter(r[0] for r in rows)),'section_rows':dict(Counter(a['section'] for a in audit if a['emitted_entry_keys'])),'exclusions':dict(excluded),'uncertain_rows':sum('uncertain' in r[14].split() for r in rows),'inputs':status['inputs'],'assembly_script_sha256':status['script_sha256'],'exporter_sha256':sha(Path(__file__)),'canonical_csv_sha256':sha(CANONICAL),'pending':['Fresh whole-output original-source audit; this preview does not approve unreviewed extraction.','Final source metadata/citation/profile/parser and closed-key validation.','Review source analyses/components retained in audit before declaring every eligible lexical unit emitted.','Exact residual identities/damage exclusions remain keyed, not silently dropped.'],'deferred':['Full database/CLDF build and browser gates: not authorized in this source-stage task.']}
 return rows,audit,summary

def main():
 rows,audit,status=prepare();csvpath=ROOT/'whole-source-export-preview.csv';auditpath=ROOT/'whole-source-export-audit.jsonl'
 with csvpath.open('w',newline='') as f:csv.writer(f).writerows(rows)
 auditpath.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit))
 status['outputs']={p.name:sha(p) for p in [csvpath,auditpath]};(ROOT/'whole-source-export-status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:status[k] for k in ['rows','audit_units','legacy_keys_preserved','uncertain_rows','section_rows','exclusions']}))
if __name__=='__main__':main()
