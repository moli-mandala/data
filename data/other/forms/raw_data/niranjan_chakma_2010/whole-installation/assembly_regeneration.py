"""Whole-book source inventory, not an importer or acceptance mechanism.

Owner readings and unapproved changes remain separate. Only explicit root glyph
adjudications modify preview_reading; immutable evidence is linked and hash-pinned.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANONICAL = Path(__file__).resolve().parent / 'legacy-742.csv'
SECTIONS = {
 'grammar':('grammar-reviewed-reading-proposal.jsonl',682),
 'reverse':('reverse-reviewed-reading-proposal.jsonl',741),
 'homophones':('homophones-reviewed-reading-proposal.jsonl',427),
 'named-groups':('named-groups-reviewed-reading-proposal.jsonl',145),
 'abbreviations':('abbreviations-first-reading.jsonl',88),
 'numbered':('numbered-structured-review.jsonl',101),
 'kokborok':('kokborok-reviewed-reading-proposal.jsonl',68),
 'brokskat':('brokskat-reviewed-reading-proposal.jsonl',72),
 'santali':('santali-reviewed-reading-proposal.jsonl',96),
 'further':('further-reviewed-reading-proposal.jsonl',84),
 'basic':('basic-reviewed-reading.jsonl',99),
}
ADJUDICATIONS = ['thematic-root-glyph-adjudication-20260927.json','brokskat-root-glyph-adjudication-20260927.json','santali-root-glyph-adjudication-20260927.json','homophones-named-root-adjudication-20260927.json']
STATUS_FILES = ['grammar-review-reconciliation-status.json','reverse-review-reconciliation-status.json','basic-reviewed-reading-status.json','brokskat-review-reconciliation-status.json','santali-review-reconciliation-status.json','kokborok-review-reconciliation-status.json','further-review-reconciliation-status.json','homophones-review-reconciliation-status.json','named-groups-review-reconciliation-status.json','abbreviations-first-reading-status.json','numbered-structured-review-status.json']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def build(root=ROOT, canonical=CANONICAL):
 inputs={};out=[];stats={};accepted={};used=set()
 def load(name):
  p=root/name;inputs[name]=sha(p)
  return [json.loads(x) for x in p.read_text().splitlines()] if name.endswith('.jsonl') else json.loads(p.read_text())
 for name in ADJUDICATIONS:
  a=load(name)
  for d in a['decisions']:
   value=d.get('accepted_reading')
   if d.get('root_decision')=='accepted literal correction':value=d['proposed']
   if value is not None:
    k=d.get('source_entry_key',d.get('witness_key',d.get('source_unit_key')))
    assert k not in accepted
    accepted[k]={'old':d['old'],'new':value,'field':d.get('field','form'),'evidence_file':name,'decision':d}
 for name in STATUS_FILES:
  status=load(name)
  for rel,expected in status.get('output',{}).items():assert sha(root/rel)==expected,(name,rel)
 identity=load('numbered-final-identity-review-proposal.json')
 for name,expected in identity['inputs'].items():assert sha(root/name)==expected,name
 numbered_ids={r['comparison_key']:r for r in identity['comparators']}
 numbered_approval=load('numbered-identity-root-adjudication-20260927.json')
 for name,expected in numbered_approval['input_sha256'].items():assert sha(root/name)==expected,name
 numbered_root={r['source_unit_key']:r for r in numbered_approval['decisions']}
 assert set(numbered_root)==set(numbered_ids)
 extra_decisions=defaultdict(list)
 for filename in ['font-control-root-adjudication-20260927.json','homophones-named-proposals-root-adjudication-20260927.json','basic-root-adjudication-20260927.json','reverse-root-adjudication-20260927.json']:
  approval=load(filename)
  for name,expected in approval['input_sha256'].items():assert sha(root/name)==expected,name
  for d in approval['decisions']:extra_decisions[d['source_unit_key']].append({**d,'evidence_file':filename})
 font_approval=load('font-control-root-adjudication-20260927.json')
 font_evidence=defaultdict(list)
 for e in font_approval['candidate_uncertainty_evidence']:
  font_evidence[e['source_unit_key']].append(e)
  for w in e.get('witnesses',[]):
   if w.get('has_ha'):font_evidence[w['witness_key']].append(e)
  if e['id'].startswith('niranjan') and e['id']!=e['source_unit_key']:font_evidence[e['id']].append(e)
 prior_font=load('h-font-control-review/keyed-font-control-decisions-20260927.json')
 for e in prior_font['earlier_caution_policy']:font_evidence[e['source_unit_key']].append(e)
 inputs['TRANSCRIPTION_REVIEW_POLICY.md']=sha(root/'TRANSCRIPTION_REVIEW_POLICY.md')
 registry_proposal=load('language-registry-proposal.json')
 language_routes={r['source_unit_key']:r for r in registry_proposal['routes']}
 registry_path=root.parents[4]/'cldf/languages.csv'
 current_registry=list(csv.DictReader(registry_path.open()))
 assert all(r in current_registry for r in registry_proposal['records'])
 inputs['../../../../../cldf/languages.csv']=sha(registry_path)
 comparison_approval=load('comparison-root-adjudication-20260927.json')
 for name,expected in comparison_approval['input_sha256'].items():assert sha(root/name)==expected,name
 comparison_decisions=defaultdict(list)
 for d in comparison_approval['decisions']:comparison_decisions[d['witness_key']].append(d)
 grammar_approval=load('grammar-root-adjudication-20260927.json')
 for name,expected in grammar_approval['input_sha256'].items():assert sha(root/name)==expected,name
 grammar_decisions=defaultdict(list)
 for d in grammar_approval['decisions']:grammar_decisions[d['source_unit_key']].append(d)
 named_metadata=load('named-group-metadata-proposal.json')
 named_ids={r['source_unit_key']:r for r in named_metadata['records']}
 further_ids={r['witness_key']:r for r in load('further-language-mapping-review.json')['witnesses']}
 caution=load('independent-hasanta-h-font-control-reopening-20260927.json')
 load('independent-ha-glyph-caution-20260927.json')
 def emit(section,key,record,disposition,form='',gloss='',proposed=None,kind='unit',parent=None,lang=None,extra=None):
  assert key and not any(x['key']==key for x in out),key
  row={'key':key,'section':section,'kind':kind,'parent_key':parent,'disposition':disposition,'source_form':form,'preview_reading':form,'source_gloss':gloss,'proposed_fields_pending':proposed or {},'language_proposal':lang,'accepted_for_installation':False,'glyph_status':'owner/source reading retained; no blanket approval','provenance':{k:record[k] for k in ['pdf_page','printed_page','block','row','physical_row','column','printed_group','group_cell','block_id','source_line_keys','source_line_key','source_parts','original_bbox'] if k in record},'evidence':{'file':stats[section]['input'],'record_key':parent or key}}
  if key in accepted:
   a=accepted[key];field=a['field']
   if field in ['form','target_form']:
    assert form==a['old'],(key,form,a['old']);row['preview_reading']=a['new']
   elif field=='definition_bn':
    assert gloss==a['old'];row['preview_gloss']=a['new']
   else:raise AssertionError((key,field))
   row['accepted_root_glyph_delta']=a;row['glyph_status']='explicit root transcription correction only; installation pending';used.add(key)
  if extra:row.update(extra)
  if key in comparison_decisions:
   row['root_comparison_decisions']=comparison_decisions[key]
   row['accepted_root_comparison_fields']={}
   for d in comparison_decisions[key]:
    field=d['field'];old=row['source_form'] if field=='raw_owner_first_reading' else row['source_language_label']
    assert old==d['old'],(key,field,old,d['old'])
    if d['accepted_value'] is not None:
     row['accepted_root_comparison_fields'][field]=d['accepted_value']
     if field=='raw_owner_first_reading':
      row['preview_reading']=d['accepted_value']
      row['accepted_root_glyph_delta']={'old':old,'new':d['accepted_value'],'field':field,'evidence_file':'comparison-root-adjudication-20260927.json','decision':d}
      row['glyph_status']='explicit root transcription correction only; installation pending'
     else:
      assert field=='source_language_label',field
      row['preview_source_language_label']=d['accepted_value']
  if key in named_ids:
   row['named_group_metadata_proposal']=named_ids[key]
   row['language_proposal']=named_ids[key]['proposed_base_id']
   row['disposition']='unresolved Tanchangya base' if row['language_proposal'] is None else 'eligible target with source group provenance'
  if key in grammar_decisions:
   row['accepted_root_metadata_fields']={}
   row['root_grammar_decisions']=grammar_decisions[key]
   for d in grammar_decisions[key]:
    assert record[d['field']]==d['old'],(key,d['field'])
    if d['accepted_value'] is not None:
     assert d['field']!='form','Literal approval needs explicit overlay extension'
     row['accepted_root_metadata_fields'][d['field']]=d['accepted_value']
   row['metadata_status']='Only keyed accepted fields approved; literal/owner uncertainties preserved'
  if key in numbered_root:
   decision=numbered_root[key]
   assert row['source_form']==(decision['literal_form'] or ''),key
   row['root_identity_decision']=decision
   row['accepted_language_id']=decision['accepted_language_id']
   row['identity_disposition']=decision['disposition']
   row['required_identity_source_note']=decision['required_source_note']
   row['source_stage_or_variety']=decision['source_stage_or_variety']
   row['secondary_work_provenance']=decision['work_provenance']
  if key in language_routes:
   row['language_proposal']=language_routes[key]['proposed_base_id']
   row['language_registry_route_proposal']=language_routes[key]
   row['disposition']='eligible identified language; registered base'
   row['accepted_language_id']=language_routes[key]['proposed_base_id']
  if key in font_evidence:
   row['font_control_evidence']=font_evidence[key]
   row['required_tags']=list(dict.fromkeys(row.get('required_tags',[])+['uncertain']))
   row['typed_encoding_uncertainty']='marked-ha-encoding; source mark retained without phonological interpretation'
  if key in extra_decisions:
   row['additional_root_decisions']=extra_decisions[key]
   for d in extra_decisions[key]:
    field=d['field'];old=row['source_gloss'] if field in ['definition_bn','raw_bengali_prompt'] else (row['source_group_label'] if field=='source_group_label' else row['source_form'])
    assert old==d['old'],(key,field,old,d['old'])
    if d['accepted_value'] is not None:
     if field in ['definition_bn','raw_bengali_prompt']:row['preview_gloss']=d['accepted_value']
     elif field=='source_group_label':row['preview_source_group_label']=d['accepted_value']
     else:
      assert row['preview_reading'] in [old,d['accepted_value']],('Conflicting accepted readings',key)
      row['preview_reading']=d['accepted_value']
      if d['accepted_value']!=old:row['accepted_root_glyph_delta']={'old':old,'new':d['accepted_value'],'field':field,'evidence_file':d['evidence_file'],'decision':d}
     row['glyph_status']=d['root_decision']
     if section=='basic':row['disposition']='eligible target with typed uncertainty' if d['required_tags'] else 'eligible target reviewed'
    elif d['root_decision']=='audit-only unresolved glyph':row['disposition']='audit-only unresolved glyph';row['lexical_emission_permitted']=False
    row['required_tags']=list(dict.fromkeys(row.get('required_tags',[])+d.get('required_tags',[])))
    if d.get('typed_uncertainty'):row.setdefault('typed_root_uncertainties',[]).append({'type':d['typed_uncertainty'],'caveat':d.get('encoding_caveat',d.get('evidence'))})
  out.append(row)
  return row
 # Headerless canonical file: preserve all fifteen fields and exact Entry_Key.
 inputs['../../20260921-niranjan-chakma.csv']=sha(canonical)
 existing=list(csv.reader(canonical.open()));assert len(existing)==742 and all(len(r)==15 for r in existing)
 assert len({r[10] for r in existing})==742
 thematic=load('draft-audit.jsonl');assert len(thematic)==733
 stats['thematic']={'input':'draft-audit.jsonl','raw_units':733,'existing_rows':742}
 childmap={r[10]:r for r in existing};seenlegacy=set()
 for r in thematic:
  key=r['entry_key'];loc=r['locator'];children=[x[10] for x in r['emitted_rows']]
  emit('thematic',key,loc,'unresolved transcription/segmentation' if r['status']=='withheld' else 'physical source cell',r['review']['reviewed_chakma'],r['review']['reviewed_english'],extra={'child_keys':children,'source_bengali_control_raw':r['candidate']['cells'].get('bengali',[]),'uncertainties':r['uncertainty_types'],'source_note':r['reason']})
  for child in children:
   row=childmap[child];seenlegacy.add(child)
   emit('thematic',child,loc,'existing target row',row[2],row[3],kind='witness',parent=key,lang=row[0],extra={'Entry_Key':child,'canonical_fields_unchanged':row,'source_citation':row[7],'uncertainties':r['uncertainty_types']})
 assert seenlegacy==set(childmap)
 for section,(name,count) in SECTIONS.items():
  rs=load(name);assert len(rs)==count,(section,len(rs));stats[section]={'input':name,'raw_units':count}
  for r in rs:
   key=r['source_unit_key']
   common={'uncertainties':r.get('uncertainties',r.get('transcription_holds',[])),'source_note':r.get('source_note',r.get('review_note',''))}
   if section=='grammar':
    disp='control' if r['disposition']=='control' else ('blank' if not r['form'] else ('unresolved notation' if r['disposition']=='notation-uncertain' else 'eligible target pending review'))
    emit(section,key,r,disp,r['form'],r['gloss_bn'],r['reviewed_fields_proposal'],lang='Chakma',extra={**common,'source_gloss_en':r['gloss_en'],'candidate_tags':r['candidate_tags'],'role':r['role'],'typed_independent_uncertainties':r['independent_typed_uncertainties'],'review_changes_pending':r['independent_review_changes'],'unmapped_source_categories':r['unmapped_explicit_categories']})
   elif section=='reverse':
    emit(section,key,r,'eligible target pending review',r['raw_source_form'],r['raw_bengali_prompt'],{'form':r['reviewed_source_form_proposal'],'gloss_bn':r['reviewed_bengali_prompt_proposal']},lang='Chakma',extra={**common,'review_findings':r['independent_review_evidence']['findings'],'font_control_holds':r['font_control_holds']})
   elif section in ['homophones','named-groups']:
    emit(section,key,r,'unresolved named-language/lect attribution' if section=='named-groups' or 'routing unresolved' in r.get('language_scope','') else 'eligible target pending review',r['target_form'],r['definition_bn'],r['reviewed_field_proposals'],lang=None if section=='named-groups' or 'routing unresolved' in r.get('language_scope','') else 'Chakma',extra={**common,'source_group_label':r.get('source_group_label'),'source_qualifiers':r.get('source_qualifiers',[]),'transcription_uncertainty':r.get('transcription_uncertainty',r.get('reading_note')),'review_findings':r['second_reading_evidence']['findings']})
   elif section=='abbreviations':
    emit(section,key,r,'source metadata declaration',r['abbreviation_first_reading'],r['expansion_first_reading'],extra={'contextual_identity_only':True,'review_note':r['review_note']})
   elif section=='basic':
    emit(section,key,r,'unresolved glyph' if r['transcription_hold'] else 'eligible target pending review',r['raw_owner_first_reading'],r['printed_english'],{'form':r['reviewed_reading']},lang='Chakma',extra={'bengali_control':r['bengali_control'],'bengali_control_disposition':'control' if r['bengali_control'] else 'explicit blank','uncertainties':[r['typed_remaining_uncertainty']] if r['typed_remaining_uncertainty'] else [],'source_caveats':r['source_caveats']})
   elif section=='numbered':
    children=([r['head']['key']] if r['head'] else [])+[c['key'] for c in r['comparisons']]+[c['key'] for c in r['source_claims']]
    emit(section,key,r,'headless commentary group' if r['head'] is None else 'numbered physical group',extra={'child_keys':children,'line_accounting':r['line_audit'],'uncertainties':r['transcription_holds']})
    if r['head']:
     h=r['head'];emit(section,h['key'],h,'eligible target pending review',h['literal_form'],r['definition']['literal'],kind='witness',parent=key,lang='Chakma',extra={'head_qualifier':h.get('qualifier'),'uncertainties':r['transcription_holds']})
    for c in r['comparisons']:
     ident=numbered_ids[c['key']]
     disp='blank' if c['disposition']=='explicit source blank' else ('unresolved label/form segmentation' if 'segmentation' in c['disposition'] else ('control comparator outside requested families' if ident['scope_disposition'].startswith('outside requested') else ('eligible comparator pending identity approval' if ident['scope_disposition']=='target-family identity proposal' else 'unresolved language comparator')))
     emit(section,c['key'],c,disp,c.get('literal_form') or '',c.get('source_specific_meaning') or '',kind='witness',parent=key,lang=ident['proposed_base_id'],extra={'identity_proposal':ident,'raw_source_value':c['raw_source_value'],'source_language_label':c['source_language_label'],'shared_definition_not_automatically_assigned':r.get('definition'),'uncertainties':r['transcription_holds']})
    for c in r['source_claims']:
     emit(section,c['key'],c,'headless/source analysis control',kind='source_claim',parent=key,extra={'source_claim':c})
   else:
    witnesses=r['witnesses']+r.get('embedded_comparisons',[])
    emit(section,key,r,'paired comparison source row',extra={**common,'child_keys':[w['witness_key'] for w in witnesses],'printed_meaning':r['printed_meaning_first_reading']})
    for w in witnesses:
     wk=w['witness_key'];ident=further_ids.get(wk);lang=ident['proposed_language_id'] if ident else w.get('proposed_language_id')
     disp='eligible target/comparator pending review' if lang else ('control comparator outside requested families' if section=='kokborok' or (ident and 'outside' in ident['identity_status']) else 'unresolved language comparator')
     emit(section,wk,r,disp,w['raw_owner_first_reading'],w.get('scoped_meaning_first_reading',''),{'form':w.get('reviewed_reading_proposal',w['raw_owner_first_reading']),'language_label':w.get('reviewed_label_proposal',w['source_language_label'])},kind='witness',parent=key,lang=lang,extra={'source_language_label':w['source_language_label'],'identity_proposal':ident,'review_changes_pending':w.get('review_changes',[]),'uncertainties':w.get('review_holds',[]),'meaning_scope':w.get('meaning_scope','embedded explicit comparison')})
 doc=load('documentary-pages-review-20260927.json');stats['documentary']={'input':'documentary-pages-review-20260927.json','raw_units':4,'source_metadata_rows':94}
 for r in doc['pages']:emit('documentary',r['source_unit_key'],r,'documentary metadata; no lexical attestations',extra={'source_table_rows':r['table_rows']})
 assert used==set(accepted),(set(accepted)-used)
 frontmatter=load('frontmatter-scope-review-20260927.json')
 assert len(frontmatter['pages'])==7 and frontmatter['target_lexical_units']==0
 stats['frontmatter']={'input':'frontmatter-scope-review-20260927.json','raw_units':7}
 for r in frontmatter['pages']:emit('frontmatter',r['source_unit_key'],r,r['disposition'],extra={'not_a_lexical_row':True,'original_review_proof':r['proof'],'source_documentary_content':r['source_documentary_content'],'source_note':r['scope_note']})
 for section,s in stats.items():
  selected=[r for r in out if r['section']==section];s['preview_records']=len(selected);s['dispositions']=dict(Counter(r['disposition'] for r in selected));s['explicit_root_glyph_deltas']=sum('accepted_root_glyph_delta' in r for r in selected)
 status={'status':'whole-book inventory preview; no installation approval','sections':stats,'preview_records':len(out),'existing_entry_keys_preserved':742,'root_glyph_corrections_applied':sum('accepted_root_glyph_delta' in r for r in out),'root_comparison_fields_accepted':sum(len(r.get('accepted_root_comparison_fields',{})) for r in out),'root_metadata_fields_accepted':sum(len(r.get('accepted_root_metadata_fields',{})) for r in out),'all_rows_installation_pending':True,'inputs':inputs,'script_sha256':sha(Path(__file__)),'canonical_changes':False,'metadata_gaps':['Named-group labels remain exact provenance, not invented geographic dialects; two source-grounded language additions Tanchangya and Chenchu are ready for root registry approval.','Numbered comparison mappings require final policy approval; generic Munda and other ambiguous labels remain distinct from legible known-language forms.','Further comparison unresolved identities and source-specific varieties remain typed.','Four auxiliary work-level references are registered per root; retain exact numbered-reference-mapping provenance as secondary through Niranjan, never independently collated witnesses. Grammar-adaptation bibliography remains pending.','Bengali-script preservation and punctuation/hasanta policy require complete output/profile validation; no IPA invented.','Front/back matter PDF1–6/110 census closed with original page proof; no target lexical quotations.'],'pending':['Remaining exact non-ha glyph/segmentation decisions and language identities; marked-ha encoding can proceed with mandatory uncertainty, not indefinite blanket holds.','Explicit alternative/blank/qualified-meaning accounting before canonical forms.','Fresh independent integrated-output audit and focused canonical validations.','Full compiled/database/browser gates deferred; no builds authorized.'],'policy':['No source-unit deduplication or inferred graph edges.','Preview reading changes only from explicit root transcription adjudications.','Unapproved reviewed fields and language mappings remain separately labelled proposals.','Every immutable evidence file is hash-pinned; record locators preserve full source detail without duplicating OCR.']}
 # Recheck all pinned bytes at exit to catch concurrent regeneration.
 for name,digest in inputs.items():assert sha(canonical if name=='../../20260921-niranjan-chakma.csv' else root/name)==digest,name
 return out,status

def main():
 rows,status=build();target=ROOT/'whole-source-assembly-preview.jsonl';target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));status['output']={target.name:sha(target)};(ROOT/'whole-source-assembly-status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:status[k] for k in ['preview_records','existing_entry_keys_preserved','root_glyph_corrections_applied','canonical_changes','output']}))
if __name__=='__main__':main()
