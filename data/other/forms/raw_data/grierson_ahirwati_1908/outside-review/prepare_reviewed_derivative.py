"""Guarded source-only outside repair/scope derivative; historical reading intact."""
import copy,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[6]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(name,value,lines=False):
    text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in value) if lines else json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    path=HERE/name
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
path=HERE/'child-witnesses-first-reading.jsonl'
assert sha(path)=='93cd9c5f83facf5e01ea0e59f16c0fc58080146778323d2c6d08b648c0249846'
source=[json.loads(l) for l in path.read_text().splitlines()];assert len(source)==62
reviewed=copy.deepcopy(source);changes=[]
for old,new in zip(source,reviewed):
    if old['parent_key']=='lsi9.2:p149:outside-lead-scope:genitive-spellings':
        assert 'kō/kā̃' in old['source_note']
        new['source_note']=old['source_note'].replace('kō/kā̃','kō/kā')
    if old['key']=='lsi9.2:p149:outside-lead-scope:genitive-spellings:w3':
        assert old['source_form']=='kā̃' and old['raw_native']=='काँ'
        new['source_form']='kā';new['raw_native']=None
        new['historical_typed_uncertainties']=new['typed_uncertainties'];new['typed_uncertainties']=[]
        new['proposed_tags']=[t for t in new['proposed_tags'] if t!='uncertain']
        new['review']='Native original corroborates straight quantity bar; no Devanagari spelling is printed for this Roman-only occurrence. Initial raw_native was a transcription error.'
    if old['key'] in ['lsi9.2:p151:outside-standard-past:f:pl:w1','lsi9.2:p151:context-imperfect:f:pl:w1']:
        assert old['source_form'] in ['hī','mārē-hī']
        new['typed_uncertainties']=['Plural feminine final upper stroke is small/bent; hī/hī̃ remains an unresolved source-glyph alternative after singular/plural controls, not an emendation from morphology.']
        if 'uncertain' not in new['proposed_tags']:new['proposed_tags'].append('uncertain')
    eligible=old['source_type_role'] in ['eligible-attributed-comparison','additional-source-scope-lead'] or old['parent_key']=='lsi9.2:p149:outside-lead-scope:b-w-pronoun'
    new['reviewed_disposition']='eligible source-comparand emission proposal' if eligible else 'audit-only surrounding control'
    if eligible:
        new['proposed_language_id']='bagri_fatehabad'
        new['proposed_dialect_id']='lsi1908_bagri_bikaner' if old['printed_page']==149 else 'lsi1908_bagri_sirsa' if old['parent_key'].startswith('lsi9.2:p151:outside-Bagri-present:') else 'lsi1908_bagri_hissar'
    new['root_scope_review']='Root complete source review: 8 Sirsa,3 Hissar past,2 later Hissar leads and2 Bikaner bō/wō rows eligible; remaining surrounding controls audit-only.'
    if old!=new:changes.append({'key':old['key'],'old_source_form':old['source_form'],'reviewed_source_form':new['source_form'],
       'changed_fields':[k for k in set(old)|set(new) if old.get(k)!=new.get(k)],
       'old_raw_native':old['raw_native'],'reviewed_raw_native':new['raw_native']})
assert [r['key'] for r in source]==[r['key'] for r in reviewed]
assert sum(a['source_form']!=b['source_form'] for a,b in zip(source,reviewed))==1
assert sum(a['source_gloss']!=b['source_gloss'] for a,b in zip(source,reviewed))==0
eligible=[r for r in reviewed if r['reviewed_disposition'].startswith('eligible')];assert len(eligible)==15
whole=next(r for r in reviewed if r['parent_key']=='lsi9.2:p151:context-Hissar-imperfect-example')
assert whole['source_form']=='hū̃ mārᵃtō-thō'
physical=[]
for rank,form in enumerate(whole['source_form'].split(),1):
    physical.append({'key':whole['key']+f':physical-word:{rank}','physical_parent_interpretation_key':whole['key'],
      'printed_page':151,'raw_form':form,'raw_gloss':'','source_type_role':'physical component of source whole example',
      'physical_span_shared_with_whole':True,'independent_lexical_emission':'not authorized by this physical accounting',
      'proposed_tags':[],'source_analysis':'Source labels the whole example imperfect and describes present-participle construction.',
      'typed_uncertainties':['Independent function of first hū̃ (pronoun versus copula) is not assigned; no atom gloss borrowed from the whole heading.'] if rank==1 else whole['typed_uncertainties'],
      'image':whole['image'],'image_sha256':whole['image_sha256']})
assert len(physical)==2
whole['physical_source_keys']=[r['key'] for r in physical]
whole['source_gloss_scope']='strike is a qualified contextual lexical gloss from printed heading I was striking, etc.; not a printed word-by-word free translation.'
put('child-witnesses-reviewed.jsonl',reviewed,True)
put('Hissar-whole-physical-atoms.jsonl',physical,True)
routes=json.loads((HERE/'identity-and-provenance-proposal.json').read_text())
routes['proposed_historical_lects'].append({'id':'lsi1908_bagri_bikaner','language':'bagri_fatehabad','name':'Bikaner (LSI1908 Bagri)','latitude':'','longitude':'','glottocode':''})
routes['stage']='Root scope-approved historical comparison route proposals; shared registry still untouched'
put('reviewed-identity-and-provenance-proposal.json',routes)
put('reviewed-derivative-continuity.json',{'status':'Exact guarded owner derivative after root outside scope/glyph review; fresh independent lexical/output acceptance remains separate',
 'counts':{'historical_interpretations_retained':62,'eligible_comparands':15,'surrounding_audit_only':47,
           'additional_physical_word_records':2,'additional_printed_source_tokens_invented':0,
           'literal_form_corrections':1,'erroneous_native_field_removed':1,'gloss_changes':0},
 'historical_input_sha256':sha(path),'firstreading_status_sha256':sha(HERE/'first-reading-status.json'),
 'output_hashes':{name:sha(HERE/name) for name in ['child-witnesses-reviewed.jsonl','Hissar-whole-physical-atoms.jsonl','reviewed-identity-and-provenance-proposal.json']},
 'changes':changes,'root_decisions':['Plain Roman kā and no printed native counterpart; feminine plural hī/hī̃ holds remain; masculine plural hā unchanged.',
           'Bikaner bō/wō one pair retained, Sirsa parallel only source note.',
           'Hissar whole source key preserved while its two physical words are separately accounted, not independently interpreted.'],
 'canonical_changes':False})
print(json.dumps({'eligible':15,'retained':62,'physical_components':2,'continuity_sha256':sha(HERE/'reviewed-derivative-continuity.json')}))
