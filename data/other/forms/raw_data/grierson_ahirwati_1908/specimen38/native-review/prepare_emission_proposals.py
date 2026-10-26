"""Full No38 source-scoped metadata, physical join and whole quotation proposals."""
import copy,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[7]
OWNER=HERE.parent
sys.path.insert(0,str(ROOT/'data'))
from tags import GRAMMATICAL_TAGS,GENDER_TAGS
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(l) for l in p.read_text().splitlines()]
def save(name,value,lines=True):
    text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in value) if lines else json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    path=HERE/name
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
reviewed=OWNER/'root-reviewed-revision/p238-atomic-reviewed.jsonl'
atoms=read(reviewed);assert len(atoms)==133
first=read(OWNER/'p238-roman-first-reading.jsonl')
original=[a for line in first for a in line['target_atoms']]
assert [a['source_unit_key'] for a in original]==[a['source_unit_key'] for a in atoms]
assert sum(bool(a['typed_uncertainties']) for a in original)==10
assert all(a['typed_uncertainties']==b['typed_uncertainties'] for a,b in zip(original,atoms))
acceptance=json.loads((OWNER/'root-reviewed-revision/root-literal-acceptance.json').read_text())
assert sha(reviewed)==acceptance['outputs']['p238-atomic-reviewed.jsonl']
grammar=HERE.parents[1]/'grammar/child-witnesses-reviewed.jsonl'
index={(a['physical_line'],a['physical_word']):a for a in atoms}
keys={a['source_unit_key']:a for a in atoms}
positions={a['source_unit_key']:n for n,a in enumerate(atoms)}
head=index[(4,10)];tail=index[(5,1)]
assert head['raw_form']=='Jai-kaḷi-' and tail['raw_form']=='kai'
assert head['raw_gloss']=='Jai-kaḷi-' and tail['raw_gloss']=='to'
join={'emission_key':head['source_unit_key'],'physical_source_keys':[head['source_unit_key'],tail['source_unit_key']],
 'emission_form':'Jai-kaḷi-kai','emission_gloss':'Jai-kaḷi-to',
 'raw_parts':[{'key':r['source_unit_key'],'form':r['raw_form'],'gloss':r['raw_gloss']} for r in [head,tail]],
 'source_type':'printed compound physically continues across lines4–5; no added letters',
 'status':'source-supported join proposal; independent boundary review pending'}
save('physical-break-join-proposal.jsonl',[join])

# Explicit analysis is deliberately narrower than English narrative morphology.
# Numeral meanings and speech-person labels are literal interlinear evidence;
# no gender, case or TAM is guessed from the English subject or translation.
metadata=[]
for a in atoms:
    clean=a['raw_form'].strip('‘’.,;?')
    tags=[];basis=[]
    if clean in ['tīn','ikīs']:
        tags=['num'];basis=['Printed numeral meanings three/twenty-one in the counted source phrases.']
    if clean in ['bōlō','bōlā']:
        tags=['verb'];basis=['p237 separately quotes these as spoken lexical verb forms.']
        if clean=='bōlō':tags+=['participle','pret'];basis+=['p237 explicitly calls y-dropping bōlō/bōlyō past-participle forms.']
    if clean=='aĩh':
        tags=['pron','demonstrative','obl'];basis=['p237 explicitly calls aĩh oblique, used instead of nominative yō; actual usage is retained without assigning a new case.']
    if clean=='Yō':
        tags=['pron','demonstrative','nom'];basis=['p237 explicitly labels the same nominative yō comparator; capitalization is kept literal.']
    if clean=='sai':
        tags=['verb','copula','pres'];basis=['p51 source present verb-substantive forms plus exact aligned is/art usage; no person inferred for this ambiguous shared surface.']
    if clean=='thō':
        tags=['verb','copula','pret'];basis=['p51 explicitly past thō; no natural gender or person inferred.']
    if not basis:basis=['Printed interlinear meaning retained. No additional grammatical class assigned without source-explicit analysis.']
    if a['typed_uncertainties']:tags.append('uncertain')
    assert set(tags)<=GRAMMATICAL_TAGS|GENDER_TAGS
    metadata.append({'key':a['source_unit_key'],'old_form':a['raw_form'],'old_gloss':a['raw_gloss'],
      'proposed_tags':tags,'source_scoped_basis':basis,'literal_uncertainties':a['typed_uncertainties'],
      'source_lect':'Ahirwati of Jhajjar/Rohtak, explicitly mixed Bangaru usage',
      'language':'mewati_jhambaus','proposed_dialect_id':'lsi1908_ahirwati_rohtak_jhajjar',
      'phonemic':'','graph_claims':[],'status':'complete keyed owner metadata proposal; independent scope review pending'})
save('all-133-keyed-metadata-proposal.jsonl',metadata)
meta={m['key']:m for m in metadata}
emissions=[];accounting=[]
for a in atoms:
    k=a['source_unit_key'];joined=k==head['source_unit_key'];is_tail=k==tail['source_unit_key']
    accounting.append({'physical_source_key':k,'emission_key':head['source_unit_key'] if is_tail else k,
                       'disposition':'physical continuation accounted through first key' if is_tail else 'atomic lexical emission proposal'})
    if is_tail:continue
    r=copy.deepcopy(a)
    r.update(emission_key=k,emission_form=join['emission_form'] if joined else a['raw_form'],
             emission_gloss=join['emission_gloss'] if joined else a['raw_gloss'],
             physical_source_keys=join['physical_source_keys'] if joined else [k],
             raw_parts=join['raw_parts'] if joined else [{'key':k,'form':a['raw_form'],'gloss':a['raw_gloss']}],
             tags=meta[k]['proposed_tags'],language=meta[k]['language'],proposed_dialect_id=meta[k]['proposed_dialect_id'],
             emission_native='',phonemic='',native_policy='Broad native script spans are provenance, not certified word-equivalent Native fields.')
    emissions.append(r)
assert len(emissions)==132 and len(accounting)==133
assert [k for r in emissions for k in r['physical_source_keys']]==[a['source_unit_key'] for a in atoms]
save('atomic-emissions-proposal.jsonl',emissions)
save('atomic-emission-accounting.jsonl',accounting)

# Source punctuation and dialogue scope determine boundaries, never physical rows.
spans=[((1,1),(1,5),'narrative sentence'),((1,6),(1,9),'narrative sentence'),
 ((2,1),(2,8),'narrative sentence'),((2,9),(3,9),'narrative and embedded question'),
 ((3,5),(3,9),'embedded question'),((3,10),(4,4),'narrative and embedded reply'),
 ((4,2),(4,4),'embedded reply'),((4,5),(4,7),'short question with quoted narrative connective'),
 ((4,6),(4,7),'embedded short question'),((4,8),(5,2),'whole quoted reply'),
 ((5,3),(5,10),'narrative and embedded question'),((5,7),(5,10),'embedded question'),
 ((6,1),(6,2),'narrative connective and short answer'),((6,2),(6,2),'short answer'),
 ((6,3),(7,3),'whole two-sentence quoted speech'),((7,4),(8,8),'narrative and two-sentence quoted reply'),
 ((7,8),(8,8),'whole quoted reply'),((8,9),(9,8),'narrative and quoted question/explanation'),
 ((9,2),(9,8),'whole quoted question/explanation'),((10,1),(11,5),'narrative and embedded request'),
 ((10,5),(11,5),'embedded request'),((11,6),(15,2),'narrative and whole multi-sentence reply'),
 ((11,9),(15,2),'whole multi-sentence quoted reply')]
native=read(HERE/'native-roman-span-alignment-proposal.jsonl')
whole=[]
for rank,(first,last,kind) in enumerate(spans,1):
    lo=positions[index[first]['source_unit_key']];hi=positions[index[last]['source_unit_key']]
    selected=atoms[lo:hi+1];sk={a['source_unit_key'] for a in selected}
    touching=[r for r in emissions if sk.intersection(r['physical_source_keys'])]
    assert all(set(r['physical_source_keys'])<=sk for r in touching),'Boundary bisects printed compound'
    form=' '.join(r['emission_form'] for r in touching)
    analysis=' '.join(a['raw_gloss'] for a in selected)
    tags=[] if kind in ['short answer','narrative connective and short answer'] else ['sentential']
    if '?' in form:tags.append('interr')
    uncertainties=[u for a in selected for u in a['typed_uncertainties']]
    if uncertainties:tags.append('uncertain')
    whole.append({'key':f'lsi9.2:specimen38:whole-utterance:{rank}',
      'first_word_key':selected[0]['source_unit_key'],'last_word_key':selected[-1]['source_unit_key'],
      'physical_source_keys':[a['source_unit_key'] for a in selected],
      'raw_source_parts':[{'key':a['source_unit_key'],'form':a['raw_form'],'gloss':a['raw_gloss']} for a in selected],
      'emission_form':form,'printed_interlinear_analysis':analysis,'emission_gloss':analysis,
      'translation_type':'exact concatenated source interlinear English; not author free translation',
      'source_free_translation':None,'lexical_free_gloss':None,'tags':tags,'scope':kind,
      'typed_uncertainties':uncertainties,'language':'mewati_jhambaus',
      'proposed_dialect_id':'lsi1908_ahirwati_rohtak_jhajjar','phonemic':'','emission_native':'',
      'native_provenance_span_keys':[r['key'] for r in native if sk.intersection(r['roman_keys'])],
      'native_alignment_policy':'Overlapping broad native spans cited as provenance only; no claim that their entire native text is equivalent to this narrower utterance.',
      'graph_claims':[],'stage':'complete source-anchored owner whole/speech proposal; independent boundary/metadata acceptance pending'})
assert set(k for r in whole for k in r['physical_source_keys'])==set(keys)
save('whole-utterances-proposal.jsonl',whole)
outputs=['physical-break-join-proposal.jsonl','all-133-keyed-metadata-proposal.jsonl','atomic-emissions-proposal.jsonl',
         'atomic-emission-accounting.jsonl','whole-utterances-proposal.jsonl']
save('emission-proposal-status.json',{'status':'complete No38 owner emission/metadata/whole-span proposal; unaccepted',
 'counts':{'physical_atoms':133,'joined_atomic_emissions':132,'line_break_joins':1,'whole_speech_spans':len(whole),
           'literal_holds_retained':10,'keyed_metadata_rows':133},
 'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [reviewed,OWNER/'root-reviewed-revision/root-literal-acceptance.json',
      OWNER/'p238-roman-first-reading.jsonl',HERE/'native-roman-span-alignment-proposal.jsonl',grammar]},
 'output_hashes':{n:sha(HERE/n) for n in outputs},'script_sha256':sha(Path(__file__)),
 'conservation':['All133 physical keys preserved in132 emissions exactly once; only one explicit cross-line compound joined.',
                 'All133 physical keys covered by whole/speech quotation proposals; overlapping dialogue spans distinguished.',
                 'Every literal uncertainty retained; no native or phonemic guess; all target keys remain mixed historical lect.'],
 'pending':['Independent metadata/boundary review; root free-translation accounting integration; independent native-glyph acceptance; whole-source export/audit and required focused gates.']},False)
print(json.dumps({'atoms':133,'emissions':132,'whole_spans':len(whole),'status_sha256':sha(HERE/'emission-proposal-status.json')}))
