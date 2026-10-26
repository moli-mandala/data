"""Independent complete p237 quotation review; root first readings untouched."""
import copy,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[7]
OWNER=HERE.parent/'root-documentary-review'
sys.path.insert(0,str(ROOT/'data'))
from tags import GRAMMATICAL_TAGS,GENDER_TAGS
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):return [json.loads(l) for l in (OWNER/name).read_text().splitlines()]
def save(name,value,lines=False):
    text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in value) if lines else json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    path=HERE/name
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
parents=read('quoted-parent-census.jsonl');source=read('quoted-emission-proposal.jsonl')
assert len(parents)==9 and len(source)==18
expected=['ēk Ahīr dukhāḷō paṛō thō','us-kā jamāī bērē-nai āyā','bōlō','bōlyō','bōlā','bōlō','bōlyō','aĩh','yō']
assert [p['raw_form'] for p in parents]==expected
assert [p['language'] for p in parents]==['mewati_jhambaus','kaithal','mewati_jhambaus','mewati_jhambaus','kaithal','mewati_jhambaus','mewati_jhambaus','mewati_jhambaus','mewati_jhambaus']
revised=copy.deepcopy(source);changes=[]
for old,new in zip(source,revised):
    if old['source_unit_key']=='lsi9.2:p237:analysis-quotation:5:word:1':
        assert old['proposed_tags']==['verb','participle','pret']
        new['proposed_tags']=['verb']
        new['notes']+=' The explicit past-participle/y-dropping analysis scopes bōlō/bōlyō; no participle or past tag inferred for this Bangaru comparison.'
    if old['source_unit_key'] in ['lsi9.2:p237:analysis-quotation:8:word:1','lsi9.2:p237:analysis-quotation:9:word:1']:
        assert 'dem' in old['proposed_tags']
        new['proposed_tags']=['demonstrative' if t=='dem' else t for t in old['proposed_tags']]
    if old!=new:changes.append({'key':old['source_unit_key'],'old_form':old['raw_form'],'old_gloss':old['raw_gloss'],
                              'old_tags':old['proposed_tags'],'proposed_tags':new['proposed_tags'],
                              'old_notes':old['notes'],'proposed_notes':new['notes']})
    for field in ['raw_form','raw_gloss','language','source_unit_key']:assert old[field]==new[field]
    assert set(new['proposed_tags'])<=GRAMMATICAL_TAGS|GENDER_TAGS
    new['independent_review']='Complete original p237 linguistic commentary: literals, attribution and exact analysis scope reviewed.'
assert len(changes)==3
save('quoted-emissions-reviewed-derivative.jsonl',revised,True)
save('independent-quoted-comparison-review.json',{'status':'All nine quotation groups/18 proposals independently reviewed; three guarded metadata repairs staged',
 'counts':{'parent_quotes':9,'physical_lexical_atoms':16,'whole_sentences':2,'emissions':18,'material_literal_corrections':0,'metadata_repairs':3},
 'parent_decisions':[{'key':p['source_unit_key'],'form':p['raw_form'],'language':p['language'],
   'literal_and_attribution':'agrees','direct_gloss_scope':'Only yō has directly printed this; other quotation words remain blank-gloss or source-analysed.'} for p in parents],
 'changes':changes,'inputs':{name:sha(OWNER/name) for name in ['quoted-parent-census.jsonl','quoted-child-first-reading.jsonl','quoted-emission-proposal.jsonl','p237-linguistic-commentary-native.png','crop-provenance.json']},
 'reviewed_derivative_sha256':sha(HERE/'quoted-emissions-reviewed-derivative.jsonl'),
 'policy':['Separately printed quotes are physical occurrences, not automatic aliases of p238 narrative.',
           'No new ancestry/component edges; inventory membership is distinct from morphological derivation.',
           'The source literal letter y is documentary morphology notation, not a separately attested lexical base.'],
 'root_concurrence':'Root agreed demonstrative enum repair and withholding Bangaru bōlā participle/pret; source rows unchanged.',
 'canonical_changes':False})
print(json.dumps({'review_sha256':sha(HERE/'independent-quoted-comparison-review.json'),'derivative_sha256':sha(HERE/'quoted-emissions-reviewed-derivative.jsonl')}))
