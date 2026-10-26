"""Inventory every lexical quotation in the p237 linguistic commentary."""
import hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
first=BASE/'documentary-first-reading.jsonl'
assert sha(first)=='366389404bde9304290bc769fcaee22cb706c0febeb09cc844ed10b23c8f694b'
crop=HERE/'p237-linguistic-commentary-native.png'
provenance=json.loads((HERE/'crop-provenance.json').read_text())
assert sha(crop)==provenance['crop_sha256']
documentary=[json.loads(l) for l in first.read_text().splitlines()]
reviewed=[]
for r in documentary:
 if r['source_unit_key']=='lsi9.2:p237:mixed-Ahirwati-Bangaru-quoted-comparisons':
  r=dict(r)
  r['first_reading_documentary_text']=r['raw_documentary_first_reading']
  assert r['raw_documentary_first_reading'].count('bērē-mai')==1
  r['raw_documentary_first_reading']=r['raw_documentary_first_reading'].replace('bērē-mai','bērē-nai')
  r['root_literal_history']={'old':'bērē-mai','new':'bērē-nai','evidence':provenance,
                            'reason':'Original italic commentary prints one-hump n, independently corroborating the narrative reading.'}
  r['review_status']='Root original-commentary reading; independent full-block literal review pending'
  reviewed.append(r)
assert len(reviewed)==1

# Every separately printed quotation gets an occurrence key. Shared or repeated
# meanings are qualified in Notes; no glossary translation is manufactured.
specs=[
 ('ēk Ahīr dukhāḷō paṛō thō','mewati_jhambaus','Ahirwati',[], 'The source explicitly identifies this first sentence as good Ahirwati.'),
 ('us-kā jamāī bērē-nai āyā','kaithal','Bangaru',[], 'The source explicitly identifies this second sentence as good Bangaru.'),
 ('bōlō','mewati_jhambaus','Rohtak local form',['verb','participle','pret'],'Source-local form contrasted with Ahirwati bōlyō; the following explanation calls the y-dropping forms past participles.'),
 ('bōlyō','mewati_jhambaus','Ahirwati',['verb','participle','pret'],'Explicit Ahirwati comparator for local bōlō.'),
 ('bōlā','kaithal','Bangaru',['verb','participle','pret'],'Explicit Bangaru comparator; occurrence-specific mixed-source comparison.'),
 ('bōlō','mewati_jhambaus','Rohtak local form',['verb','participle','pret'],'Separately printed parenthetical past-participle example: bōlō for bōlyō.'),
 ('bōlyō','mewati_jhambaus','Ahirwati',['verb','participle','pret'],'Separately printed parenthetical past-participle comparator.'),
 ('aĩh','mewati_jhambaus','Rohtak Ahirwati',['pron','dem','obl'],'Source says this oblique form is used in place of nominative yō, this; the exact lexical Gloss is not separately printed here.'),
 ('yō','mewati_jhambaus','Ahirwati',['pron','dem','nom'],'Explicit nominative comparator, directly glossed this.')]
parents=[];children=[];emissions=[]
for index,(form,language,label,tags,note) in enumerate(specs,1):
 key=f'lsi9.2:p237:analysis-quotation:{index}'
 words=form.split()
 parent={'source_unit_key':key,'printed_page':237,'documentary_parent_key':reviewed[0]['source_unit_key'],
         'raw_form':form,'language':language,'source_language_label':label,'notes':note,
         'native_witness_sha256':provenance['source_sha256'],'native_crop_sha256':sha(crop),
         'kind':'separately printed linguistic quotation','root_review':'direct native-pixel reading; independent review pending'}
 parents.append(parent)
 for word_index,word in enumerate(words,1):
  child={**parent,'source_unit_key':key+f':word:{word_index}','physical_parent_key':key,'raw_form':word,
         'raw_gloss':'this' if index==9 else '', 'proposed_tags':tags,
         'graph_policy':'No inferred morphology, unattested bases or donor relations.'}
  children.append(child)
  emissions.append({**child,'kind':'atomic lexical quotation'})
 if len(words)>1:
  emissions.append({**parent,'kind':'whole quoted sentence','raw_gloss':'','proposed_tags':['sentential'],
                    'component_source_keys':[key+f':word:{w}' for w in range(1,len(words)+1)],
                    'component_graph_policy':'Inventory membership only; no morphological compound graph.'})
assert len(parents)==9 and len(children)==16 and len(emissions)==18
assert len({r['source_unit_key'] for r in emissions})==18
assert len([r for r in children if r['raw_gloss']])==1
def write(name,value,jsonl=True):
 text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in value) if jsonl else json.dumps(value,ensure_ascii=False,indent=2)+'\n'
 path=HERE/name
 if path.exists(): assert path.read_text()==text, 'Refuse change to an existing reading inventory'
 else: path.write_text(text)
write('linguistic-commentary-reviewed-derivative.jsonl',reviewed)
write('quoted-parent-census.jsonl',parents)
write('quoted-child-first-reading.jsonl',children)
write('quoted-emission-proposal.jsonl',emissions)
write('quoted-accounting.json',{'status':'Complete nine-quotation first reading; independent original and metadata review pending',
 'counts':{'quote_parents':9,'physical_lexical_atoms':16,'whole_sentences':2,'proposed_emissions':18,'directly_printed_atom_glosses':1},
 'scope':'Entire p237 linguistic commentary; every lexical quotation, including separately printed repeated parenthetical forms.',
 'source_labels':'Only explicitly Bangaru quotations route to Haryanvi/kaithal. Mixed narrative remains source-qualified Ahirwati.',
 'source_key_policy':'New quotation occurrences are distinct physical printings. Do not merge them with p238 narrative solely from matching strings.',
 'source_claims':[{'kind':'reported morphological variant','source_forms':['bōlyō','bōlō'],'direction':'y dropping in past participle','status':'source claim recorded, graph endpoints not yet integrated'}],
 'inputs':{'first_documentary_sha256':sha(first),'native_crop_sha256':sha(crop)},
 'outputs':{n:sha(HERE/n) for n in ['quoted-parent-census.jsonl','quoted-child-first-reading.jsonl','quoted-emission-proposal.jsonl','linguistic-commentary-reviewed-derivative.jsonl']},
 'canonical_changes':False,'full_ingestion_complete':False},False)
print(json.dumps({'status':'PASS','quote_parents':9,'atoms':16,'whole_sentences':2,'canonical_changes':False}))
