"""Read-only source-metadata recommendations; never mutates owner readings."""
from pathlib import Path
import ast,hashlib,json
P=Path(__file__).resolve().parent
REPO=P.parents[4]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
table=P/'table-reviewed-reading.json'
expected='26ec49104bf99d932994bfbf42607973c828d9d4fc6ac34d8c1a475eef2da34e'
assert sha(table)==expected,'Owner input changed; review must be rebased explicitly.'
data=json.loads(table.read_text());prompts={r['prompt_id']:r for r in map(json.loads,(P/'table-prompt-inventory.jsonl').read_text().splitlines())}
tagpath=REPO/'tags.py'
registered=set()
for node in ast.parse(tagpath.read_text()).body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['GRAMMATICAL_TAGS','GENDER_TAGS'] for t in node.targets):
        registered.update(ast.literal_eval(node.value))

finite=set(range(156,168))|{172,173,174}|set(range(179,195))|set(range(201,217))
questions={220,221,222,223,239,240}
imperative_sentences={227,234,235,236,237,238}
nonfinite={170,171,177,178,218,219}
items=[]
for r in data['readings']:
    n=r['prompt_id'];old=r.get('tags',[]);tags=[{'prs':'pres','imp':'impv'}.get(t,t) for t in old]
    notes=[];reasons=[]
    def add(*values,reason):
        before=set(tags)
        for value in values:
            if value not in tags:tags.append(value)
        if set(tags)!=before:reasons.append(reason)
    if tags!=old:reasons.append('Draft prs/imp are not registered tags or aliases; explicit registered equivalents are pres/impv.')
    if 32<=n<=76 or 101<=n<=131 or 138<=n<=155:
        add('noun',reason='The lexical prompt names a nominal item or the explicit noun/number/case paradigm; no gender class or unattested morphology is inferred.')
    if n in {16,19,22,25,28,31}:
        add('poss',reason='The source distinguishes possessive mine/our/thine/yours/his/their from its separate of-pronoun prompts; existing genitive analysis retained.')
    if n in {92,93,94}:add('interr',reason='Explicit interrogative who/what/why prompt; no extra pronoun POS imposed on why.')
    if n in {95,96}:add('conj',reason='Explicit conjunction prompts and/but.')
    if n==97:
        add('conditional',reason='Source explicitly calls the response a suffix under if; conditional function is attested.')
        notes.append('Printed if response is explicitly labelled a suffix. No boundary hyphen supplied in literal Form.')
    if n==100:add('interj',reason='Explicit exclamation alas.')
    if n in {105,114,123}:add('pl',reason='Explicit two fathers/daughters/good men gives plural reference, not a source grammatical dual.')
    if n in {134,137}:
        add('degree',reason='Explicit best/highest degree prompts; no registered superlative tag is invented.')
        notes.append('Source prompt is superlative: '+prompts[n]['english_prompt']+'.')
    if 138<=n<=155:
        notes.append('Preserve the exact source animal-sex/number prompt; lexical male/female distinctions alone do not establish a target grammatical gender class.')
    if 156<=n<=174:add('copula',reason='The explicit to-be paradigm and chapter copula discussion identify this lexical function; no new forms or internal morphology inferred.')
    if n in finite:add('sentential',reason='Complete finite person-marked statement supplied by the source; distinct from isolated nonfinite forms.')
    if n in nonfinite:
        notes.append('Literal English nonfinite prompt: '+prompts[n]['english_prompt']+'. Retain verb without inferring a present/past participle, gerund, or conjunctive subtype from English -ing/having/gone alone. Chapter paradigms distinguish verbal nouns, conjunctive and relative participles in other named lects; no exact identity proof supplied here.')
    if n in {191,192}:
        add('progressive',reason='Explicit whole-clause am/was beating construction; tag describes the prompted construction, not an inferred target suffix.')
    if n==193:
        add('perfect',reason='Explicit had beaten is a past-perfect construction; existing pret retained, no passive analysis inferred.')
        notes.append('Source English prompt is past perfect (I had beaten); the tag applies to the complete construction.')
    if 220<=n<=240:add('sentential',reason='Source supplies a complete question, statement or command, not an isolated phrase.')
    if n in questions:add('interr',reason='Complete explicit source question.')
    if n in imperative_sentences:add('verb','impv',reason='Complete explicit source imperative command; person/number not inferred from English imperative alone.')
    if n==241:
        assert 'sentential' not in tags
        notes.append('This source response is the prepositional answer fragment to preceding From whom did you buy that?, not a complete independent sentence. Preserve multiword-expression without sentential or an invented finite verb.')
    if any(' ' in form.strip() for form in r['forms']):
        add('multiword-expression',reason='At least one exact source response contains printed spaces; final exporter must apply this per emitted alternative.')
    if r.get('uncertainty'):
        add('uncertain',reason='Preserved current typed transcription uncertainty must remain visible on emitted rows; historical superseded uncertainty must instead stay in audit.')
    assert set(tags)<=registered,(n,set(tags)-registered)
    items.append(dict(prompt_id=n,source_unit_key=prompts[n]['key'],english_prompt=prompts[n]['english_prompt'],
        old_tags=old,proposed_tags=tags,notes_additions=notes,reasons=reasons,
        literal_changes=False,status='metadata correction proposed' if tags!=old or notes else 'existing metadata compatible',
        source_forms=r['forms']))

written={r['prompt_id'] for r in data['readings']}
assert len(items)==233 and len(written)==233
report={'status':'complete independent metadata review; owner reconciliation pending, not installation approval',
 'reviewer':'rohru_finish','input_sha256':expected,'prompt_inventory_sha256':sha(P/'table-prompt-inventory.jsonl'),'registered_tags_sha256':sha(tagpath),
 'coverage':{'all_written_cells':233,'all_source_prompts':241,'nonwritten_prompts':[{'prompt_id':n,'english_prompt':prompts[n]['english_prompt'],'disposition':'explicit comparative blank' if n in [133,136] else 'source future cross-reference; no newly reconstructed form'} for n in sorted(set(prompts)-written)]},
 'changed_tag_cells':sum(i['old_tags']!=i['proposed_tags'] for i in items),
 'items':items,
 'material_adjudications':[
 {'proposal':'all220–241 sentential','decision':'Reject for241 only: From a shopkeeper of the village is an answer fragment. Approve220–240.'},
 {'proposal':'all170/171/177/178/218/219 participle','decision':'Do not accept generic English-to-target morphology transfer. Retain verb, exact nonfinite prompt in Notes/audit; finer subtype requires same-lect/source-specific evidence.'},
 {'proposal':'prs/imp alias normalization','decision':'Required explicit pres/impv correction: neither draft token is registered nor present in GRAMMATICAL_ALIASES.'},
 {'proposal':'grammatical gender from animal sex','decision':'Do not infer. Keep exact sex distinction in Gloss and source notes; no automatic m/f tag.'},
 ],
 'export_requirements':['Preserve all explicit source alternatives and scope; do not split internal coordinated-sentence commas.','Keep original prompt labels in audit; strip successfully parsed parenthetical past-tense metadata from lexical gloss where exporter separates lexical gloss and tags.','Apply actual-space multiword-expression per emitted form; no hyphen-only automatic multiword rule.','Keep source numeral optional-ending parentheses and exact suffix notes until source-specific expansion is accepted.','Preserve first-reading history and root-adjudicated literal changes; this report proposes zero Form changes.'],
 'prompt157_independent_read':{'form':'Inu māñji','image':'tmp/kui-table157-djvu-original.png','sha256':sha(REPO.parent/'tmp/kui-table157-djvu-original.png'),'finding':'Inu clear; separate short horizontal stroke above a, left of higher ñ mark, supports māñji. Coarse ink warrants exact upper-mark caveat. No paradigm reconstruction.'}}
(P/'independent-table-metadata-review-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'written_cells':len(items),'changed_tag_cells':report['changed_tag_cells'],'report_sha256':sha(P/'independent-table-metadata-review-20260927.json')},indent=2))
