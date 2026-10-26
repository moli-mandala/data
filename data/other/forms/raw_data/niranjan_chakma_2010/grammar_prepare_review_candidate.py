"""Apply keyed second-reading corrections without overwriting the first reading.

Candidate only: no canonical writes, profile conversion, inferred translations or
phonetic reconstruction. Grammar claims come from exact source roles/headings.
"""
import ast
import collections
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
ALIASES = {
    'masculine':'m', 'feminine':'f', 'neuter':'n', 'common-gender':'mf',
    'indeclinable':'indecl', 'pronoun':'pron', 'verb-root':'stem',
    'present':'pres', 'past':'pret', 'future':'fut', 'continuous':'progressive',
    'imperative':'impv', 'negative':'neg', 'ablative':'abl', 'genitive':'gen',
    'locative':'loc', 'instrumental':'instr', 'recipient':'dat',
    'subject':'subj', 'object':'obj', 'agent':'subj',
}
TABLE_TAGS = {
    (15,3):['pres'], (15,4):['pres','progressive'],
    (16,1):['pres','progressive'], (16,2):['pres','perfect'],
    (16,3):['impv'], (17,1):['impv'], (17,2):['pret'],
    (17,3):['pret','habitual'], (18,1):['pret','habitual'],
    (18,2):['pret'], (19,1):['pret'],
}

def main():
    source = P/'grammar-first-reading.jsonl'
    review = json.loads((P/'grammar-second-reading-corrections.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == review['first_reading_sha256']
    records = [json.loads(line) for line in source.read_text().splitlines()]
    first_keys={r['source_unit_key'] for r in records}
    additions=json.loads((P/'grammar-second-reading-additions.json').read_text())['records']
    records.extend(additions)
    bykey = {r['source_unit_key']:r for r in records}
    assert len(bykey) == len(records)
    for change in review['corrections']:
        row = bykey[change['source_unit_key']]
        assert row[change['field']] == change['before'], change
        row[change['field']] = change['after']
        row.setdefault('second_reading_changes',[]).append(change)
    # Preserve old group strings as historical first readings, but make the
    # candidate group match corrected predicate cells rather than stale forms.
    groups = collections.defaultdict(list)
    for r in records:
        if r['role']=='predicate':
            c=r['table_context']; groups[(r['block_id'],c['physical_group'])].append(r['form'])
    for r in records:
        tags=[]; unresolved=[]
        for tag in r['source_tags']:
            if tag in {'finite','nonfinite'}:
                unresolved.append({'label':tag,'reason':'Exact broad source category retained; no narrower nonfinite subtype inferred.'})
            else: tags.append(ALIASES.get(tag,tag))
        c=r.get('table_context')
        if c:
            r['first_reading_table_context']=dict(c)
            c['printed_predicate_group']='/'.join(groups[(r['block_id'],c['physical_group'])])
            # Person labels in Bengali: উত্তম first, মধ্যম second, প্রথম third.
            if r['role'] in {'predicate','ending'}:
                person={'উত্তম':'1','মধ্যম':'2','প্রথম':'3'}[c['person_label_bn']]
                number={'একবচন':'sg','বহুবচন':'pl'}[c['number_label_bn']]
                tags.append(person+number)
                block=int(r['block_id'].split('block')[1])
                tags.extend(TABLE_TAGS.get((r['pdf_page'],block),[]))
                if (r['pdf_page'],block)==(19,2):
                    unresolved.append({'labels':['সাধারণ অতীত','Future Indefinite'],'reason':'Contradictory Bengali and English headings; both preserved, tense mapping awaiting editorial adjudication.'})
        if r['pdf_page']==20 and r['role']=='sentence':
            if r['block_id'].endswith('block2'):tags.append('interr')
            if r['block_id'].endswith('block3'):tags.append('neg')
        if r['disposition'] in {'control','source-blank'}:tags=[]
        r['candidate_tags']=sorted(set(tags))
        r['unmapped_explicit_categories']=unresolved
        r['review_status']='Owner second reading completed; independent glyph, typography and semantic review pending.'
    out=P/'grammar-review-candidate.jsonl'
    # Literal registry sets suffice for this check; do not import a compiled
    # pipeline or load CLDF merely to check the finite tag vocabulary.
    data_root=P.parents[4]
    registry=set()
    for node in ast.parse((data_root/'tags.py').read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'GRAMMATICAL_TAGS','GENDER_TAGS'} for t in node.targets):
            registry.update(ast.literal_eval(node.value))
    unknown={t for r in records for t in r['candidate_tags']}-registry
    assert not unknown,unknown
    out.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in records))
    report={
        'status':'Review candidate only; not accepted source-stage output',
        'records':len(records), 'all_first_reading_keys_preserved':first_keys<={r['source_unit_key'] for r in records},
        'additional_physical_occurrences':len(additions), 'unknown_candidate_tags':sorted(unknown),
        'first_reading_sha256':review['first_reading_sha256'],
        'candidate_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
        'keyed_corrections':len(review['corrections']),
        'second_read_pages':review['reviewed_pdf_pages'],
        'canonical_changed':False,
        'pending':['Independent fine-glyph decisions','Source-wide virama/disjoining final adjudication','Whole-book integration and source-stage gates'],
    }
    (P/'grammar-review-candidate-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
