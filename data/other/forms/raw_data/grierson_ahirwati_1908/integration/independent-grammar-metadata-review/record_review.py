"""Complete source-grounded metadata/graph review; recommendations only."""
import collections,csv,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1]
ROOT=next(p for p in HERE.parents if (p/'data/make_cldf.py').exists())
sys.path.insert(0,str(ROOT/'data'))
from tags import GENDER_TAGS,GRAMMATICAL_TAGS
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(s) for s in p.read_text().splitlines()]
INPUT=HERE.parent/'grammar-metadata-proposal-r2'
proposal_path=INPUT/'all98-keyed-grammar-metadata-proposal.jsonl'
graph_path=INPUT/'source-graph-proposal.jsonl'
source_path=PACKAGE/'grammar/child-witnesses-reviewed.jsonl'
assert sha(proposal_path)=='89025ac3ebf122a8db117de76dd408116eaf7f88c499f46c98cf95fc007ff9a3'
assert sha(graph_path)=='d170b450f5dfa1c89209f7732d1ffd0277e8ebf4cbbdeb8280fac4f43b146cd0'
rows,graphs=read(proposal_path),read(graph_path)
source={r['witness_key']:r for r in read(source_path)}
assert len(rows)==len(source)==98
assert [r['source_unit_key'] for r in rows]==list(source)
languages={r['ID'] for r in csv.DictReader((ROOT/'data/cldf/languages.csv').open())}
suffix='lsi9.2:p51:'
changes={
 suffix+'Panjabi-comparison-whole-copulas:w1':dict(remove=['pres'],add=[],notes=['Grierson says gō is sometimes added to the present copula, comparing Punjabi gā.'],reason='Printed present-copula attachment is context, not a stated intrinsic tense function of the added syllable.'),
 suffix+'Panjabi-comparison-whole-copulas:w2':dict(remove=['pres'],add=[],notes=['Grierson compares the syllable gā of Punjabi with gō added to the Ahirwati present copula.'],reason='Comparison of the added syllables does not independently classify gā as a present-tense exponent.'),
 suffix+'locative-genitive-whole-example:w6':dict(remove=[],add=['m'],notes=['The same printed ghari is expressly illustrated as a masculine consonant-final locative noun later in this paragraph.'],reason='Exact same-form source-class corroboration, not natural gender inferred from English house.'),
 suffix+'genitive-locative-dative-use:w1':dict(remove=[],add=['1sg','sg'],notes=['The source glosses the genitive-locative form mērai as “to me”, illustrating its dative use.'],reason='Explicit lexical first-person singular meaning; retains reported genitive/locative/dative function without inferring an affix or third-person agreement.'),
 suffix+'relative-when-example:w1':dict(remove=[],add=['demonstrative'],notes=[],reason='Author expressly describes this relative-pronoun example as employed in the demonstrative sense, then as well as when. No independently inferred adverb POS.')}
decisions,overlays=[],[]
for r in rows:
    key=r['source_unit_key'];s=source[key]
    assert (r['exact_form_guard'],r['exact_gloss_guard'],r['exact_language_label_guard'],r['exact_disposition_guard'])==(s['source_form'],s['source_gloss'],s['language_label'],s['disposition'])
    assert r['physical_parent_key']==s['parent_key']
    assert set(r['proposed_tags'])<=GENDER_TAGS|GRAMMATICAL_TAGS
    if r['disposition'].startswith('eligible'):
        assert r['proposed_language_id'] in languages
    else:
        assert r['proposed_language_id'] is None and r['proposed_tags']==[]
    d=dict(source_unit_key=key,exact_owner_proposal=r,literal_guard_pass=True,
           evidence_page=s['printed_page'],source_quote_is_context_summary=True,
           result='source-scoped metadata agrees',proposed_changes=None)
    if key in changes:
        c=changes[key]
        tags=[t for t in r['proposed_tags'] if t not in c['remove']]+[t for t in c['add'] if t not in r['proposed_tags']]
        notes=list(dict.fromkeys(r['proposed_notes']+c['notes']))
        d.update(result='root-adjudication requested for keyed tag-scope refinement',proposed_changes=dict(old_tags=r['proposed_tags'],new_tags=tags,old_notes=r['proposed_notes'],new_notes=notes,reason=c['reason']))
        overlays.append(dict(source_unit_key=key,exact_form_guard=s['source_form'],exact_gloss_guard=s['source_gloss'],**d['proposed_changes']))
    if ':Bawal-Kot-Kasam-regional-copulas:' in key:
        assert not set(r['proposed_tags'])&{'m','f','1sg','2sg','3sg','sg','pl'}
        d['route_condition']='Jointly attributed Bawal/Kot Kasam historical scope only; no Gurgaon tag, no invented feminine/masculine parenthetical pairing or coordinates. Base stays existing Ahirwadi.'
    if r['disposition']=='unattested_source_hypothesis':
        d['exclusion_reason']='Author explicitly says no examples encountered; held in complete audit, no lexical node or relation.'
    decisions.append(d)
eligible={r['source_unit_key']:r for r in rows if r['disposition'].startswith('eligible')}
graph_decisions=[]
for g in graphs:
    assert g['child_key'] in eligible
    assert eligible[g['child_key']]['exact_form_guard']==g['exact_child_form_guard']
    assert len(g['parent_keys'])==len(g['exact_parent_form_guards'])
    for key,form in zip(g['parent_keys'],g['exact_parent_form_guards']):
        assert key in eligible and eligible[key]['exact_form_guard']==form
    assert g['child_key'] not in g['parent_keys']
    assert g['relation'] in {'variant','component'}
    reason='Explicit alternatives in the same printed functional slot. Child points to the first printed alternative; qualified realization, not unrestricted interchangeability.'
    if g['relation']=='component':
        assert g['exact_child_form_guard']=='sai-gō' and g['exact_parent_form_guards']==['sai','gō']
        reason='Author explicitly says gō is added to the present and prints sai-gō. Preserve ordered sai+gō components. The printed “he is” context selects the third-singular paradigm witness for literal sai, with no new person tag projected onto the whole formula. Punjabi comparison is not a donor or local component.'
    graph_decisions.append(dict(exact_owner_claim=g,review='source-supported qualified relation; root final approval pending',reason=reason))
assert collections.Counter(g['relation'] for g in graphs)=={'variant':12,'component':1}
for name,records in [('all98-keyed-decisions.jsonl',decisions),('keyed-tag-scope-proposals.jsonl',overlays),('all13-graph-decisions.jsonl',graph_decisions)]:
    (HERE/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
manifest_path=PACKAGE/'scope/native-grammar-image-manifest.json'
images=json.loads(manifest_path.read_text())['pages']
for i in images: assert sha(ROOT/i['lossless_png_path'])==i['lossless_png_sha256']
report=dict(status='complete independent original-backed98 metadata/13 graph review; five exact tag refinements proposed, final approval pending',
            counts=dict(witnesses=98,eligible=sum(r['disposition'].startswith('eligible') for r in rows),documentary=sum(r['disposition']=='documentary' for r in rows),unattested_hypotheses=sum(r['disposition']=='unattested_source_hypothesis' for r in rows),literal_guard_failures=0,unknown_tags=0,unregistered_base_routes=0,tag_scope_proposals=len(overlays),variants=12,ordered_component_sets=1,ordered_component_edges=2,regional_copulas=4),
            inputs={str(p.relative_to(ROOT)):sha(p) for p in [proposal_path,graph_path,source_path,manifest_path]},
            original_images=[dict(path=i['lossless_png_path'],sha256=i['lossless_png_sha256'],printed_page=i['printed_page_expected']) for i in images],
            output_hashes={p.name:sha(p) for p in [HERE/'all98-keyed-decisions.jsonl',HERE/'keyed-tag-scope-proposals.jsonl',HERE/'all13-graph-decisions.jsonl']},
            boundaries=['All98 physical/shared interpretations conserved; quoted context is a summary, not certified verbatim prose.',
                        'Four unencountered oblique hypotheses stay audit-only; their separately printed base pronouns remain eligible.',
                        'Bagri second-plural borrowing claim supports loanword/source Etymology; no unique donor endpoint invented.',
                        'No inferred gender pairing of regional sō(sī,sā), no hypothetical base node, no cross-language similarity/borrower graph.',
                        'The reviewer authored initial grammar literals; Asur independently checked all95 initial interpretations and root accepted98 derivative. This review is independently against root-authored metadata/graph, not represented as independent initial transcription.'],
            canonical_or_owner_changes=False)
(HERE/'independent-full-grammar-metadata-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report['counts']))
