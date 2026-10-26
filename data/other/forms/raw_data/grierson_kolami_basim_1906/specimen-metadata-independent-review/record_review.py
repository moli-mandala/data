"""Independent source-scoped recommendations, not owner/canonical edits."""
from pathlib import Path
import collections, hashlib, json

OUT = Path(__file__).resolve().parent
P = OUT.parent
ROOT = P.parents[5]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p): return [json.loads(x) for x in p.read_text().splitlines()]
MP = P/'integration/specimen-metadata-proposal.jsonl'
EP = P/'specimens/sharp-reviewed-revision/expression-emission-proposal.jsonl'
CP = P/'integration/combined-review-candidate.csv'
assert sha(MP)=='61cba0010f2ea47878a4278b4261586eaee363c01e66288f1fc694e4620f78ca'
assert sha(EP)=='fecc6462f86e2713bbf78e9fb8deab3061cd597cf34681d5bc86e31e6c42c43c'
assert sha(CP)=='d70237705abcb7bcdb59a32f1898ebb0007ad9d83cbedcdab85d2473f6fa0509'
rr=rows(MP); ee=rows(EP)
assert len(rr)==560 and len(ee)==51
def clean(g): return g.strip(' .;,?!‘’“”"').lower()
# Exact printed gloss classes individually assessed against the four originals.
# These are review proposals, not a general English/prose parser.
add = {
 'to-feed':['verb','inf'], 'to-say':['verb','inf'], 'to-entreat':['verb','inf'],
 'to-spend':['verb','inf'], 'to-eat':['verb','inf'],
 'field-to':['noun'], 'field-in':['noun'], 'cloth':['noun'], 'command':['noun'],
 'harlots':['noun','pl'], 'harlots(-to)':['noun','pl'],
 'distance-from':['noun'], 'safety-in':['noun'], 'in-presence':['noun','loc'],
 'in-days':['noun','pl','loc'], 'servants-in-from':['noun','pl'],
 'with-hunger':['noun'], 'father-':['noun','kinship'],
 'did':['verb','pret'], 'gathered':['verb','pret'],
 'comes':['verb','pres'], 'it-appeared':['verb','pret'],
 'i-broke':['verb','pret','1sg','first-person'], 'i-did':['verb','pret','1sg','first-person'],
 'i-do':['verb','pres','1sg','first-person'],
 'he-went':['verb','pret','3sg','third-person'],
 'he-made':['verb','pret','3sg','third-person'], 'madest':['verb','pret','2sg','second-person'],
 'life-came':['verb','pret'], 'to-life-came':['verb','pret'],
 'service-lived':['verb','pret'], 'destroyed-made':['verb','pret'],
 'eat-made':['verb','pret'], 'was-found':['verb','pret'],
 'lost-was':['verb','pret'], 'dead-was':['verb','pret'],
 'had-died':['verb','pret'], 'came-then':['verb','pret'],
 'come-is':['verb'], 'is-got':['verb','pres'], 'was-not-got':['verb','pret','neg'],
 'is-this':['verb','copula','pres'], 'is-proper':['verb','copula','pres'],
 'making-art':['verb','pres'],
 'gave-not':['pret'], 'went-not':['pret'], 'broke-not':['pret'],
 'gavest-not':['pret','2sg','second-person'], 'came-not':['pret'],
 'am-not':['copula','pres','1sg','first-person'],
 'thine-only':['pron','poss','2sg','second-person','emph'],
 'those-from':['pron','demonstrative','pl'],
 'any-one':['pron','indef'], 'anybody':['pron','indef'], 'anything':['pron','indef'],
 'good':['adj'], 'big':['adj'], 'worthy':['adj'], 'proper':['adj'], 'merry':['adj'],
 'the-younger':['adj'], 'he-younger':['adj'], 'other':['adj'],
 'then':['adv'], 'there':['adv'], 'riotously':['adv'], 'thus':['adv'],
 'henceforward':['adv'], 'again':['adv'], 'inside':['adv'], 'outside':['adv'],
 'ever':['adv'], 'always':['adv'], 'quickly':['adv'], 'well':['adv'],
}
changes=[]; reviews=[]
for r in rr:
    g=clean(r['exact_raw_gloss_guard']); old=r['proposed_tags']; new=list(old); reasons=[]
    for t in add.get(g,[]):
        if t not in new:new.append(t)
    if new!=old:reasons.append('Exact printed aligned lexical/function gloss and its original narrative scope support these tags; no target morpheme segmentation or unprinted base is supplied.')
    if g in {'put-on','put'}:
        new=[t for t in new if t not in {'pret','loc'}]
        for t in ('verb','impv'):
            if t not in new:new.append(t)
        reasons.append('Original quoted instructions to the servants are directives. English put is tense-ambiguous and on in put-on is a phrasal particle, not a target locative claim.')
    if g=='dancing':
        new=[t for t in new if t!='verb']
        reasons.append('Original music / dancing / heard places this aligned unit in an event-object list. English -ing alone does not establish target verb POS; hold subtype rather than infer it.')
    if new!=old:
        changes.append({'source_unit_key':r['source_unit_key'],'field':'proposed_tags','old':old,'new':new,
            'exact_form_guard':r['exact_form_guard'],'exact_raw_gloss_guard':r['exact_raw_gloss_guard'],
            'reason':' '.join(reasons),'printed_page':int(r['source_unit_key'].split(':p')[1].split(':')[0])})
    reviews.append({'source_unit_key':r['source_unit_key'],'exact_form_guard':r['exact_form_guard'],
        'exact_raw_gloss_guard':r['exact_raw_gloss_guard'],'status':'keyed_metadata_recommendation' if new!=old else 'current_metadata_supported_or_explicitly_unforced',
        'proposed_tags':new,'literal_change':False})
assert len({r['source_unit_key'] for r in reviews})==560
OUT.mkdir(exist_ok=True)
for name,record in [('all-560-keyed-review.jsonl',reviews),('keyed-tag-recommendations.jsonl',changes)]:
    (OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in record))
expression_reviews=[]
for e in ee:
    assert e['source_unit_key']==e['physical_source_keys'][0]+':expression'
    expression_reviews.append({'source_unit_key':e['source_unit_key'],'physical_source_keys':e['physical_source_keys'],
        'source_gloss_guard':e['raw_aligned_interlinear_gloss'],'source_form_guard':e['raw_source_form'],
        'sentential_scope':'supported: complete narrative clause or source-bounded quoted clausal construction, including nominal predicates; never assigned from physical line count',
        'decision':'retain sentential; do not apply additional whole-expression tense/person tags from internal clauses'})
(OUT/'all-51-expression-scope-review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in expression_reviews))
images=ROOT/'tmp/jambu-next-source/kolami-specimen-originals'
evidence=[images/f'owner-sharp-p{p}-body{n}.png' for p in (564,565,568,569) for n in range(4)]
report={'reviewer':'/root/asur_finish','status':'complete_structured_metadata_review_repair_required',
 'scope':'All 560 atomic emissions, all 291 distinct exact raw gloss/current-tag classes, all 51 source-bounded expressions, every original specimen page 564/565/568/569 and retained body crops.',
 'input_pins':{str(p.relative_to(P)):sha(p) for p in (MP,EP,CP,P/'prepare_combined_candidate.py')},
 'source_evidence':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in evidence],
 'counts':{'atomic_rows_reviewed':560,'distinct_raw_gloss_current_tag_classes':291,'expressions_reviewed':51,'keyed_tag_recommendations':len(changes)},
 'recommendation_files':{p.name:sha(p) for p in [OUT/'all-560-keyed-review.jsonl',OUT/'keyed-tag-recommendations.jsonl',OUT/'all-51-expression-scope-review.jsonl']},
 'class_findings':['Missing source-glossed lexical noun/verb POS, including finite multiword English glosses aligned to one source token.',
   'Inconsistent infinitive handling: to-make parsed, but five other explicitly to-verb gloss classes unparsed.',
   'False locative and past from English put-on/put; source quotation is directive.',
   'Clear adjective/adverb/indefinite-pronoun meanings left without POS; proposed only exact reviewed classes, not generic English suffix scans.',
   'Dancing in the music/dancing/heard object list cannot receive confident verb solely from English -ing.',
   'Literal source fields, source keys, boundaries, dialects, uncertainty and approved source history remain unchanged by these recommendations.'],
 'remaining_scope_adjudications':[
   {'class':'obj from English me/him/them/thee', 'decision':'Review target grammatical role before accepting blanket obj: English pronoun case alone does not prove target object function, particularly Them-of and existential Him-to two sons were. Keep source-stated gen/dat and person; do not invent direct/indirect object roles.', 'automatic_change':False},
   {'class':'English prepositions/case tags', 'decision':'Where printed English -to/-from/-in/-of matches explicit chapter morphology, current dat/abl/loc/gen may be retained as source analysis; do not extend CASE suffix matching to English phrasal verbs or for/near based on parser alone.', 'automatic_change':False},
   {'class':'to-say-proper / dead-man / so-many / all / some / far / much / very / because / and', 'decision':'Preserve exact gloss and source context; these residual combined, quantificational or syntactic scopes need an explicit owner decision rather than blanket morphology or inferred token splitting.', 'automatic_change':False},
   {'class':'English -ing / having-', 'decision':'Verb function can be recorded for clearly verbal adjuncts, but no participle/gerund/perfect subtype from English alone. Existing having-gone/saying is conservatively verb only; no automatic -ing subtype.', 'automatic_change':False}],
 'prior_conservation_evidence':'specimen-conservation-review/sharp-revised-conservation-review.json independently validates all 561 physical atoms, one join, 560 atomic rows and source-derived 51 boundaries against separately root-approved ledger. This report does not replace a fresh final-output sample.',
 'limits':'No canonical mutations or owner edits. This is metadata review, not a new full literal acceptance, source-stage installation approval, compiled build or browser check.'}
(OUT/'independent-full-specimen-metadata-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'counts':report['counts'],'report_sha256':sha(OUT/'independent-full-specimen-metadata-review.json')}))
