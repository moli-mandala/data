"""Close every residual scope against exact reviewed source occurrences."""
from pathlib import Path
import hashlib,json,collections,ast,sys
BASE=Path(__file__).resolve().parent;P=BASE.parent;ROOT=P.parents[5]
OUT=BASE/'residual-revision3' if '--revision3' in sys.argv else BASE/'residual-revision2' if '--revision2' in sys.argv else BASE
OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
mp=P/'integration/specimen-metadata-proposal.jsonl'
prior=BASE/'keyed-tag-recommendations.jsonl'
assert sha(mp)=='61cba0010f2ea47878a4278b4261586eaee363c01e66288f1fc694e4620f78ca'
source=rows(mp); first={r['source_unit_key']:r for r in rows(prior)}
def clean(g):return g.strip(' .;,?!‘’“”"').lower()
lexical={'so-many':['quantifier'],'all':['quantifier'],'some':['quantifier','indef'],
 'far':['adv'],'much':['quantifier'],'very':['adv','degree'],'because':['conj'],'and':['conj'],
 'with':['postp'],'before':['postp']}
complete=[];residual=[];merged=[]
for r in source:
    key=r['source_unit_key'];g=clean(r['exact_raw_gloss_guard'])
    old=list(r['proposed_tags']);after138=list(first[key]['new']) if key in first else list(old)
    new=list(after138);reasons=[];holds=[];claims=[]
    if 'obj' in new:
        new.remove('obj');reasons.append('English me/him/them/thee is an English oblique pronoun form, not an explicit target object-function label; retain person and independently source-supported cases.')
    if g.endswith('-for') or g.endswith('-near'):
        case='dat' if g.endswith('-for') else 'loc'
        if case in new:new.remove(case)
        if 'uncertain' not in new:new.append('uncertain')
        holds.append({'type':'source-alignment-case-scope','detail':'The printed English for/near meaning is secure. No explicit complete-word target case analysis is given for this expression; do not equate the English relational meaning with an intrinsic dative/locative inflection.'})
        reasons.append('Preserve exact for/near translation and lexical head, but withhold case inferred solely from English preposition.')
    for t in lexical.get(g,[]):
        if t not in new:new.append(t)
    if g in lexical:reasons.append('Original narrative scope supports this conservative lexical function; no morphological segmentation or grammatical feature inferred from English word endings.')
    if g=='dancing':
        if 'verb' in new:new.remove('verb')
        if 'noun' not in new:new.append('noun')
        claims.append({'source_page':563,'source_claim':'The source explicitly calls endākād a verbal noun.'})
        reasons.append('Exact specimen form endākād is independently identified as a verbal noun in chapter prose, not merely an English -ing gloss.')
    if g=='to-say-proper':
        for t in ['noun','uncertain']:
            if t not in new:new.append(t)
        claims.append({'source_page':563,'source_claim':'The source says the base seems to be used as a verbal noun and gives enet-lāṅg, saying-of-worthy.'})
        holds.append({'type':'qualified-source-analysis','detail':'The author qualifies this verbal-noun analysis with seems. No infinitive or participial subtype is imposed on the complete enet-lāṅg construction.'})
        reasons.append('Preserve the qualified source verbal-noun claim for the exact printed complete expression.')
    if g=='dead-man':
        if 'uncertain' not in new:new.append('uncertain')
        holds.append({'type':'source-analysis-scope','detail':'The specimen prints dead-man, while chapter p566 lists tiktānden under past-copula context as he was dead. Retain both exact source analyses; no confident noun-versus-finite predicate POS or new stem segmentation is imposed here.'})
        reasons.append('Source analytical scopes differ for this same literal form; preserve documentary cross-reference and typed uncertainty instead of harmonizing the specimen gloss.')
    if r['exact_form_guard'].strip(' .,;?!‘’“”"').lower()=='mākkun' and g=='neck-on':
        new=[t for t in new if t!='loc']
        if 'dat' not in new:new.append('dat')
        claims.append({'source_page':562,'source_claim':'The source explicitly prints mākkun, to the neck, under accusative/dative suffix examples.'})
        reasons.append('Explicit chapter case example takes precedence over English on; literal specimen gloss neck-on is unchanged.')
    # Same-language source-marked constructions, not a general -ing parser.
    kolami=(':p564:' in key or ':p565:' in key)
    bare=r['exact_form_guard'].strip(' .,;?!‘’“”"').lower()
    verbal_nā={'pāyāknā','ghumkāknā','setnā','avaletnā','sūltnā','ōltnā','tūltnā','kōtnā','tintnā','vātnā'}
    basim_conj={'sāhādūn','saddan','innekādūn','inne-kādūn'}
    if (kolami and bare in verbal_nā) or (not kolami and bare in basim_conj):
        for t in ['verb','participle','conjunctive-participle']:
            if t not in new:new.append(t)
        claims.append({'source_page':563 if kolami else 567,
          'source_claim':'Chapter defines the conjunctive-participle class and its printed realization; specimen form occurs as a verbal adjunct. No unprinted root or variant is generated.'})
        reasons.append('Scoped chapter formation statement plus the actual nonfinite specimen use supports conjunctive-participle analysis; not English -ing alone.')
    if bare=='amnet' and g=='my':
        if 'uncertain' not in new:new.append('uncertain')
        holds.append({'type':'source-person-analysis-variation','detail':'The exact specimen gloss differs from the separately printed personal-pronoun paradigm analysis for this same spelling. Preserve the occurrence-specific English interpretation without overriding it from the paradigm.'})
        reasons.append('Record explicit between-occurrence person/possessive analysis variation, without creating new homonyms or changing literal glosses.')
    if bare=='avande' and g=='he':
        if 'uncertain' not in new:new.append('uncertain')
        holds.append({'type':'source-person-analysis-variation','detail':'The specimen prints avande, he, and the independently reviewed p566 personal-pronouns13w1 paradigm prints avande, his. This is an exact short-e spelling contrast in source analyses; preserve each occurrence rather than harmonize the personal/possessive interpretation.'})
        reasons.append('Exact grammar child confirms short-e avande. Earlier root long-ē assertion conflated a different specimen p568l3w5 correction; preserve that diagnostic history, not its inaccurate related-form note.')
    if bare=='amd' and g=='they':
        if 'uncertain' not in new:new.append('uncertain')
        holds.append({'type':'source-number-analysis-variation','detail':'Specimen gloss is they, while the source pronoun paradigm prints amd, he. Preserve the explicitly plural occurrence gloss without harmonizing it to singular.'})
        reasons.append('Retain printed occurrence-specific number and mark its differing source paradigm scope.')
    scholarly_notes=[]
    if any(h['type']=='source-alignment-case-scope' for h in holds):scholarly_notes.append(f"The interlinear source translates {r['exact_form_guard']} as {r['exact_raw_gloss_guard']}, without a separate case analysis of this complete expression.")
    if bare=='amnet' and g=='my':scholarly_notes.append('The specimen glosses amnet as my, while the separately printed paradigm gives his.')
    if bare=='avande' and g=='he':scholarly_notes.append('The specimen glosses avande as he; the separately printed paradigm glosses the same avande as his.')
    if bare=='amd' and g=='they':scholarly_notes.append('The specimen glosses amd as they; the separately printed paradigm glosses amd as he.')
    if g=='dead-man':scholarly_notes.append('The specimen glosses tiktānden as dead-man; the grammar lists it under past copula as he was dead.')
    if g=='to-say-proper':scholarly_notes.append('Grierson says that the base seems to be used as a verbal noun, illustrating enet-lāṅg, saying-of-worthy.')
    if bare=='mākkun' and g=='neck-on':scholarly_notes.append('The grammar gives mākkun, to the neck, among accusative/dative examples; the specimen gloss is neck-on.')
    if new!=after138 or holds or claims:
        residual.append({'source_unit_key':key,'field':'proposed_tags','old':after138,'new':new,
          'baseline_tags':old,'exact_form_guard':r['exact_form_guard'],'exact_raw_gloss_guard':r['exact_raw_gloss_guard'],
          'reason':' '.join(reasons),'typed_metadata_uncertainties':holds,'scholarly_source_claims':claims,'scholarly_notes':scholarly_notes})
    final={'source_unit_key':key,'exact_form_guard':r['exact_form_guard'],'exact_raw_gloss_guard':r['exact_raw_gloss_guard'],
      'proposed_tags':new,'typed_metadata_uncertainties':holds,'scholarly_source_claims':claims,'scholarly_notes':scholarly_notes,
      'status':'all_scope_decisions_closed_with_typed_source_evidence' if holds else 'reviewed_source_scope_supported_or_deliberately_unforced'}
    complete.append(final)
    if new!=old:
        merged.append({'source_unit_key':key,'field':'proposed_tags','old':old,'new':new,
          'exact_form_guard':r['exact_form_guard'],'exact_raw_gloss_guard':r['exact_raw_gloss_guard'],
          'prior_138_recommendation':first.get(key),'residual_decisions':residual[-1] if residual and residual[-1]['source_unit_key']==key else None})
for name,rr in [('residual-keyed-decisions.jsonl',residual),('all-560-final-metadata-proposal.jsonl',complete),('combined-keyed-tag-recommendations.jsonl',merged)]:
    (OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr))
assert len({r['source_unit_key'] for r in complete})==560
assert {r['source_unit_key'] for r in complete}=={r['source_unit_key'] for r in source}
registered=set()
for node in ast.walk(ast.parse((ROOT/'data/tags.py').read_text())):
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'GRAMMATICAL_TAGS','GENDER_TAGS'} for t in node.targets):
        registered.update(ast.literal_eval(node.value))
assert all(t in registered or t.startswith('dialect:') for r in complete for t in r['proposed_tags'])
assert all(len(r['proposed_tags'])==len(set(r['proposed_tags'])) for r in complete)
report={'reviewer':'/root/asur_finish','status':'complete_residual_source_scope_proposal','baseline_input_sha256':sha(mp),'prior_138_ledger_sha256':sha(prior),
 'counts':{'complete_atomic_scope_decisions':560,'residual_keyed_decisions':len(residual),'combined_changed_tag_rows':len(merged),'typed_scope_hold_rows':sum(bool(r['typed_metadata_uncertainties']) for r in complete),'explicit_object_function_withheld_rows':sum('obj' in r['proposed_tags'] for r in source),'case_from_for_near_withheld_rows':sum(any(h['type']=='source-alignment-case-scope' for h in r['typed_metadata_uncertainties']) for r in complete)},
 'source_cross_references':['Original p562 accusative/dative mākkun and case paragraphs; original p563 verbal nouns, conjunctive-participle formation and imperative; original p566 past copula/pronoun paradigms; original p567 conjunctive examples; all four narrative specimen pages.'],
 'policy':['All ordinary residual scopes are now closed by conservative lexical function or explicit typed hold, not pending owner extraction.','No generic English -ing subtype, guessed root, fabricated English paraphrase, new word alignment or inferred graph relation.','Complete for/near expressions lack an explicit intrinsic case analysis; exact meanings remain, case Tags withheld.','Printed to/from/in/of analyses corroborated by chapter usage remain; explicit chapter mākkun dat corrects the on-based loc guess.','Qualifiers/source-analysis variations remain audit data; actual scholarly grammar claims may be carried into appropriate source Notes.'],
 'outputs':{(OUT/n).name:sha(OUT/n) for n in ['residual-keyed-decisions.jsonl','all-560-final-metadata-proposal.jsonl','combined-keyed-tag-recommendations.jsonl']},
 'canonical_changes':False,'owner_input_changes':False,'installation_approval':False}
report['revision2_scope_refinement']={'tiriknā_again':'Both p565 l4w1/l18w3 are iterative adverbs modifying life-came/to-life-came; -nā resemblance alone does not establish verbal adjunct status. Retain adv only.',
 'avaletnā':'Coming-to-senses is a source-context verbal adjunct; include verb with participle/conjunctive-participle.'}
if '--revision3' in sys.argv:
    before=rows(BASE/'residual-revision2/all-560-final-metadata-proposal.jsonl')
    bm={r['source_unit_key']:r for r in before};changed=[r['source_unit_key'] for r in complete if r!=bm[r['source_unit_key']]]
    assert changed==['grierson1906kolamibasim:p568:specimen76-line:6:word:7'],changed
    assert all(r['proposed_tags']==bm[r['source_unit_key']]['proposed_tags'] for r in complete)
    report['revision3_exact_continuity']={'changed_key':changed,'tags_forms_glosses_keys_unchanged':True,
      'root_diagnostic':'Root rechecked p566 personal-pronouns13w1 avande/his: earlier avandē assertion conflated specimen p568l3w5 quantity correction. Revision2 remains history; only one scope-detail/source-note record corrected.'}
(OUT/'residual-complete-scope-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'counts':report['counts'],'report_sha256':sha(OUT/'residual-complete-scope-review.json')}))
