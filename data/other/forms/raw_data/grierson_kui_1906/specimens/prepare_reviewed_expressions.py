"""Guarded joins and exhaustive, keyed expression-type review; no installation."""
import collections
import copy
import json
from reconcile_specimen40_review import HERE, read, write, sha

# Manually reviewed finite/imperative or predicative evidence in every aligned
# span. These are exact source gloss substrings, not an English suffix classifier.
EVIDENCE = {
40: ['stealing-became.', 'eating-was.', 'gathering-came.', 'what doest my tree-in?',
     'saying said.', 'will-you-do?', 'I will-not-come;', 'much beat.'],
41: ['two sons were.', 'that give.', 'share-making gave.', 'going was.',
     'spent-made.', 'distress-in fell.', 'staying-became.', 'field-to sent.',
     'I hunger-from dying-am.', 'sin having-done am.', 'I worthy am-not.',
     'one servant make.', 'father near went.', 'embraced and kissed.',
     'sin having-done am.', 'I worthy am-not.', 'shoes feet-on give.',
     'merry shall-make;', 'beginning-became.', 'field-in was.', 'singing heard.',
     'wherefore becoming-is?', 'feast giving-is.', 'wishing-not became.',
     'entreating made.', 'transgressing-not-was.', 'giving wast-not.',
     'feast gavest.', 'thou always me-with are.', 'that all thine.', 'proper-is.'],
42: ['Buffaloes eating-were.', 'presence-to went.', 'pushing drove-off.',
     'seize could-not.', 'pleased shall-be.', 'kept are.', 'way-on fled.',
     'why going-art?', 'intention clearly caught-is.', 'saying fled.'],
43: ['two sons were.', 'that give.', 'them gave.', 'goods there squandered.',
     'distress suffered.', 'field-in sent.', 'gave-not.', 'mind made.',
     'we hunger-in dying-are.', 'sin did.', 'me keep.', 'father near went.',
     'catching kissed.', 'am-not.', 'shoes give.', 'merry will-be;',
     'to-make made.', 'field-in was.', 'house-to came.', 'why thus doing-are?',
     'feast giving-is.', 'to-go wished-not.', 'much said.', 'work doing-I-am.',
     'giving-wast-not.', 'feast gavest.', 'that all thine.', 'we doing-are.'],
}

def main():
    source=HERE/'expression-spans-reviewed.jsonl'
    atoms_path=HERE/'interlinear-reviewed.jsonl'
    assert sha(source)=='a40817f15279ed8be6bf3c78841d4d7986bdc3caf36ca51d734cb2d3722d9bef'
    assert sha(atoms_path)=='d6960e5336320d539b83be14010b57fdf4b0183640c21e2936d64649843aee5d'
    rows=read(source.name)
    atoms=read(atoms_path.name)
    keyed={r['source_unit_key']:r for r in atoms}
    join_path=HERE/'explicit-continuations-first-reading.json'
    joins=json.loads(join_path.read_text())
    assert len(joins)==2
    decisions=[]
    applied=[]
    for r in rows:
        n=int(r['source_unit_key'].rsplit('expression',1)[1])
        cue=EVIDENCE[r['specimen']][n-1]
        assert cue in r['raw_interlinear_gloss'],(r['source_unit_key'],cue)
        r['emission_form']=r['raw_source_form']
        r['emission_gloss']=r['raw_interlinear_gloss']
        r['explicit_word_joins']=[]
        for join in joins:
            if join['atom_keys'][0] not in r['atom_keys']:
                continue
            index=r['atom_keys'].index(join['atom_keys'][0])
            assert r['atom_keys'][index:index+2]==join['atom_keys']
            parts=[keyed[k]['raw_source_form'] for k in join['atom_keys']]
            assert parts==join['raw_form_parts']
            assert ''.join(parts)==join['combined_form']
            old=' '.join(parts)
            assert r['emission_form'].count(old)==1
            r['emission_form']=r['emission_form'].replace(old,join['combined_form'],1)
            old_gloss=' '.join(keyed[k]['raw_gloss'] for k in join['atom_keys'])
            assert r['emission_gloss'].count(old_gloss)==1
            r['emission_gloss']=r['emission_gloss'].replace(old_gloss,join['combined_gloss'],1)
            r['explicit_word_joins'].append(copy.deepcopy(join))
            applied.append(dict(expression_key=r['source_unit_key'],atom_keys=join['atom_keys'],
                                old_form=old,new_form=join['combined_form'],
                                old_gloss=old_gloss,new_gloss=join['combined_gloss']))
        # All77 were individually read as complete predicates/clauses or joined
        # narrative sentences. No title, isolated vocative, or answer fragment is
        # included in this span inventory. Embedded speech is not a new boundary.
        r['expression_type']='sentential'
        r['tags']=['sentential','multiword-expression']
        assert ' ' in r['emission_form']
        r['typed_uncertainties']=[dict(atom_key=k,evidence=keyed[k]['typography_uncertainty'])
                                  for k in r['atom_keys'] if keyed[k]['typography_uncertainty']]
        if r['typed_uncertainties']:
            r['tags'].append('uncertain')
        r['review_status']='all77 source-aligned spans individually classified; approved physical word joins applied'
        r['first_locator']=r['atom_keys'][0]
        r['last_locator']=r['atom_keys'][-1]
        reason='Aligned source supplies a complete finite predicate or coordinated predicates; quoted speech and dependent clauses stay in their attested contiguous span.'
        if (r['specimen'],n)==(41,12):
            reason='Complete quoted imperative make me thy servant; understood addressee does not make this a fragment.'
        if (r['specimen'],n)==(41,30):
            reason='Possessive predication: mine what is, that all thine. The source supplies a complete relative-plus-predicative statement despite an unexpressed second English copula.'
        if (r['specimen'],n)==(40,8):
            reason='Long coordinated narrative and quoted dialogue contain complete predicates through beat; internal ?- remains inside its printed atom. Preserve the source-contiguous multi-clause span, not a lexical lemma or inferred smaller segmentation.'
        decisions.append(dict(source_unit_key=r['source_unit_key'],classification='sentential',
                              exact_aligned_evidence=cue,reason=reason,tags=r['tags'],
                              printed_pages=r['printed_pages'],first_atom=r['first_locator'],last_atom=r['last_locator']))
    assert len(rows)==len(decisions)==sum(map(len,EVIDENCE.values()))==77
    assert len(applied)==2
    original=read(source.name)
    assert [r['source_unit_key'] for r in rows]==[r['source_unit_key'] for r in original]
    assert all(a['raw_source_form']==b['raw_source_form'] and a['raw_interlinear_gloss']==b['raw_interlinear_gloss']
               and a['atom_keys']==b['atom_keys'] for a,b in zip(rows,original))
    covered=[k for r in rows for k in r['atom_keys']]
    assert collections.Counter(covered)==collections.Counter(r['source_unit_key'] for r in atoms if r['unit_type']!='title_atom')
    write('expression-emissions-reviewed.jsonl',rows)
    report=dict(status='77 complete aligned expressions reviewed; no phrase/fragment among these spans; six title atoms separate',
        inputs={p.name:sha(p) for p in [source,atoms_path,join_path]},
        output_sha256=sha(HERE/'expression-emissions-reviewed.jsonl'),decisions=decisions,joins=applied,
        counts=dict(sentential=77,phrase=0,fragment=0,physical_atoms_preserved=884,title_atoms_separate=6,
                    uncertain_expressions=sum('uncertain' in r['tags'] for r in rows)),
        original_policy='Emit emission_form as source Original after removing only the two explicitly printed physical line-wrap spaces. raw_source_form/raw_interlinear_gloss and both physical keys preserve exact layout evidence; all punctuation remains literal.',
        display_policy='Unmodified generic parser strips boundary characters1234⁴5⁵67⁷,;. before profile. No change proposed to parser. Quotes/question marks/internal punctuation remain. Actual parser previously verified exact Original retention; rerun after integrated emission.',
        limitations=['This classifies complete source-aligned spans, not internal morphemes or every English tense label.','Fresh integrated output audit and final source metadata/graph/profile gates remain.'],canonical_changes=False)
    (HERE/'expression-type-and-join-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(counts=report['counts'],output_sha256=report['output_sha256']),indent=2))

if __name__=='__main__':
    main()
