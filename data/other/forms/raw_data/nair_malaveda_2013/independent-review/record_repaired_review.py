"""Record independent original-page comparisons of the repaired freeze."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
selection_path = HERE / 'selection-repaired-20260927.json'
selection = json.loads(selection_path.read_text())
for name, digest in selection['inputs_sha256'].items():
    assert sha(PACKAGE / name) == digest, name
old_csv = PACKAGE / 'revision2-pre-independent-repairs/proposed.csv'
assert sha(old_csv) == '38574c8a5365c956852a13cfe75632cfb2d900f2546750f69184d58a2c84bee5'
old = list(csv.reader(old_csv.open(newline='')))
new = list(csv.reader((PACKAGE / 'proposed.csv').open(newline='')))
assert [r[10] for r in old] == [r[10] for r in new]
by_key = {r[10]: r for r in new}
changes = []
for before, after in zip(old, new):
    for index in range(15):
        if before[index] != after[index]:
            assert index in (11, 14), (after[10], index)
            changes.append(dict(entry_key=after[10], field='Tags' if index == 14 else 'Variant_Of_Key',
                before=before[index], after=after[index]))
continuity = json.loads((PACKAGE / 'review-repair-continuity.json').read_text())
assert changes == continuity['changes']
assert sum(c['field'] == 'Tags' for c in changes) == 8
assert sum(c['field'] == 'Variant_Of_Key' for c in changes) == 12

tag_evidence = {
    'accusative-examples:w2': 'p40 explicitly contrasts singular iññu with plural imm- before other cases. Original p41 prints imma as the accusative example; 2pl supported by that exact source base, not English you alone.',
    'genitive-examples:w1': 'Original p41 avariṭe is glossed their: 3pl and poss, with no gender assertion.',
    'genitive-examples:w2': 'Original p41 kiḷani-ite is of the sister: explicit kinship, genitive unchanged.',
    'genitive-examples:w4': 'Original p40 identifies iññu as second-person singular oblique; p41 iññ-aṭe of you uses this source base and genitive heading. 2sg and poss supported without deriving an unprinted form.',
    'genitive-examples:w5': 'Original p41 atuv-uṭe is of her: third-person singular feminine possessive.',
    'locative-examples:w1': 'Original p41 kiḷani-ilu → kilannii in sister: segmented member of explicitly glossed kinship locative.',
    'locative-examples:w2': 'Original p41 arrowed kilannii retains in sister gloss scope: kinship, with no new derivation edge.',
    'locative-examples:w7': 'Original p41 atuv-ukkam is in her: 3sg f supported, locative retained and possession not added.',
}
targeted = []
for change in changes:
    if change['field'] == 'Tags':
        key = change['entry_key'].split(':p41:', 1)[1]
        assert key in tag_evidence
        assert set(change['before'].split()) <= set(change['after'].split())
        targeted.append(dict(change, original_evidence=tag_evidence[key], independent_result='pass'))
assert len(targeted) == 8

evidence = [
    'p40 final pronoun paragraph explicitly calls ivvu neuter proximate without number; source a/proximate/distant wording remains contradictory and uncertain, not resolved by inference.',
    'p40 feminine proximate form ituvu (Sg.) gives the scoped 3sg f prox; source deixis anomaly is retained.',
    'p40 first-person singular paragraph explicitly prints oblique en- as in ennaaru. Blank lexical gloss is appropriate for the unglossed base.',
    'p40 first-person plural paragraph prints ommaaru we (Dat), after om oblique base; no inclusive/exclusive distinction invented.',
    'p41 last numeral paragraph prints aññuuṛavu five hundred; ñ and ṛ literal marks retained.',
    'p41 gender examples print tampuraan respectable person; no unprinted gender assignment added.',
    'p41 numeral paragraph prints n̲aanuuṛavu four hundred with visible drawn lowline under initial n.',
    'p41 dative forms paragraph gives –aaru and –e; emitted en-dash and suffix/dat tags agree.',
    'p42 Personal Terminations row II singular prints eeni,eni; eni retains 2sg suffix and exact alternate endpoint eeni.',
    'p42 imperative singular paragraph prints vera come (you); singular imperative source scope retained.',
    'p42 Personal Terminations III(Fem) singular cell aa, aatu supports suffix 3sg f and exact alternate endpoint aa; the separate II(Mas) typo is not silently repaired.',
    'p42 Personal Terminations row II singular prints eeni,eni; eeni base endpoint preserved.',
    'p43 interrogative suffix paragraph prints atuvu varutavee? is she coming; question boundary, interr and sentential/multiword tags agree.',
    'p43 affirmative/negative paragraph explicitly prints -alle and ille are negative forms; raw leading hyphen retained and no invented lexical gloss.',
    'p43 affirmative questions use eelaa; the Malayalam allee?/illee? comparators remain separate audit controls.',
    'p43 conditional paragraph prints aaṇṭe, with suffix/conditional source function; no English-based additional tense interpretation.',
    'p44 continued vocabulary prints kadave mouth, literal vowels preserved.',
    'p44 community-term list prints veñcaḷi Ezhava; community referent does not change the source language attribution of the term.',
    'p44 continued vocabulary prints pommayavu breast; single lexical head retained.',
    'p44 continued vocabulary prints cidava good; adjectival English definition and source term agree.',
]
findings = []
assert len(evidence) == 20
for entry, note in zip(selection['selection'], evidence):
    assert by_key[entry['witness_key']] == entry['actual_csv_row']
    findings.append(dict(sample_index=entry['sample_index'], source_unit_key=entry['source_unit_key'],
        witness_key=entry['witness_key'], actual_csv_row=entry['actual_csv_row'],
        original_page=entry['printed_page'], original_image=entry['original_image'],
        original_image_sha256=sha(PACKAGE / entry['original_image']), source_locator=entry['source_locator'],
        original_evidence=note, literal_gloss_scope='pass', metadata_graph_uncertainty='pass', material_error=False))
assert not set(selection['excluded_prior_witness_keys']) & {f['witness_key'] for f in findings}

graph = json.loads((PACKAGE / 'graph-proposal.json').read_text())
edges = []
for edge in graph['proposals']:
    child, parent = by_key[edge['child_key']], by_key[edge['variant_of_key']]
    assert child[11] == parent[10] and child[0] == parent[0] and child[10] != parent[10]
    assert child[2] == edge['child_form'] and parent[2] == edge['head_form']
    edges.append(dict(child_key=child[10], parent_key=parent[10], independent_result='source-supported and closed',
        evidence='Original p41 free-variation clauses' if edge['printed_page'] == 41 else 'Original p42 same-cell alternatives or explicit imperative suffix/exemplar alternation; source II(Mas.) ambiguity remains typed'))
assert sum(bool(r[11]) for r in new) == len(edges) == 12
for row in new:
    for index in (11, 12, 13):
        assert not row[index] or all(k in by_key for k in row[index].split('|'))

titles = json.loads((PACKAGE / 'metadata/raw-title-evidence.json').read_text())
bib = (PACKAGE / 'metadata/proposed-references.bib').read_text()
for title in titles['decisions']:
    assert title['printed_title'] in bib and title['display_title'] in bib
    assert 'not independently consulted' in bib
title_review = [dict(d, independent_result='pass', evidence='Original p44 reported bibliography visibly prints this misspelling; note distinguishes editorial display correction from independent thesis consultation') for d in titles['decisions']]
report = dict(status='pass: repaired frozen candidate independently checked against originals; no installation claim',
    seed=selection['seed'], selection_sha256=sha(selection_path), inputs_sha256=selection['inputs_sha256'],
    original_pdf_sha256=selection['source_pdf_sha256'], sample_rows=20, unique_witness_keys=20,
    unique_physical_blocks=len({f['source_unit_key'] for f in findings}), prior_twenty_witness_keys_excluded=True,
    physical_block_overlap_disclosure=selection['method'], material_errors=0, literal_gloss_scope_passes=20,
    overall_passes=20, findings=findings, targeted_tag_review=targeted, independently_computed_delta=changes,
    continuity='Exact eight Tags plus twelve Variant_Of_Key fields; all other fields and ordered223keys unchanged.',
    graph_review=dict(edges=12, closed=True, source_supported=True, findings=edges),
    raw_title_review=title_review, lowline_continuity='All forms unchanged from prior complete14-vector original review; fresh n̲aanuuṛavu original reconfirmed.',
    accepted_source_anomalies='Unresolved third-person deixis, II(Mas.) table label and conditional analysis/gloss conflict remain recorded; no silent scholarly repair.',
    canonical_changes=False, full_build_browser='not run; source-stage only')
(HERE / 'independent-original-output-review-pass2.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print('PASS: twenty disjoint witness rows, eight targeted tag scopes, twelve variant endpoints, two raw-title notes; zero material errors')
