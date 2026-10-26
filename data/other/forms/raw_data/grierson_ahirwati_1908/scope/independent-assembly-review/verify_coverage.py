"""Read-only independent census/assembly conservation; writes reviewer evidence only."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
ASSEMBLY = PACKAGE / 'integration/review-assembly-r1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p): return [json.loads(x) for x in p.read_text().splitlines()]
def strings(x):
    if isinstance(x, str): yield x
    elif isinstance(x, list):
        for y in x: yield from strings(y)
    elif isinstance(x, dict):
        for y in x.values(): yield from strings(y)
def write(name, value):
    (HERE/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
status = json.loads((ASSEMBLY/'status.json').read_text())
scope_status = json.loads((PACKAGE/'scope/root-current-census-reconciliation-r2-status.json').read_text())
for manifest in [status, scope_status]:
    for rel, expected in manifest['inputs'].items(): assert sha(PACKAGE/rel) == expected, rel
for rel, expected in status['outputs'].items(): assert sha(ASSEMBLY/rel) == expected, rel
census_path = PACKAGE/'scope/root-current-census-reconciliation-r2.jsonl'
assert sha(census_path) == scope_status['output_sha256']
census = rows(census_path)
raw_census = rows(PACKAGE/'scope/physical-source-unit-census.jsonl')
audit = rows(ASSEMBLY/'combined-review-audit.jsonl')
csvrows = list(csv.reader((ASSEMBLY/'combined-review-candidate.csv').open()))
assert len(csvrows) == 653 and len(audit) == 1961 and len(census) == len(raw_census) == 173
assert [x['physical_census_key'] for x in census] == [x['key'] for x in raw_census]
assert all(x['raw_census_record'] == y for x,y in zip(census, raw_census))
entry_lookup = {x[10]: x for x in csvrows}
assert len(entry_lookup) == 653 and all(len(x) == 15 for x in csvrows)
assert len({x['source_unit_key'] for x in audit}) == len(audit)
emissions = [x for x in audit if 'actual_row' in x]
assert len(emissions) == 653
assert Counter(x['source_unit_key'] for x in emissions) == Counter(entry_lookup.keys())
assert all(x['entry_keys'] == [x['source_unit_key']] and x['actual_row'] == entry_lookup[x['source_unit_key']] for x in emissions)
assert all(k in entry_lookup for x in audit for k in x.get('entry_keys', []))
parents = [x for x in audit if x['kind'] == 'structural-parent']
assert Counter(x['source_record']['key'] for x in parents) == Counter(x['physical_census_key'] for x in census)
physical = [x for x in audit if 'physical_key' in x]
assert len(physical) == 1059 and len({x['physical_key'] for x in physical}) == 1059
fine_physical = [k for x in census for k in x['fine_source_keys'] if ':aligned-line:' in k or ':native-line:' in k]
assert Counter(fine_physical) == Counter(x['physical_key'] for x in physical)
expected_refs = defaultdict(set)
roman_allocations = Counter()
native_allocations = Counter()
joins = []
for x in emissions:
    source = x['source_record']
    if x['kind'].startswith('No37-'):
        rk, nk = source['roman_physical_keys'], source['native_physical_keys']
        for k in rk + nk: expected_refs[k].add(x['source_unit_key'])
        if x['kind'] == 'No37-Roman-lexical-emission':
            roman_allocations.update(rk)
            native_allocations.update(nk)
            if len(rk)>1: joins.append({'section':'No37', 'entry_key':x['source_unit_key'], 'physical_keys':rk, 'form':x['actual_row'][2]})
    elif x['kind'] in ['No38-atomic','No38-expression']:
        for k in source['physical_source_keys']: expected_refs[k].add(x['source_unit_key'])
        if x['kind'] == 'No38-atomic':
            roman_allocations.update(source['physical_source_keys'])
            if len(source['physical_source_keys'])>1: joins.append({'section':'No38', 'entry_key':x['source_unit_key'], 'physical_keys':source['physical_source_keys'], 'form':x['actual_row'][2]})
assert len(joins)==5 and Counter(x['section'] for x in joins)=={'No37':4,'No38':1}
roman_physical = [x['physical_key'] for x in physical if x['kind'].endswith('Roman')]
assert Counter(roman_physical) == roman_allocations and len(roman_physical)==500
for x in physical: assert set(x['entry_keys']) == expected_refs[x['physical_key']], x['physical_key']
native37 = [x['physical_key'] for x in physical if x['kind']=='No37-physical-Native']
assert set(native37) == set(native_allocations)
shared = {k:n for k,n in native_allocations.items() if n>1}
assert shared == {'lsi9.2:p233:native-line:3:word:1':2}
shared_rows = [x for x in emissions if x['kind']=='No37-Roman-lexical-emission' and 'lsi9.2:p233:native-line:3:word:1' in x['source_record']['native_physical_keys']]
assert len(shared_rows)==2 and all(not x['actual_row'][4] for x in shared_rows)
assert len({x['source_record']['shared_native_group_key'] for x in shared_rows})==1
native38 = [x for x in physical if x['kind']=='No38-physical-Native']
assert len(native38)==194 and all(x['entry_keys']==[] for x in native38)
assert all(not x['actual_row'][4] for x in emissions if x['kind'] in ['No38-atomic','No38-expression'])
all_strings = set(strings(audit))
fine_nonphysical = [k for x in census for k in x['fine_source_keys'] if k not in set(fine_physical)]
assert all(k in all_strings for k in fine_nonphysical)
grammar = [x for x in audit if x['kind'] in ['grammar','grammar-documentary']]
assert len(grammar)==98
assert Counter(x['source_record']['disposition'] for x in grammar if x['kind']=='grammar-documentary') == {'documentary':19,'unattested_source_hypothesis':4}
section_counts = Counter(x['kind'] for x in emissions)
assert dict(section_counts)==status['counts']['sections']
assert sum(section_counts.values()) == 75+363+27+132+23+18+15 == 653
outside = [x for x in audit if x['kind'].startswith('outside-')]
assert Counter(x['kind'] for x in outside)=={'outside-eligible':15,'outside-control':47,'outside-physical-components':2}
findings=[{'type':'documentary-text-preservation', 'source_key':'lsi9.2:p233:two-specimen-script-and-lect-introduction',
           'details':'Assembly retains only broad structural census for this parent. The actual recovered introduction in scope/p233-specimen-introduction-first-reading.json is referenced by census-r2 but is not pinned by assembly status or embedded as documentary context. Preserve/link the recovered source text in the final assembly; no lexical-row omission is asserted.',
           'evidence_sha256':sha(PACKAGE/'scope/p233-specimen-introduction-first-reading.json')}]
parent_decisions = [{'parent_key':x['physical_census_key'],'disposition':x['disposition'],
                    'fine_key_count':len(x['fine_source_keys']), 'exact_structural_record_retained':True,
                    'fine_key_closure':all(k in all_strings for k in x['fine_source_keys'])} for x in census]
write('all173-parent-closure.json',parent_decisions)
write('five-printer-join-closure.json',joins)
report={'status':'physical/emission conservation passed; one documentary-context refinement required',
        'reviewer':'/root/asur_finish', 'counts':{'parents':173,'physical_keys':1059,'Roman':500,'Devanagari':365,'Persian_editorial_spacegroups':194,
            'emissions':653,'grammar_interpretations':98,'grammar_emitted':75,'grammar_documentary':19,'unattested_hypotheses':4,
            'No37_lexical':363,'No38_lexical':132,'whole_expressions':50,'quoted_occurrences':18,'outside_emissions':15,
            'outside_control':47,'outside_physical_only':2},
        'missing_or_duplicate_physical_keys':[], 'missing_or_duplicate_emission_keys':[],
        'shared_native_provenance':{'physical_key':next(iter(shared)), 'lexical_references':2,'narrow_native_fields_blank':True,'physical_record_count':1},
        'interpretation_limits':['Whole expressions deliberately overlap lexical contributors and each other; they are not additional physical occurrences.',
            'p237 quotations are separately printed occurrences, not literal reuse of p238 narrative.',
            'No external reuse disposition is asserted by assembly r1; aliases are the five printer joins and the shared cross-script group.',
            'All194 Persian groups are broad provenance only; this structural check does not certify their literals or per-word Native.',
            'Metadata proposals, documentary original review, final source/profile/parser and canonical/build/browser gates remain separately pending.'],
        'findings':findings,'pins':{'census_r2':sha(census_path),'census_status':sha(PACKAGE/'scope/root-current-census-reconciliation-r2-status.json'),
            'assembly_status':sha(ASSEMBLY/'status.json'),'assembly_csv':sha(ASSEMBLY/'combined-review-candidate.csv'),'assembly_audit':sha(ASSEMBLY/'combined-review-audit.jsonl')},
        'canonical_changes':False}
write('independent-completeness-review.json',report)
print(json.dumps({'status':report['status'],'counts':report['counts'],'findings':len(findings),'report_sha256':sha(HERE/'independent-completeness-review.json')},indent=2))
