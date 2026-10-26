"""Independent guarded metadata proposals; never rewrites owner inputs."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[2]
REVIEW = PACKAGE / 'specimens/specimen37/independent-metadata-review/residual-revision4-proposal'
BASE = PACKAGE / 'specimens/specimen37/independent-review/projection-revision3/candidate.csv'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
def jsonl(name, rows):
    (HERE / name).write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))
status = json.loads((REVIEW / 'status.json').read_text())
for rel, expected in status['inputs'].items():
    assert sha(PACKAGE / rel) == expected, rel
for rel, expected in status['outputs'].items():
    assert sha(REVIEW / rel) == expected, rel
old = list(csv.reader(BASE.open()))
new = list(csv.reader((REVIEW / 'candidate.csv').open()))
decisions = [json.loads(x) for x in (REVIEW / 'all390-keyed-decisions.jsonl').read_text().splitlines()]
assert len(old) == len(new) == len(decisions) == 390
assert len({r[10] for r in new}) == 390
protected = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]
changed = 0
generic = ('The printed interlinear English supplies the time, voice or subject interpretation; '
           'the corresponding intrinsic form classification is not specified.')
proposals = []
full = []
classes = Counter()
number_review = []
for before, row, decision in zip(old, new, decisions):
    assert len(row) == 15 and before[10] == row[10] == decision['source_unit_key']
    assert decision['old_row'] == before and decision['proposed_row'] == row
    assert all(before[i] == row[i] for i in protected)
    delta = [i for i in range(15) if row[i] != before[i]]
    assert set(delta) <= {6, 7, 14}
    changed += bool(delta)
    revised = list(row)
    reasons = []
    page = row[10].split(':')[1][1:]
    if generic in revised[6]:
        revised[6] = revised[6].replace(generic, '').strip()
        classes['generic_intrinsic_analysis_notes'] += 1
        reasons.append('Retain typed intrinsic-analysis uncertainty in audit; repeated reviewer reasoning is not a scholarly source claim.')
    if revised[6] == 'Printed interlinear English analysis.':
        revised[6] = ''
        classes['whole_raw_analysis_notes'] += 1
        reasons.append('The exact printed English is already Gloss and audit evidence; this generic provenance note is redundant.')
    if revised[6].startswith('The printed interlinear analysis reads '):
        revised[6] = ''
        classes['raw_head_only_agent_notes'] += 1
        if row[2] in ['Un', 'un', 'un-taĩ']:
            revised[6] = ('Grierson lists un as an oblique plural demonstrative also used in the singular; '
                          'the specimen uses it in a personal-reference context.')
            classes['source_un_claim_retained'] += 1
        elif '-nē̃' not in row[2]:
            revised[7] = f'grierson1908ahirwati[p. {page}, specimen 37]'
            classes['unsupported_agent_comparison_citations'] += 1
        else:
            revised[6] = ('Grierson describes nē̃ as both the dative and agent-case suffix; '
                          'this occurrence is glossed with “by”.')
            classes['source_agent_claim_retained'] += 1
        reasons.append('Replace raw-gloss repetition with the actual p. 51 nē̃ or un claim where applicable; taĩ receives no nē̃-based classification.')
    if revised[6].startswith('The interlinear analysis reads ‘might-have-done’'):
        revised[6] = ''
        classes['modal_raw_analysis_note'] += 1
        reasons.append('Preserve raw modal translation and subtype hold in audit rather than duplicate it in Notes.')
    tags = row[14].split()
    if 'noun' in tags and 'pl' in tags:
        revised[14] = ' '.join(x for x in tags if x != 'pl')
        classes['unsupported_nominal_plural'] += 1
        reasons.append('The printed English plural denotes referents; pp. 50–51 do not identify this literal as a plural noun class or provide a plural paradigm. No target number inferred from English or nasal endings.')
        number_review.append({'source_unit_key': row[10], 'form': row[2], 'raw_gloss': row[3],
                              'decision': 'withhold noun pl only', 'target_number_scope': 'not independently specified',
                              'source_class_limit': 'Strong masculine paragraph concerns nominative/oblique singular, not these plural readings.'})
    if ('Grierson gives apᵃṇū/apᵃṇō' in revised[6] or row[2] in ['‘mhārē', 'karᵃṇō']):
        if '51' not in revised[7]:
            revised[7] = f'grierson1908ahirwati[pp. 51 and {page}, specimen 37 and grammatical analysis]'
            classes['missing_specific_claim_citations'] += 1
            reasons.append('The retained scholarly statement quotes p. 51; include its actual evidence page alongside the primary occurrence.')
    edits = {str(i): {'old': row[i], 'new': revised[i]} for i in [6,7,14] if row[i] != revised[i]}
    item = {'source_unit_key': row[10], 'exact_form_guard': row[2], 'exact_gloss_guard': row[3],
            'protected_row_sha256': hashlib.sha256(json.dumps(row, ensure_ascii=False).encode()).hexdigest(),
            'reviewed_r3_to_r4_columns': delta, 'decision': 'secure refinement proposed' if edits else 'retain',
            'changes': edits, 'reasons': reasons,
            'basis': 'Independent complete No37 original review plus native p. 51 grammar and p. 237 commentary; lexical POS distinct from intrinsic morphology.'}
    full.append(item)
    if edits: proposals.append(item)
assert changed == 138
assert classes['generic_intrinsic_analysis_notes'] == 47
assert classes['whole_raw_analysis_notes'] == 27
assert classes['unsupported_nominal_plural'] == 15
assert all('obj' not in r[14].split() or '(obj.)' in r[3] for r in new)
assert all(not ({'pret','pass','participle','fut','pres','2sg'} & set(r[14].split()))
           for r in new if 'matches a copula' in r[6])
jsonl('all390-keyed-review.jsonl', full)
jsonl('guarded-refinement-proposals.jsonl', proposals)
jsonl('nominal-number-scope-review.jsonl', number_review)
evidence = [
    PACKAGE / 'specimens/specimen37/independent-review/independent-full-original-review.json',
    PACKAGE / 'specimen38/root-documentary-review/p237-linguistic-commentary-native.png',
    Path('tmp/jambu-next-source/ahirwati-native-grammar-20260927/p51-first-reading-copulas.png'),
    Path('tmp/jambu-next-source/ahirwati-native-grammar-20260927/p51-first-reading-pronouns.png'),
    Path('tmp/jambu-next-source/ahirwati-native-grammar-20260927/p51-first-reading-nominals.png'),
    Path('tmp/jambu-next-source/ahirwati-native-grammar-20260927/printed-50-pdf-67-Im67.png'),
]
report = {'status': 'refinement required; no owner inputs modified', 'reviewer': '/root/asur_finish',
          'reviewed_rows': 390, 'reviewed_changed_rows': 138, 'protected_fields_pass': True,
          'proposal_rows': len(proposals), 'finding_counts': dict(classes),
          'candidate_sha256': sha(REVIEW / 'candidate.csv'), 'projection_r3_sha256': sha(BASE),
          'input_status_sha256': sha(REVIEW / 'status.json'),
          'source_evidence_sha256': {str(x): sha(x) for x in evidence},
          'retained_classes': ['p51 exact present/past copulas including 2sg sā and masculine-plural thā',
              'p237 explicitly past-participial bōlyō', 'p51 apᵃṇā masculine oblique',
              'p51 ghari masculine locative and mhārē agreement',
              'p51 future-passive participle karᵃṇō', 'explicit printed obj label',
              'qualified final-component copula comparisons without whole-compound TAM/person'],
          'nominal_count_clarification': 'Actual frozen candidate contains 15 noun+pl rows, not 17. Three pronominal-plural rows and one thā copula-plural row are separate retained categories.',
          'limits': 'Independent complete keyed metadata scope review; prior full literal review reused with hashes. No new claim of Native recertification, canonical installation, parser/full-build or browser validation.',
          'outputs': {name: sha(HERE/name) for name in ['all390-keyed-review.jsonl','guarded-refinement-proposals.jsonl','nominal-number-scope-review.jsonl']}}
dump('independent-residual-review.json', report)
print(json.dumps({'rows':390,'changed_reviewed':138,'proposals':len(proposals),'findings':dict(classes),'report_sha256':sha(HERE/'independent-residual-review.json')},indent=2))
