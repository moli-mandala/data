"""Independent complete Native ledger review; no owner input or canonical mutation."""
from pathlib import Path
import json, hashlib, collections

P = Path(__file__).resolve().parent
N = P.parent / 'native-review'
ROOT = Path.cwd()
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    return [json.loads(s) for s in p.read_text().splitlines()]
def write(name, rows):
    (P / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))

atoms = read(N / 'native-atoms-first-reading.jsonl')
lines = read(N / 'native-lines-first-reading.jsonl')
spans = read(N / 'native-roman-span-alignment-proposal.jsonl')
roman = read(P.parent / 'root-reviewed-revision/p238-atomic-reviewed.jsonl')
assert sha(N / 'native-atoms-first-reading.jsonl') == 'b613794c828fb391372d37b0f7a3b131ef6027ac159c77ccbe3cb9a57f2e20b9'
assert sha(N / 'native-lines-first-reading.jsonl') == 'c69a263475fe51b07703e2a93218c72640f1ba160c67525e4510f96b57b983d0'
assert sha(N / 'native-roman-span-alignment-proposal.jsonl') == 'e0df41e7b8c213117d1985669dd1cd895c813e0b51067390bb087944aabce564'
assert sha(P.parent / 'root-reviewed-revision/p238-atomic-reviewed.jsonl') == 'e55cc2a548f8df6b521960224f403502f89436d930a821d3db9a06ab33deab5b'
assert len(atoms) == 194 and len(lines) == 11 and len(spans) == 8
by = {r['key']: r for r in atoms}
rm = {r['source_unit_key']: r for r in roman}
controls = json.loads((P / 'line-control-manifest.json').read_text())
complete = {r['line']: r for r in json.loads((P / 'last-three-line-complete-control-manifest.json').read_text())}
controls = {r['line']: complete.get(r['line'], r) for r in controls}
narrow = {r['name']: r for r in json.loads((P / 'narrow-native-control-manifest.json').read_text())}
for r in list(controls.values()) + list(narrow.values()):
    assert sha(ROOT / r.get('image', r.get('path'))) == r['sha256']
    assert sha(ROOT / r['source']) == r['source_sha256']

def key(line, word):
    return f'lsi9.2:p237:native-line:{line}:word:{word}'

correction = {key(4, 15): ('گھروالے', 'گھروالو', 'line4-second-house-final')}
holds = {
    key(7, 7): {'type': 'Native-letter-encoding',
        'detail': 'The source visibly has the initial m/w/n sequence and compact adjoining h-shaped bodies. Distinction of heh versus do-chashmi-heh in the middle/final letter sequence is not independently secure; retain provisional مونہہ with مونھہ as an image-reading alternative, not a normalized modern spelling.',
        'control': 'line7-mouth'},
    key(9, 16): {'type': 'Native-letter-encoding',
        'detail': 'Compact joined final h/m bodies after the visible j/y/n sequence remain ambiguous; retain provisional جینہم and previously recorded جینھم/جینھہ alternatives. No reconstruction from Roman jaĩh-nai.',
        'control': 'line9-relative-tail'},
    key(11, 4): {'type': 'Native-letter-encoding',
        'detail': 'Compact joined final h/m bodies remain ambiguous in this separately printed occurrence; retain provisional جینہم and previously recorded جینھم/جینھہ alternatives. Do not resolve by the other occurrence or Roman Jaĩh-par.',
        'control': 'line11-relative-tail'},
}
special_observations = {
    key(3, 1): 'Native close control supports the visible alif/y/n/heh sequence encoded اینہ; this is a script reading, not an automatic transliteration of Roman aĩh.',
    key(4, 3): 'Printed name spelling جیکلی is retained, including absence of letters that a Roman-driven normalization might insert.',
    key(4, 13): 'Second occurrence separately supports جیکلی; literal Native spelling is retained independently of Jai-kaḷi.',
    key(6, 13): 'Source presents continuous سونا group; Roman sō and nā remain separate. No per-word Native division forced.',
    key(10, 8): 'Whole-line Native supports the source name spelling حکملے; no Roman-driven repair of the Native name. The narrow personal-name search control clips part of the name and is not used to certify its initial letter.',
    key(11, 6): 'Visible printed numeral-group تین is retained literally; no imported nasal/number phonology.',
    key(11, 7): 'Separate printed numeral-group تین is retained literally; not silently merged with its repetition.',
    key(11, 12): 'Visible تین at this later addressee scope is kept exactly; differing Roman Taĩ-nai is a separate source witness, not permission to emend Native.',
    key(11, 15): 'Narrow native control shows the final nūn bowl with a distinct dot. Provisional spelling دیدون is supported literally; no vocalization or modern nasal phonology inferred.',
}
decisions = []
for a in atoms:
    k = a['key']; line = a['physical_line']; role = a['unit_role']
    evidence = [controls[line]]
    if k in correction:
        old, new, name = correction[k]
        assert a['raw_native'] == old
        evidence.append(narrow[name])
        state = 'secure-literal-correction-proposed'
        observation = 'Last letter is a rounded wāw with descending hook, not the terminal bari-ye shape. Change only final ے to و; source group/key and remaining letters retained. Do not force split Native assignment to Roman lexical rows.'
        current_holds = []
    elif k in holds:
        new = a['raw_native']; state = 'unresolved-literal-glyph-scope'
        current_holds = [holds[k]]
        evidence.append(narrow[holds[k]['control']])
        observation = holds[k]['detail']
    elif role == 'separator-glyph':
        assert a['raw_native'] == '—'
        new = a['raw_native']; state = 'visible-separator-confirmed-editorial-encoding'
        observation = 'Separate printed horizontal stroke confirmed at this exact position. U+2014 is an explicit editorial encoding of the stroke, not a recovered Unicode scalar or an inferred Roman sentence boundary.'
        current_holds = []
    elif role == 'redundant-printed-numeral':
        assert k == key(9, 18) and a['raw_native'] == '۲۱'
        new = a['raw_native']; state = 'visible-additional-numeral-confirmed'
        observation = 'The original prints Eastern Arabic/Persian numeral ۲۱ after the separately spelled ایکیس. Preserve this additional numeral as its own documentary source group, not a duplicate lexical row or omitted token.'
        current_holds = []
    else:
        new = a['raw_native']; state = 'independent-visible-letter-match'
        observation = special_observations.get(k, 'Letters and dots agree with the proposed unvocalized Native spelling at this exact original occurrence. No extra vowels, retroflex phonology or modern Urdu spelling inserted.')
        current_holds = []
    decisions.append({'source_unit_key': k, 'physical_line': line, 'word_rank': a['word_rank'],
        'old_native_guard': a['raw_native'], 'proposed_native': new, 'unit_role': role,
        'decision': state, 'observation': observation,
        'current_typed_holds': current_holds, 'historical_owner_uncertainties': a['typed_uncertainties'],
        'evidence': evidence, 'policy': 'Independent Native-letter/source-group decision only; no per-Roman-word equivalence or phonemic value certified.'})
assert len({r['source_unit_key'] for r in decisions}) == 194
write('all-194-keyed-native-decisions.jsonl', decisions)
write('secure-keyed-native-corrections.jsonl', [r for r in decisions if r['decision'] == 'secure-literal-correction-proposed'])
write('current-keyed-native-glyph-holds.jsonl', [r for r in decisions if r['current_typed_holds']])

line_review = []
for line in lines:
    selected = [by[k] for k in line['native_word_keys']]
    assert ' '.join(r['raw_native'] for r in selected) == line['raw_native_line']
    assert all(r['physical_line'] == selected[0]['physical_line'] for r in selected)
    line_review.append({'source_line_key': line['key'], 'native_keys': line['native_word_keys'],
        'old_line_guard': line['raw_native_line'], 'group_count': len(selected),
        'decision': 'Complete line visually reviewed; original letter decisions recorded for every keyed editorial spacegroup including separators and numeral.',
        'evidence': controls[selected[0]['physical_line']]})
write('all-11-line-review.jsonl', line_review)

counter_n = collections.Counter(k for s in spans for k in s['native_keys'])
counter_r = collections.Counter(k for s in spans for k in s['roman_keys'])
assert counter_n == collections.Counter(by.keys())
assert counter_r == collections.Counter(rm.keys())
span_review = []
for s in spans:
    assert s['raw_native_span'] == ' '.join(by[k]['raw_native'] for k in s['native_keys'])
    roman_current = ' '.join(rm[k]['raw_form'] for k in s['roman_keys'])
    gloss_current = ' '.join(rm[k]['raw_gloss'] for k in s['roman_keys'])
    span_review.append({'span_key': s['key'], 'native_keys': s['native_keys'], 'roman_keys': s['roman_keys'],
        'native_physical_lines': s['native_physical_lines'], 'old_raw_native_guard': s['raw_native_span'],
        'current_native_proposal_with_keyed_correction': ' '.join(next(r['proposed_native'] for r in decisions if r['source_unit_key'] == k) for k in s['native_keys']),
        'current_native_hold_keys': [k for k in s['native_keys'] if k in holds],
        'historical_roman_span_guard': s['raw_roman_span'], 'reviewed_roman_physical_span': roman_current,
        'historical_interlinear_guard': s['printed_interlinear_analysis'], 'reviewed_interlinear_span': gloss_current,
        'stale_roman_span_projection': s['raw_roman_span'] != roman_current,
        'decision': 'Broad narrative/dialogue progression supports this contiguous provenance span. Exact word-for-word Native alignment is NOT asserted; retain all shared keys and script differences.',
        'scope': 'Physical Native lines and broad speech succession, not sentence boundaries or lexical row equivalence.'})
write('all-8-broad-script-span-decisions.jsonl', span_review)

report = {'status': 'Complete independent194 Native source-group review; one secure correction and three explicit letter-encoding holds for parent adjudication',
    'reviewer': '/root/asur_finish',
    'counts': {'Native_editorial_spacegroups': 194, 'physical_Native_lines': 11,
        'script_letter_groups': 164, 'horizontal_separator_groups': 29, 'additional_numeral_groups': 1,
        'independent_visible_letter_matches': 160, 'secure_literal_corrections_proposed': 1,
        'unresolved_literal_groups': 3, 'broad_script_spans': 8,
        'Roman_physical_keys_conserved_in_broad_spans': 133},
    'reading_scope': 'All11 original Native lines read against full top/middle/bottom native-pixel crops, immutable line controls, and narrowly scoped letter controls. Source-original spelling retained without Roman normalization.',
    'segmentation_policy': '194 are the preserved first-reading editorial spacegroups, not a claim of194 distinct lexical words or exact linguistic word boundaries in disjoined Persian typography. Spaces inside some clusters may be ambiguous; no new Native grouping or per-word Roman projection manufactured.',
    'punctuation_policy': '29 visible horizontal separators preserved as U+2014 under an explicit editorial encoding policy; not interpreted as quotation punctuation or automatically matched to Roman sentence periods.',
    'numeral_policy': 'Eastern Arabic/Persian ۲۱ is a separately printed redundant numeral after ایکیس; documentary accounting remains explicit and no extra target lexeme invented.',
    'qualified_safe_spans': 'Eight contiguous script spans conserve every194 Native and133 Roman source key exactly once and preserve broad narrative progression; they are safe provenance spans, not certified equivalent full Native fields for narrower atoms or nested quotations.',
    'history_policy': 'Historical first-reading Native strings/typed holds and stale Roman-span projections remain guarded. Any accepted correction must be a separate keyed derivative; never overwrite frozen inputs.',
    'stale_roman_span_projection_count': sum(r['stale_roman_span_projection'] for r in span_review),
    'input_hashes': {str(f.relative_to(ROOT)): sha(f) for f in [N / 'native-atoms-first-reading.jsonl',
        N / 'native-lines-first-reading.jsonl', N / 'native-roman-span-alignment-proposal.jsonl',
        P.parent / 'root-reviewed-revision/p238-atomic-reviewed.jsonl', N / 'original-crop-provenance.json']},
    'outputs': {f.name: sha(f) for f in P.glob('*.jsonl')},
    'control_manifests': {f.name: sha(f) for f in [P / 'line-control-manifest.json',
        P / 'last-three-line-complete-control-manifest.json', P / 'narrow-native-control-manifest.json']},
    'script_sha256': sha(Path(__file__)),
    'deferred': ['Parent exact-key correction/hold adjudication and separately guarded accepted Native provenance derivative.',
        'Narrow lexical and whole row Native remain blank; no certification of perword Native equivalence.',
        'No canonical installation, DB/full build, remote jobs or publication performed.']}
(P / 'independent-full-native-194-review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report['counts']))
