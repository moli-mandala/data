"""Record root's exhaustive original-page review of the numbered chapter."""
import json
from collections import Counter
from pathlib import Path
from record_brokskat_first_reading import sha

ROOT = Path(__file__).resolve().parent
EXPECTED = 'bc4ad9434df7ec6ae51c4a37910abc40083dbe5459f110411b7f581900ce2f32'
EXTRA = {
    8: ['Final Che comparator কড্ডা remains a compact glyph hold; do not infer a normalized consonant cluster.'],
    11: ['Original confirms the unresolved ওড়ি line contains an additional hyphenated sequence; retain the whole literal pending label/form segmentation.'],
    14: ['The label before কোইল on the Sanskrit baseline has a small upper mark: টা. versus টাঁ./চাঁ.-like shape remains unresolved. Do not assign its language from the spelling alone.'],
    23: ['Prakrit গব্ভরঅ appears to show separately printed first ব with lower diagonal before ভ. Candidate গব্‌ভরঅ requires narrow confirmation; retain first reading as provisional.'],
    43: ['Head দিঙ and the Persian দীঅ comparator must remain distinct; compact ঙ/ও-like shape is not grounds for automatic cross-column harmonization.'],
    54: ['Final হ্ is not certified: ordinary হ has a similar lower diagonal. Hold head পাহ্ versus পাহ pending font controls.'],
    65: ['Full origin/components/ordered developments/cited quotation verified. Root cannot certify ড versus ড় or the lower mark in first head alternate; preserve typed hold and both first readings.'],
    79: ['Final হ্ is not certified; hold যুহ্ versus যুহ, with no automatic u-vowel insertion.'],
    87: ['Final হ্ is not certified; hold লামাহ্ versus লামাহ. Distinct লম্পক origin and cited-work লম্বক independently confirmed.'],
    97: ['Prakrit হমান appears to contain a compact initial join, possibly হ্মান; no unreviewed replacement. Keep head and comparator separate.'],
}
SCOPE = {
    7: 'Santali label has an explicit blank response; never propagate neighboring Munda form.',
    16: 'Two pairs of language-labelled comparators share baselines; split witnesses by explicit labels, not physical lines alone.',
    19: 'Final গা comparator is explicitly blank.',
    20: 'Tibetan white-scarf and other comparator clothing qualifiers retain their own scopes.',
    25: 'Sanskrit parenthetical meaning differs from the head definition; continuation has four further labelled comparators.',
    29: 'Milky-way head sense, path, embankment and Persian group-of-people senses are separate.',
    38: 'Two Sanskrit forms and head qualifier পেখ retained; no collapse of source comparators.',
    42: 'Five original lines including one Santali line; overlapping OCR fragments do not add a witness.',
    44: 'Parenthetical Sanskrit meaning applies to Sanskrit comparator, not every cell.',
    48: 'Entire headless prose verified; place names and quoted নগর are not an invented Chakma head.',
    52: 'Old Bengali comparator explicitly cites Charyapada; historical stage and work attribution both retained.',
    55: 'Head line supplies two alternatives separated by hyphens; do not mistake second for a gloss.',
    65: 'Explicit components and source development sequence are separate from the cited literary quotation.',
    68: 'Internal head hyphen in বা - না retained as notation; not split into two unrelated words.',
    72: 'Source definition শাক্য বিশেষ retained despite expected ordinary spelling.',
    75: 'Page continuation retains seven additional labelled comparators after the first page.',
    76: 'Sanskrit man and Tamil elephant meanings are distinct scoped source claims.',
    80: 'Head at bottom-left continues with definition and comparators at top-right.',
    86: 'All 21 labelled comparison witnesses retained, including broad and unresolved language abbreviations.',
    87: 'Origin term, named work, named author and eighteen divisions all preserved; no silent harmonization.',
    88: 'Persian two-line definition applies to that comparator only.',
    92: 'Two-line definition precedes three comparator lines continuing onto next page.',
    95: 'Sanskrit two-line gloss differs from head sense and stays scoped to Sanskrit.',
    101: 'Explicit usage abbreviation অপ্র. is outside the head; expansion remains contextual, not guessed.',
}


def main():
    source = ROOT / 'numbered-literal-first-reading.jsonl'
    freeze_path = ROOT / 'numbered-literal-first-reading-freeze-20260927.json'
    freeze = json.loads(freeze_path.read_text())
    assert sha(source) == EXPECTED == freeze['output_sha256']
    for relative, expected in freeze['files'].items():
        assert sha(ROOT / relative) == expected, relative
    rows = [json.loads(s) for s in source.read_text().splitlines()]
    assert [r['printed_group'] for r in rows] == list(range(1, 102))
    pages = {s['pdf_page'] for r in rows for s in r['literal_segments']}
    assert pages == set(range(79, 90))
    assert sum(len(r['literal_segments']) for r in rows) == 115
    assert sum(len(r['line_witnesses']) for r in rows) == 620
    assert sum(r['raw_ocr_fragment_count'] for r in rows) == 621
    coverage = []
    for r in rows:
        group = r['printed_group']
        coverage.append({
            'source_unit_key': r['source_unit_key'], 'printed_group': group,
            'reviewed_line_keys': [w['key'] for w in r['line_witnesses']],
            'reviewed_literal_lines': [w['literal'] for w in r['line_witnesses']],
            'original_pdf_pages': list(dict.fromkeys(s['pdf_page'] for s in r['literal_segments'])),
            'source_head_present': r['source_head_present'],
            'inherited_owner_uncertainty': r['source_reading_notes'],
            'additional_root_findings': EXTRA.get(group, []),
            'scope_check': SCOPE.get(group, 'Head, full definition, all printed labels, comparator forms and qualifiers reviewed.'),
            'status': 'reviewed; retained literal with explicit holds' if r['source_reading_notes'] or group in EXTRA else 'agrees at reviewed original resolution',
            'accepted_for_installation': False,
        })
    target = ROOT / 'independent-numbered-complete-review-20260927.json'
    result = {
        'date': '2026-09-27', 'reviewer': 'root',
        'status': 'all 101 groups independently reviewed; structured extraction and residual glyph adjudication pending',
        'inputs': {source.name: sha(source), freeze_path.name: sha(freeze_path)},
        'source_pdf_sha256': freeze['source_pdf_sha256'],
        'method': 'Read all eleven frozen page transcriptions against all 22 original 3x overlapping half-page images, including every continued group. This is exhaustive second-reading coverage, not blanket glyph acceptance.',
        'original_image_paths': [f'tmp/pdfs/chakma-discovery/numbered-review/p{p}-{half}.jpg' for p in range(79, 90) for half in (1, 2)],
        'counts': {'groups': 101, 'segments': 115, 'literal_lines': 620, 'raw_ocr_fragments': 621,
                   'groups_with_additional_root_findings': len(EXTRA)},
        'pending_pages': [], 'coverage': coverage,
        'policy': ['Preserve immutable first readings and every owner glyph hold unless explicitly resolved.',
                   'No etymological normalization of source spellings.',
                   'No automatic ancestry from comparisons; explicit source developments require separate modelling.',
                   'h-class font caution remains active.',
                   'Language labels and cited-work abbreviations need separate identity/reference mapping.'],
        'canonical_changes': False,
    }
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': target.name, 'sha256': sha(target), **result['counts']}))


if __name__ == '__main__':
    main()
