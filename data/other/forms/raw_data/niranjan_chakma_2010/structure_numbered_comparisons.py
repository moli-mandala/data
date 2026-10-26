"""Structure every reviewed numbered-source line, retaining undecided evidence.

This emits a review intermediate, never the canonical CSV. Language identities,
alternates and graph decisions are deliberately explicit later stages.
"""
import json
import re
from collections import Counter
from pathlib import Path
from record_brokskat_first_reading import sha

ROOT = Path(__file__).resolve().parent
INPUT_SHA = 'bc4ad9434df7ec6ae51c4a37910abc40083dbe5459f110411b7f581900ce2f32'
NUMBER = re.compile(r'^[০-৯]+\.\s*')
LABEL = re.compile(r'^(.+?)\s*-\s*(.*)$')

# Original-page reviewed exceptions: physical baselines are not lexical units.
MULTIPLE = {
    (14, 'সং - কোকিল টা. কোইল'): [('সং', 'কোকিল'), ('টা.', 'কোইল')],
    (16, 'মানি - কদে অস - কুঅং'): [('মানি', 'কদে'), ('অস', 'কুঅং')],
    (16, 'কিরা - কদনু রাজ - কোনঠে'): [('কিরা', 'কদনু'), ('রাজ', 'কোনঠে')],
    (43, 'ফা. দীঅ'): [('ফা.', 'দীঅ')],
}
HEAD_QUALIFIERS = {
    38: ('তিথি', 'পেখ', 'bird qualifier attached to head'),
    72: ('বাস্তূয়া', 'শাক', 'vegetable qualifier attached to head'),
    75: ('ভূই', 'ভুইমাদি', 'source parenthetical alternate; relationship pending'),
    101: ('র্যা', 'অপ্র.', 'source usage abbreviation; expansion unresolved'),
}


def scope_form(text):
    """Keep terminal source punctuation separately; never change spelling."""
    raw = text
    terminal = ''
    while text and text[-1] in '।,':
        terminal = text[-1] + terminal
        text = text[:-1].rstrip()
    before, opening, after = text.partition('(')
    if opening:
        assert text.endswith(')'), text
        return before.strip(), after[:-1], terminal, raw
    return text, None, terminal, raw


def main():
    source = ROOT / 'numbered-literal-first-reading.jsonl'
    review_path = ROOT / 'independent-numbered-complete-review-20260927.json'
    review = json.loads(review_path.read_text())
    assert sha(source) == INPUT_SHA == review['inputs'][source.name]
    assert not review['pending_pages']
    coverage = {r['source_unit_key']: r for r in review['coverage']}
    rows = [json.loads(s) for s in source.read_text().splitlines()]
    result = []
    role_counts = Counter()
    for row in rows:
        g = row['printed_group']
        lines = row['line_witnesses']
        checked = coverage[row['source_unit_key']]
        assert [w['key'] for w in lines] == checked['reviewed_line_keys']
        assert [w['literal'] for w in lines] == checked['reviewed_literal_lines']
        roles = []
        group = {'source_unit_key': row['source_unit_key'], 'printed_group': g,
                 'head': None, 'definition': None, 'comparisons': [], 'source_claims': [],
                 'line_audit': roles, 'transcription_holds': row['source_reading_notes'] + checked['additional_root_findings'],
                 'accepted_for_installation': False}
        def audit(line, role, links):
            roles.append({'source_line_key': line['key'], 'literal': line['literal'],
                          'role': role, 'record_keys': links})
            role_counts[role] += 1
        if g == 48:
            key = row['source_unit_key'] + ':headless-prose'
            group['source_claims'].append({'key': key, 'type': 'headless source commentary',
                'literal_lines': [w['literal'] for w in lines],
                'disposition': 'audit only; no attested Chakma head',
                'named_work': 'তৈত্তিরীয় আরণ্যক', 'quoted_term': 'নগর',
                'reported_place_names': ['চম্পানগর', 'চম্পকনগর'],
                'reference_identity_status': 'work named; edition and passage unspecified'})
            for line in lines:
                audit(line, 'headless-commentary', [key])
            result.append(group)
            continue
        head_line = lines[0]
        head = NUMBER.sub('', head_line['literal']).rstrip(' -')
        assert head != head_line['literal'] and head
        headkey = row['source_unit_key'] + ':head'
        group['head'] = {'key': headkey, 'raw_source_head': head, 'literal_form': head,
                         'language_id': 'Chakma', 'source_line_key': head_line['key'],
                         'qualifier': None, 'alternate_policy': 'literal source group; splitting requires explicit review'}
        if g in HEAD_QUALIFIERS:
            form, qualifier, kind = HEAD_QUALIFIERS[g]
            assert form in head and '(' + qualifier + ')' in head
            group['head'].update(literal_form=form, qualifier={'literal': qualifier, 'type': kind})
        if g == 55:
            assert head == 'পিলেই - পিলোই'
            group['head']['explicit_alternates'] = ['পিলেই', 'পিলোই']
        audit(head_line, 'head', [headkey])
        definition_lines = []
        i = 1
        while i < len(lines):
            line = lines[i]
            literal = line['literal']
            if literal.startswith('অর্থ'):
                assert not definition_lines
                definition_lines.append(re.sub(r'^অর্থ\s*-\s*', '', literal))
                audit(line, 'head-definition', [headkey])
                i += 1
                continue
            if g in (65, 87):
                key = row['source_unit_key'] + ':explicit-source-analysis'
                rest = lines[i:]
                claim = {'key': key,
                    'type': 'explicit origin/components and cited-work prose',
                    'literal_lines': [w['literal'] for w in rest],
                    'graph_status': 'unlinked pending language/reference/claim resolution'}
                if g == 65:
                    assert [w['literal'] for w in rest] == [
                        'উৎসঃ -বাট (পথ) + চেই (চেয়ে)',
                        'বাটচেই>বাড়চেই> বাচ্চেই।',
                        '(শ্রী.কৃ. কী.- এ আছে - ‘বাট', 'চাহিঅ’ অর্থাৎ অপেক্ষা করে)']
                    claim.update(
                        components=[{'key': key + ':component:1', 'literal_form': 'বাট', 'meaning': 'পথ'},
                                    {'key': key + ':component:2', 'literal_form': 'চেই', 'meaning': 'চেয়ে'}],
                        ordered_source_development=['বাটচেই', 'বাড়চেই', 'বাচ্চেই'],
                        cited_work_abbreviation='শ্রী.কৃ. কী.',
                        literary_quotation={'literal_form': 'বাট চাহিঅ', 'meaning': 'অপেক্ষা করে'},
                        language_stage_status='not inferred from form or work abbreviation')
                else:
                    assert [w['literal'] for w in rest] == [
                        'উৎসঃ ‘লম্পক’ শব্দ। অর্থ-', 'অধ্যায়।',
                        'ক্ষেমেন্দ্রের ‘বৃহৎকথা মঞ্জরী’ রচনা',
                        'আঠারো ‘লম্বক’ বা অধ্যায় দ্বারা', 'বিভক্ত।']
                    claim.update(source_origin={'literal_form': 'লম্পক', 'meaning': 'অধ্যায়'},
                        cited_author='ক্ষেমেন্দ্র', cited_work='বৃহৎকথা মঞ্জরী',
                        cited_work_term={'literal_form': 'লম্বক', 'meaning': 'অধ্যায়'},
                        source_reported_divisions=18,
                        reference_status='work and author explicit; edition/passage unspecified',
                        transcription_policy='Keep source-origin প and cited-work ব distinct.')
                group['source_claims'].append(claim)
                for w in rest:
                    audit(w, 'explicit-source-analysis', [key])
                i = len(lines)
                continue
            pairs = MULTIPLE.get((g, literal))
            match = LABEL.match(literal)
            if pairs is None and match:
                pairs = [match.groups()]
            if pairs is None:
                assert definition_lines and not group['comparisons'], (g, literal)
                definition_lines.append(literal)
                audit(line, 'head-definition-continuation', [headkey])
                i += 1
                continue
            # Multiline parentheses attach only to this comparator. No next
            # definition or following language receives their scoped meaning.
            consumed = [line]
            if len(pairs) == 1:
                label, text = pairs[0]
                while (text.count('(') > text.count(')') or
                       (i + 1 < len(lines) and lines[i + 1]['literal'].startswith('('))):
                    i += 1
                    continuation = lines[i]
                    consumed.append(continuation)
                    text += ' ' + continuation['literal']
                pairs = [(label, text)]
            links = []
            for ordinal, (label, text) in enumerate(pairs, 1):
                key = line['key'] + f':comparison:{ordinal}'
                classification = None
                if g == 45 and label == 'বাং.':
                    assert text == 'ঢপ (গড়ন), দেশি'
                    text, classification = 'ঢপ (গড়ন)', 'দেশি'
                form, scoped, punctuation, raw = scope_form(text)
                record = {'key': key, 'source_line_keys': [w['key'] for w in consumed],
                          'source_language_label': label.strip(), 'raw_source_value': raw,
                          'literal_form': form or None, 'source_specific_meaning': scoped,
                          'terminal_punctuation': punctuation, 'language_id': None,
                          'identity_status': 'explicit source label; canonical mapping pending',
                          'disposition': 'explicit source blank' if not form else 'comparison witness pending mapping',
                          'relation': 'comparison only; no ancestry inferred',
                          'accepted_for_installation': False}
                if g == 11 and label == 'ওড়ি':
                    record.update(disposition='unresolved label/form segmentation',
                                  source_specific_meaning=None,
                                  scope_hold='Additional hyphenated item retained whole; do not assign it a language or install as one form.')
                if g == 14 and label == 'টা.':
                    record['scope_hold'] = 'Language-label glyph has an unresolved upper mark.'
                if g == 52 and label == 'প্রা. বাং.':
                    record.update(source_specific_meaning=None, cited_work_literal=scoped)
                if g == 45 and label == 'বাং.':
                    # Parenthetical sense followed by source register label.
                    record['raw_source_value'] = 'ঢপ (গড়ন), দেশি'
                    record['source_classification'] = classification
                    record['scope_hold'] = 'Trailing দেশি is source classification, not lexical head or sense.'
                group['comparisons'].append(record)
                links.append(key)
            for offset, w in enumerate(consumed):
                audit(w, 'comparison' if offset == 0 else 'comparison-meaning-continuation', links)
            i += 1
        assert definition_lines
        group['definition'] = {'literal': ' '.join(definition_lines), 'language': 'Bengali',
                               'scope': 'head; comparators without own meanings inherit only after semantic review'}
        assert [a['source_line_key'] for a in roles] == [w['key'] for w in lines]
        result.append(group)
    assert len(result) == 101
    assert sum(len(r['line_audit']) for r in result) == 620
    keys = [c['key'] for r in result for c in r['comparisons']]
    assert len(keys) == len(set(keys))
    target = ROOT / 'numbered-structured-review.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in result))
    status = {'status': 'whole chapter structurally extracted; identity/glyph/semantic validation pending',
              'groups': 101, 'heads': sum(r['head'] is not None for r in result),
              'comparisons': len(keys), 'line_roles': dict(role_counts),
              'dispositions': dict(Counter(c['disposition'] for r in result for c in r['comparisons'])),
              'inputs': {source.name: sha(source), review_path.name: sha(review_path)},
              'output': {target.name: sha(target)}, 'canonical_changes': False}
    (ROOT / 'numbered-structured-review-status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(status, ensure_ascii=False))


if __name__ == '__main__':
    main()
