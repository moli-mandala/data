"""Account for every candidate and resolve references without installing edges.

Exact spelling plus an explicitly printed homonym is the only matching rule.
An omitted homonym may match only a unique headword. Ambiguity, self references,
source placeholders and source damage remain explicit review tasks.
"""
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def audit(candidates, entries):
    by_native = collections.defaultdict(list)
    for row in candidates:
        by_native[row['native']].append(row)
    result = []
    for row in candidates:
        source = entries[row['entry_key']]
        meaningful = [t for t in source['tokens'] if t['font'] != 'SegoeUIEmoji']
        bare = all(t['font'] == 'KohinoorDevanagariBold' for t in meaningful)
        if row['kind'] == 'cross-reference':
            classification = 'explicit-cross-reference'
        elif bare:
            classification = 'printed-bare-headword'
        elif any(s['english_gloss'] for s in row['senses']):
            classification = 'definition-present'
        elif row['scientific_name_candidates']:
            classification = 'scientific-identification-without-English-definition'
        elif row['annotations']:
            classification = 'annotation-without-English-definition'
        elif any(s['hindi_raw'] for s in row['senses']):
            classification = 'Hindi-without-English-definition'
        else:
            # Source head + transcription + optional POS, with no definition.
            head_only = all(t['font'] in {'KohinoorDevanagariBold', 'CharisSIL-Bold'}
                            or (t['font'] == 'CharisSIL-Italic' and
                                t['text'] in {'n', 'v', 'clf'}) for t in meaningful)
            classification = 'printed-undefined-transcribed-headword' if head_only else 'unresolved-definition'
        refs = []
        for ref in row['references']:
            targets = []
            for target in ref['targets']:
                matches = by_native[target['native']]
                if target['homonym']:
                    matches = [r for r in matches if r['homonym'] == target['homonym']]
                keys = [r['entry_key'] for r in matches]
                status = ('missing' if not keys else 'ambiguous' if len(keys) > 1
                          else 'self-reference' if keys[0] == row['entry_key'] else 'unique')
                targets.append(dict(target, status=status, candidate_keys=keys))
            refs.append(dict(ref, targets=targets))
        result.append({
            'entry_key': row['entry_key'], 'classification': classification,
            'native': row['native'], 'printed_IPA': row['ipa'],
            'missing_English_senses': [s['printed_sense'] for s in row['senses']
                                       if not s['english_gloss']],
            'review_flags': row['review_flags'], 'references': refs,
            'decision': 'pending-editorial-review', 'installed_rows': 0,
        })
    return result


if __name__ == '__main__':
    candidates = list(map(json.loads, (HERE / 'candidates.jsonl').open()))
    entries = {r['entry_key']: r for r in map(json.loads, (HERE / 'entries.jsonl').open())}
    rows = audit(candidates, entries)
    counts = collections.Counter(r['classification'] for r in rows)
    references = collections.Counter(t['status'] for r in rows for ref in r['references']
                                     for t in ref['targets'])
    payload = {'status': 'uninstalled-review-proposal', 'entries': len(rows),
               'classification_counts': dict(counts), 'reference_target_counts': dict(references),
               'graph_edges_installed': 0, 'records': rows}
    (HERE / 'audit.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in payload.items() if k != 'records'}, indent=2))
