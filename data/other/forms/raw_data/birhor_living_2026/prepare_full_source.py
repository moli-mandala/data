"""Build a reviewable whole-snapshot proposal; never install or build CLDF."""
import collections
import csv
import hashlib
import json
import unicodedata
from pathlib import Path

from import_source import build, SOURCE
from prepare_senses import ROOT, SNAPSHOT_SHA

OUT = ROOT / 'whole-snapshot-recovery'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nfc(value):
    return unicodedata.normalize('NFC', value or '')


def add_tags(row, *tags):
    row[14] = ' '.join(dict.fromkeys(row[14].split() + list(tags)))


def prepare():
    assert sha(ROOT / 'lexical-snapshot.jsonl') == SNAPSHOT_SHA
    rows, head_audit, references = build()
    legacy_keys = {r[10] for r in rows}
    heads = {a['entry_id']: a for a in head_audit}
    reviews = sorted(
        [json.loads(line) for path in OUT.glob('annotation-review-*.jsonl')
         for line in path.open()], key=lambda a: a['annotation_ordinal'])
    assert len(reviews) == 98 and len({r['entry_id'] for r in reviews}) == 98
    audit = []
    review_hashes = {p.name: sha(p) for p in OUT.glob('annotation-review-*.jsonl')}
    # Head keys and original alternatives remain stable; new note attestations get
    # independent keys. We never copy head IPA or Native onto a note-only reading.
    for a in head_audit:
        a['source_unit_key'] = f"{SOURCE}:{a['entry_id']}:head"
        a['entry_keys'] = [r[10] for r in a['emitted_rows']]
        a['section'] = 'structured-head-and-senses'
        del a['emitted_rows']
        audit.append(a)

    packet = json.loads((OUT / 'head-and-grammar-review-packet-20260927.json').read_text())
    assert packet['snapshot_sha256'] == SNAPSHOT_SHA
    for decision in packet['records']:
        eid = decision['entry_id']
        a = heads[eid]
        assert a['raw'] == decision['raw']
        proposed = decision['proposed']
        if isinstance(proposed, list):
            assert not a['entry_keys'], (eid, 'previously emitted head unexpectedly changed')
            for ordinal, form in enumerate(proposed, 1):
                key = f'{SOURCE}:{eid}:recovered-head:{ordinal}'
                tags = ['uncertain']
                if ' ' in form['form']:
                    tags.append('multiword-expression')
                citations = [f'{SOURCE}[entry {eid}, head and scientific identification]']
                citations.extend(r['citation'] for r in a['source_roles'] if r['citation'])
                rows.append(['Birhor', '', form['form'], form['gloss'],
                             form['native'], form['phonemic'], decision['rationale'],
                             ';'.join(citations),
                             '', '', key, '', '', '', ' '.join(tags)])
                a['entry_keys'].append(key)
            a['historical_status'] = a['status']
            a['status'] = 'recovered-head-decision-reviewed-output-pending'
        else:
            for row in rows:
                if row[10] in a['entry_keys']:
                    add_tags(row, *proposed['add_tags'])
        a['whole_source_decision'] = decision
        a['targeted_review'] = 'root-head-and-grammar-review-20260927.json; full output review remains pending'

    media_only = {'1WHapX0E84z6m3mif0wa', 'RKSE2mtUXUausQTLXFtF', 'VXA9A83rCZlUXTZIXsLw'}
    for eid in media_only:
        e = heads[eid]['raw']['entry']
        assert not e['morphology'] and not e['interlinearization']
        for row in rows:
            if row[10] in heads[eid]['entry_keys']:
                row[6] = ''
        heads[eid]['media_note_handling'] = 'Raw note preserved in audit only; not lexical commentary.'

    scope_review = json.loads((OUT / 'head-gloss-scope-review-20260927.json').read_text())
    for decision in scope_review['decisions']:
        eid = decision['entry_id']
        assert heads[eid]['raw'] == decision['source_record']
        matches = [r for r in rows if r[10] in heads[eid]['entry_keys']]
        assert len(matches) == 1
        row = matches[0]
        if 'glosses' in decision:
            row[3] = decision['glosses'][0]
            second = list(row)
            second[3] = decision['glosses'][1]
            second[10] = row[10].replace(':meaning:1:', ':meaning:2:')
            second[7] = row[7].replace('meaning 1]', 'meaning 2]')
            assert second[10] != row[10]
            rows.append(second)
            heads[eid]['entry_keys'].append(second[10])
        else:
            row[3] = decision['gloss']
            add_tags(row, *decision['tags'])
        heads[eid]['head_gloss_scope_decision'] = decision

    expected_candidates = {json.loads(l)['candidate_key']
                           for l in (OUT / 'parenthetical-candidates.jsonl').open()}
    seen_candidates = set()
    for review in reviews:
        eid = review['entry_id']
        note = json.loads(heads[eid]['raw']['entry']['notes'])['default']
        assert hashlib.sha256(note.encode()).hexdigest() == review['whole_note_sha256']
        assert review['full_note_read']
        decisions = []
        for d in review['parenthetical_decisions']:
            assert note[d['start']:d['end']] == d['raw_parenthesis']
            seen_candidates.add(d['candidate_key'])
            decisions.append((d['candidate_key'], d, 'parenthesis'))
        for ordinal, d in enumerate(review['outside_parentheses'], 1):
            decisions.append((f'{SOURCE}:{eid}:note:outside:{ordinal}', d, 'outside'))
        for key, d, kind in decisions:
            original_decision = d
            if kind == 'outside' and (d.get('form') in {
                    'dael', 'bhang', 'laddu', 'chutney', 'pidha', 'okhli'}):
                d = dict(d, disposition='documentary-prose-label-unassigned-language',
                         reason='Independent full-context review: English explanatory prose does not securely assign this label to Birhor; retain documentary evidence without a target lexical row.')
            forms = d.get('forms') or ([d['form']] if d.get('form') else [])
            emit = bool(forms and d.get('gloss') and d['disposition'].startswith('explicit-'))
            emit = emit and d['disposition'] != 'explicit-other-language-control'
            item = {'source_unit_key': key, 'entry_id': eid,
                    'section': 'source-note-attestation', 'decision': d,
                    'whole_note_sha256': review['whole_note_sha256'],
                    'entry_keys': [], 'status': 'proposed' if emit else 'audit-only-context'}
            if d is not original_decision:
                item['historical_owner_decision'] = original_decision
                item['independent_attribution_decision'] = 'Parent full-context review, 2026-09-27; kattha/paan/dael/bhang/laddu/chutney/pidha/okhli documentary-only; explicitly paired loo retained.'
            if emit:
                for ordinal, form in enumerate(forms, 1):
                    row_key = f'{key}:reading:{ordinal}'
                    tags = list(d.get('tags', []))
                    if ' ' in form:
                        tags.append('multiword-expression')
                    uncertainty = d.get('uncertainty') or ('attribution-review' in d['disposition'])
                    if uncertainty:
                        tags.append('uncertain')
                    notes = ('Attested in the dictionary annotation. English rendering is contextual; '
                             'the literal note spelling is retained without an independently supplied '
                             'phonemic transcription.')
                    if uncertainty:
                        notes += ' Attribution or interpretation is qualified: ' + str(
                            d.get('uncertainty') or d.get('reason', 'language attribution uncertain'))
                    if d.get('scope_note'):
                        notes += ' ' + d['scope_note']
                    locator = (f"entry {eid}, note parenthesis {key.split(':')[-1]}"
                               if kind == 'parenthesis' else
                               f"entry {eid}, note outside-parenthesis item {key.split(':')[-1]}")
                    # An upstream botanical or ethnobotanical citation attached to
                    # the head does not automatically support every subsidiary gloss.
                    row = ['Birhor', '', form, d['gloss'], '', '', notes,
                           f'{SOURCE}[{locator}]', '', '', row_key, '', '', '',
                           ' '.join(dict.fromkeys(tags))]
                    rows.append(row)
                    item['entry_keys'].append(row_key)
                if d.get('relation') == 'explicit alternatives' or d['disposition'] == 'explicit-named-term-alternatives':
                    for row in rows[-len(forms):][1:]:
                        row[11] = item['entry_keys'][0]
            audit.append(item)
    assert seen_candidates == expected_candidates
    for row in rows:
        if ' ' in row[2]:
            add_tags(row, 'multiword-expression')
        row[:] = [nfc(v) for v in row]
    assert len({r[10] for r in rows}) == len(rows)
    assert legacy_keys <= {r[10] for r in rows}
    # Reuse only exactly matching subsidiary analyses, never head identities or
    # a form alone. Every physical note attestation retains its own audit key.
    representatives = {}
    aliases = {}
    retained = []
    for row in rows:
        if ':note:' not in row[10]:
            retained.append(row)
            continue
        analysis = tuple(v for i, v in enumerate(row) if i not in (7, 10))
        if analysis not in representatives:
            representatives[analysis] = row
            retained.append(row)
        else:
            first = representatives[analysis]
            aliases[row[10]] = first[10]
            first[7] = ';'.join(dict.fromkeys(first[7].split(';') + row[7].split(';')))
    for row in retained:
        row[11] = aliases.get(row[11], row[11])
    for a in audit:
        a['pre_reuse_entry_keys'] = list(a['entry_keys'])
        a['entry_keys'] = [aliases.get(key, key) for key in a['entry_keys']]
        if a['entry_keys'] != a['pre_reuse_entry_keys']:
            a['exact_attestation_reuse'] = True
    candidate_count = len(rows)
    rows = retained
    live = {r[10] for r in rows}
    assert legacy_keys <= live
    assert all(not r[11] or r[11] in live for r in rows)
    summary = {'status': 'whole-snapshot proposal; independent review and installation pending',
               'snapshot_sha256': SNAPSHOT_SHA, 'source_entries': len(head_audit),
               'rows': len(rows), 'candidate_rows': candidate_count, 'exact_note_reuses': len(aliases),
               'audit_units': len(audit), 'legacy_keys_preserved': len(legacy_keys),
               'annotation_fields': len(reviews), 'parenthetical_spans_accounted': len(seen_candidates),
               'head_statuses': dict(collections.Counter(a['status'] for a in head_audit)),
               'review_input_hashes': review_hashes,
               'head_decisions_sha256': sha(OUT / 'head-and-grammar-review-packet-20260927.json'),
               'head_gloss_scope_sha256': sha(OUT / 'head-gloss-scope-review-20260927.json'),
               'deferred': ['independent output and coverage audit', 'profile routing and actual parser validation',
                            'metadata and references registration', 'canonical installation',
                            'full CLDF/database build', 'compiled graph and browser QA']}
    return rows, audit, summary


if __name__ == '__main__':
    rows, audit, summary = prepare()
    with (OUT / 'proposal.csv').open('w', newline='') as f:
        csv.writer(f).writerows(rows)
    (OUT / 'proposal-audit.jsonl').write_text(''.join(json.dumps(a, ensure_ascii=False) + '\n' for a in audit))
    summary['proposal_hashes'] = {name: sha(OUT / name) for name in ['proposal.csv', 'proposal-audit.jsonl']}
    (OUT / 'proposal-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))
