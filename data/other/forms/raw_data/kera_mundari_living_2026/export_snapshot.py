"""Export every lexical record from the pinned public snapshot, without parsing."""
import collections
import hashlib
import json
import re
import sqlite3
from pathlib import Path

P = Path(__file__).resolve().parent
EXPECTED = 'ea5178108325f961ae5e34484ea6cafef267862e142377e16b49c553e7b95b00'
ADMIN = {'dirty', 'created_by_user_id', 'updated_by_user_id'}


def record(row):
    return {k: row[k] for k in row.keys() if k not in ADMIN}


def main():
    dbpath = P / 'snapshot/kera-mundari.db'
    assert hashlib.sha256(dbpath.read_bytes()).hexdigest() == EXPECTED
    c = sqlite3.connect(f'file:{dbpath.resolve()}?mode=ro', uri=True)
    c.row_factory = sqlite3.Row
    assert c.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    entries = [record(r) for r in c.execute('SELECT * FROM entries ORDER BY id')]
    senses = [record(r) for r in c.execute('SELECT * FROM senses ORDER BY id')]
    assignments = [record(r) for r in c.execute('SELECT * FROM entry_dialects ORDER BY id')]
    dialects = {r['id']: record(r) for r in c.execute('SELECT * FROM dialects ORDER BY id')}
    assert len(entries) == len({e['id'] for e in entries}) == 446
    assert len(senses) == len({s['id'] for s in senses}) == 446
    assert {s['entry_id'] for s in senses} == {e['id'] for e in entries}
    assert all(x['entry_id'] in {e['id'] for e in entries} and x['dialect_id'] in dialects for x in assignments)
    html = (P / 'snapshot/dictionary.html').read_text()
    advertised = re.search(r'entry_count:(\d+)', html)
    assert advertised and int(advertised[1]) == len(entries)
    rows = []
    for e in entries:
        row = {'entry_key': 'living-kera-mundari2026:' + e['id'], 'upstream_entry_id': e['id'],
               'entry': e, 'senses': [s for s in senses if s['entry_id'] == e['id']],
               'source_dialect_assignments': [x for x in assignments if x['entry_id'] == e['id']],
               'status': 'complete raw lexical snapshot; interpretation not yet approved'}
        rows.append(row)
    out = P / 'lexical-snapshot.jsonl'
    out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    dispositions = {}
    lexical_extra = {'entry_relationships', 'relationship_types', 'entry_tags', 'tags', 'sentences',
                     'senses_in_sentences', 'sources', 'texts', 'text_dialects', 'text_tags',
                     'grammar_sections', 'section_sentences', 'glossing_abbreviations', 'clause_slots', 'ignored_forms'}
    extras = {}
    for table in tables:
        count = c.execute('SELECT count(*) FROM "' + table + '"').fetchone()[0]
        if table in {'entries', 'senses', 'dialects', 'entry_dialects'}:
            status = 'fully exported lexical records or annotation labels'
        elif table in lexical_extra:
            extras[table] = [record(r) for r in c.execute('SELECT * FROM "' + table + '" ORDER BY id')]
            status = 'fully exported additional lexical/context table; empty in this snapshot' if not count else 'fully exported additional lexical/context table'
        elif table in {'audio', 'audio_speakers', 'speakers', 'photos', 'sense_photos', 'sense_videos', 'sentence_photos', 'sentence_videos', 'videos', 'video_speakers'}:
            status = 'media/account provenance retained only in public original; no media binaries or personal metadata exported'
        else:
            status = 'administrative/synchronization or UI table; original retained and count accounted'
        dispositions[table] = {'count': count, 'disposition': status}
    (P / 'lexical-context.json').write_text(json.dumps({'dialects_as_source_labels': list(dialects.values()),
        'additional_tables': extras, 'snapshot_metadata': dict(c.execute('SELECT key,value FROM db_metadata'))}, ensure_ascii=False, indent=2) + '\n')
    field_counts = {}
    for table, items in [('entries', entries), ('senses', senses)]:
        field_counts[table] = dict(collections.Counter(k for r in items for k,v in r.items() if v not in (None, '', '[]', '{}')))
    report = {'status': 'complete raw published snapshot acquired; source-stage interpretation and installation deferred',
              'snapshot_sha256': EXPECTED, 'lexical_export_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
              'expected_live_entry_count': int(advertised[1]), 'raw_entries': len(entries), 'raw_senses': len(senses),
              'exported_entry_keys': len(rows), 'distinct_sense_ids': len({s['id'] for s in senses}),
              'all_entry_sense_foreign_keys_resolve': True, 'all_annotation_foreign_keys_resolve': True,
              'pagination': 'Not applicable: complete public SQLite export, not a paginated subset. Every source table counted, every entry and sense exported; count matches independently saved live dictionary metadata.',
              'table_dispositions': dispositions, 'nonempty_fields': field_counts,
              'source_label_semantics': 'The three dialect-table labels are loan-origin annotations, not varieties. Preserve certainty distinctions and do not register them as geographic dialects.',
              'review_required': ['All lexical/POS/native/phonetic fields and notes', 'Source abbreviation provenance (e.g. AN, HU) and contributors',
                                  'Borrowing labels versus explicit donor edges', 'Kera variety registration under mu; no Kurux merge',
                                  'House transcription policy, complete parser and independent audit'],
              'canonical_installation': False}
    (P / 'snapshot-completeness.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['raw_entries','raw_senses','exported_entry_keys','lexical_export_sha256']}))


if __name__ == '__main__':
    main()
