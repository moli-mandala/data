#!/usr/bin/env python3
"""Prepare an offline Kharia Living Dictionaries proposal from its pinned SQLite.

User confirmed all source reuse permissions on 2026-09-11. The snapshot is
the same public compressed SQLite used by its viewer.
No audio, photographs, or speaker personal metadata are copied into the audit.
"""
import argparse
import collections
import csv
import hashlib
import json
import random
import re
import sqlite3
import shutil
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SOURCE = 'living-kharia2026'
STEM = '20260911-kharia-living'
SQLITE_SHA256 = 'c0bf44b14565a019de565ca82b45e01da177f2453cda00c460428acaa22ea3d2'
URL = 'https://snapshots.livingdictionaries.app/dictionaries/kharia.db.gz'
DIALECTS = {'Dhelki': ['dhelki'], 'Dudh': ['dudh'], 'Dhelki/Dudh': ['dhelki', 'dudh']}


def nfc(s):
    return unicodedata.normalize('NFC', s or '').strip()


def expand_phonetic(value):
    """Expand printed alternation/optional segments; two English comments are prose."""
    comments = re.findall(r'\(same as (?:baby|child)\)', value)
    value = re.sub(r'\s*\(same as (?:baby|child)\)', '', value)
    forms = []
    for part in value.split('~'):
        pending = [nfc(part)]
        while any('(' in p for p in pending):
            expanded = []
            for p in pending:
                match = re.search(r'\(([^()]*)\)', p)
                if not match:
                    assert '(' not in p and ')' not in p, p
                    expanded.append(p)
                    continue
                assert ' ' not in match[1], f'Unknown parenthetical prose: {p}'
                expanded.extend([p[:match.start()]+match[1]+p[match.end():],
                                 p[:match.start()]+p[match.end():]])
            pending = expanded
        forms.extend(pending)
    return list(dict.fromkeys(forms)), comments


def emit_record(record):
    row = record['base_row']
    forms, comments = expand_phonetic(record['raw_phonetic']) if record['raw_phonetic'] else ([row[4]], [])
    gloss = row[3]
    senses = re.split(r'(?:^|\s)[12]\.\s*', gloss)[1:] if gloss.startswith('1. ') else [gloss]
    result = []
    for si, gloss in enumerate(senses, 1):
        for vi, form in enumerate(forms, 1):
            child = row.copy()
            child[2] = nfc(form)
            child[3] = gloss.strip().rstrip('.') if len(senses) > 1 else gloss
            child[5] = nfc(form) if record['raw_phonetic'] else ''
            child[6] = '; '.join(comments)
            child[10] += f':reading:{si}:v{vi}'
            child[11] = row[10]+f':reading:{si}:v1' if vi > 1 else ''
            result.append(child)
    return result


def propose(database, output):
    assert hashlib.sha256(database.read_bytes()).hexdigest() == SQLITE_SHA256, 'Snapshot changed; review the new revision explicitly'
    c = sqlite3.connect(f'file:{database.resolve()}?mode=ro', uri=True)
    c.row_factory = sqlite3.Row
    assert c.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    entries = [dict(r) for r in c.execute('SELECT * FROM entries ORDER BY id')]
    assert len(entries) == 452
    dialects = {r['id']: json.loads(r['name'])['default'] for r in c.execute('SELECT * FROM dialects')}
    assigned = collections.defaultdict(list)
    for r in c.execute('SELECT * FROM entry_dialects ORDER BY entry_id,dialect_id'):
        assigned[r['entry_id']].append(dialects[r['dialect_id']])
    rows, audit = [], []
    for entry in entries:
        key = f"{SOURCE}:{entry['id']}"
        senses = [dict(r) for r in c.execute('SELECT * FROM senses WHERE entry_id=? ORDER BY id', (entry['id'],))]
        assert len(senses) == 1
        sense = senses[0]
        assert not any(entry[k] for k in ['interlinearization','morphology','notes','linguistic_history','sources','scientific_names','coordinates','unsupported_fields','homograph','citations','review','tone_class'])
        assert not any(sense[k] for k in ['definition','parts_of_speech','noun_class','plural_form','variant','sources'])
        glosses = json.loads(sense['glosses'])
        native = nfc(json.loads(entry['lexeme'])['default'])
        phonetic = nfc(entry['phonetic'])
        tags = []
        for label in assigned[entry['id']]:
            for d in DIALECTS[label]:
                tags.append(f'dialect:kh:{d}:{d.title()}')
        review = []
        if not phonetic:
            review.append('transcription:source-has-no-phonetic-form')
        if phonetic and any(ch in phonetic for ch in 'ăĕĭŏŭ̆ˀ'):
            review.append('transcription:preserve-source-shortening-or-glottalization-notation')
        # Native-only additions remain native-script attestations. Do not invent
        # a phonetic transcription by applying Hindi schwa-deletion rules.
        row = ['kh', '', phonetic or native, nfc(glosses.get('en')), native, phonetic, '',
               f"{SOURCE}[entry {entry['id']}, sense {sense['id']}]", '', '',
               f"{key}:sense:{sense['id']}", '', '', '', ' '.join(tags)]
        if review:
            row[-1] += ' uncertain'
            row[-1] = row[-1].strip()
        record = dict(entry_key=key, upstream_entry_id=entry['id'], upstream_sense_id=sense['id'],
                      raw_lexeme=entry['lexeme'], raw_phonetic=entry['phonetic'],
                      raw_glosses=glosses, raw_semantic_domains=json.loads(sense['semantic_domains'] or '[]'),
                      raw_dialects=assigned[entry['id']],
                      entry_updated_at=entry['updated_at'], sense_updated_at=sense['updated_at'],
                      review=review, status='proposed',
                      base_row=row)
        record['rows'] = emit_record(record)
        rows.extend(record['rows'])
        audit.append(record)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'proposed.csv').open('w') as f:
        csv.writer(f).writerows(rows)
    with (output/'audit.jsonl').open('w') as f:
        for r in audit:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    (output/'sample.json').write_text(json.dumps(random.Random(20260912).sample(audit,20),ensure_ascii=False,indent=2)+'\n')
    report = dict(source=SOURCE, snapshot_url=URL, sqlite_sha256=hashlib.sha256(database.read_bytes()).hexdigest(),
                  expected_articles=452, proposed_rows=len(rows), native_only_rows=sum(not r[5] for r in rows),
                  variant_rows=sum(bool(r[11]) for r in rows),
                  installed_rows=0, installation_ready=False,
                  permission='User confirmed all selected-source reuse permissions on 2026-09-11',
                  source_dialects=dict(collections.Counter(x for r in audit for x in r['raw_dialects'])),
                  excluded_material='438 audio records, 69 photographs, speaker metadata, and 3 example sentences are not lexical headword rows; no media files fetched',
                  deferred_gates=['fresh-audit','focused-tests','full-build','full-tests','browser-QA'])
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'tmp/kharia-living-proposal-20260911')
    p.add_argument('--install',action='store_true')
    a=p.parse_args()
    report=propose(a.database,a.output)
    if a.install:
        shutil.copyfile(a.output/'proposed.csv',ROOT/f'data/other/forms/{STEM}.csv')
        audit=[json.loads(line) for line in (a.output/'audit.jsonl').read_text().splitlines()]
        for r in audit:
            r['status']='installed-raw'
        (ROOT/f'data/other/forms/raw_data/{STEM}-audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit))
        report['installed_rows']=report['proposed_rows']
        (ROOT/f'source_checklists/{STEM}-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
