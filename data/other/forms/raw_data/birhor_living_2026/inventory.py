"""Freeze public lexical fields from a pinned upstream snapshot, read-only."""
import argparse
import collections
import hashlib
import json
import sqlite3
from pathlib import Path

SHA256 = 'f52cb1314d54ed313a9dffa70f78b57dc19f384d6830995cf29c8b165c6535d4'
ENTRY_FIELDS = 'id lexeme phonetic interlinearization morphology notes linguistic_history sources scientific_names coordinates unsupported_fields homograph citations review created_at updated_at'.split()
SENSE_FIELDS = 'id entry_id definition glosses parts_of_speech semantic_domains write_in_semantic_domains noun_class plural_form variant sources created_at updated_at'.split()

def inventory(database, output):
    assert hashlib.sha256(database.read_bytes()).hexdigest() == SHA256, 'Different snapshot; review revision first'
    c = sqlite3.connect(f'file:{database.resolve()}?mode=ro', uri=True)
    c.row_factory = sqlite3.Row
    assert c.execute('pragma integrity_check').fetchone()[0] == 'ok'
    output.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    fields = collections.Counter()
    lexeme_keys = collections.Counter()
    gloss_keys = collections.Counter()
    with (output/'lexical-snapshot.jsonl').open('w') as handle:
        for entry in c.execute('select '+','.join(ENTRY_FIELDS)+' from entries order by id'):
            senses = [dict(s) for s in c.execute('select '+','.join(SENSE_FIELDS)+' from senses where entry_id=? order by id',(entry['id'],))]
            record = {'entry': dict(entry), 'senses': senses}
            handle.write(json.dumps(record,ensure_ascii=False)+'\n')
            counts['entries'] += 1
            counts['senses'] += len(senses)
            for k,v in dict(entry).items():
                if v not in (None,'','[]','{}'): fields['entry.'+k] += 1
            lexeme_keys.update(json.loads(entry['lexeme']).keys())
            for s in senses:
                for k,v in s.items():
                    if v not in (None,'','[]','{}'): fields['sense.'+k] += 1
                gloss_keys.update(json.loads(s['glosses'] or '{}').keys())
    source_fields = 'id slug citation abbreviation author year url license type orthography'.split()
    sources = [dict(r) for r in c.execute('select '+','.join(source_fields)+' from sources order by id')]
    (output/'source-references.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n')
    counts.update({t:c.execute(f'select count(*) from {t}').fetchone()[0] for t in ['dialects','entry_dialects','entry_relationships','sources','sentences','audio','photos','videos','grammar_sections']})
    report={'snapshot_sha256':SHA256,'metadata':dict(c.execute('select key,value from db_metadata')),'counts':dict(counts),'nonempty_fields':dict(fields),'lexeme_keys':dict(lexeme_keys),'gloss_languages':dict(gloss_keys),'lexical_snapshot_sha256':hashlib.sha256((output/'lexical-snapshot.jsonl').read_bytes()).hexdigest(),'installed_rows':0,'status':'inventory only; no parsing or lexical acceptance yet'}
    (output/'inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(); inventory(a.database,a.output)
