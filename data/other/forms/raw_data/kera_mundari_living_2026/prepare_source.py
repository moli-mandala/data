"""Complete deterministic review proposal; never installs canonical files."""
import collections
import csv
import hashlib
import io
import json
import re
import sqlite3
import unicodedata as ud
from pathlib import Path

P = Path(__file__).resolve().parent
SOURCE = 'living-kera-mundari2026'
TAG = 'dialect:mu:kera_mundari_living:Kera%C9%81%20Mundari'
PIN = '8999ed134b13a54bf7780098635ec1c6e07a6f142736ee372278cf793c387764'
DB_PIN = 'ea5178108325f961ae5e34484ea6cafef267862e142377e16b49c553e7b95b00'
POS = {'n': ['noun'], 'adj': ['adj'], 'pro': ['pron'], 'adp': ['adposition'],
       'adv': ['adv'], 'adv.p': ['adv'], 'sent': ['sentential']}

# Exact source pronoun glosses. No grammar is inferred from arbitrary English prose.
PRON = {
 'their-2': ('their', ['third-person', 'du', 'poss']),
 'we': ('we', ['1pl']), 'ours': ('ours', ['1pl', 'poss']),
 'that': ('that', ['demonstrative', 'dist']), 'this': ('this', ['demonstrative', 'prox']),
 'what': ('what', ['interr']), 'who': ('who', ['interr']), 'which': ('which', ['interr']),
 'how many': ('how many', ['interr']), 'how': ('how', ['interr']),
 'why': ('why', ['interr']), 'where': ('where', ['interr', 'spatial']),
 'when': ('when', ['interr', 'temporal']), 'then': ('then', ['temporal']),
 'there': ('there', ['demonstrative', 'dist', 'spatial']),
 'here': ('here', ['demonstrative', 'prox', 'spatial']),
 'you (pl.)': ('you', ['2pl']), 'your (pl.)': ('your', ['2pl', 'poss']),
 'your, you': ('your, you', ['second-person']),
 'his (or her)': ('his (or her)', ['3sg', 'poss']), 'their': ('their', ['3pl', 'poss']),
 'I': ('I', ['1sg']), 'my': ('my', ['1sg', 'poss']), 'our': ('our', ['1pl', 'poss']),
 'he (or she)': ('he (or she)', ['3sg']), 'they': ('they', ['3pl']),
 'they-2': ('they', ['third-person', 'du']),
 'we 2 (not you)': ('we', ['first-person', 'du', 'exclusive']),
 'we 3+ (not you)': ('we', ['1pl', 'exclusive']),
 'me and you': ('me and you', ['first-person', 'du', 'inclusive']),
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def unpack(value, default=None):
    return json.loads(value) if value else default

def readings(raw):
    """Only attested tilde alternatives and single optional parenthesized segments."""
    results = []
    for form in raw.split(' ~ '):
        if '(' in form:
            assert re.fullmatch(r'[^()]+\([^()]+\)[^()]*', form), form
            results.extend([re.sub(r'[()]', '', form), re.sub(r'\([^()]+\)', '', form)])
        else:
            results.append(form)
    return list(dict.fromkeys(ud.normalize('NFC', x.strip()) for x in results))

def make_row(form, gloss, key, locator, tags, notes='', variant='', dialect=TAG):
    return ['mu', '', form, gloss, '', '', notes, SOURCE + '[' + locator + ']', '', '',
            key, variant, '', '', ' '.join(dict.fromkeys(([dialect] if dialect else []) + tags))]

def prepare():
    assert sha(P / 'lexical-snapshot.jsonl') == PIN
    assert sha(P / 'snapshot/kera-mundari.db') == DB_PIN
    raw = [json.loads(line) for line in (P / 'lexical-snapshot.jsonl').read_text().splitlines()]
    context = json.loads((P / 'lexical-context.json').read_text())
    assert len(raw) == 446 and len({x['upstream_entry_id'] for x in raw}) == 446
    labels = {x['id']: unpack(x['name'])['default'] for x in context['dialects_as_source_labels']}
    c = sqlite3.connect(f'file:{P / "snapshot/kera-mundari.db"}?mode=ro', uri=True)
    c.row_factory = sqlite3.Row
    audio = collections.defaultdict(list)
    for r in c.execute('SELECT id,entry_id,storage_path,source,timings,created_at,updated_at FROM audio ORDER BY id'):
        media = dict(r)
        assert media['storage_path'].startswith('kera-mundari/audio/')
        media['original_url'] = 'https://media.livingdictionaries.app/' + media['storage_path']
        audio[r['entry_id']].append(media)
    c.close()
    rows, audit = [], []
    for item in raw:
        e = item['entry']; eid = e['id']; assert len(item['senses']) == 1
        s = item['senses'][0]; key = item['entry_key'] + ':sense:' + s['id']
        lexeme = unpack(e['lexeme'])['default']; gloss = unpack(s['glosses'])['en'].strip()
        raw_note = unpack(e['notes'], {}).get('default', '')
        match = re.fullmatch(r'([A-Za-z]+) (\d+[a-z]?);\s*(9/28/2011)?', raw_note) if raw_note else None
        assert not raw_note or match, raw_note
        pos = unpack(s['parts_of_speech'], []); assert all(x in POS for x in pos)
        tags = [t for p in pos for t in POS[p]]; notes = []
        if pos == ['pro']:
            assert gloss in PRON, gloss
            gloss, extra = PRON[gloss]; tags += extra
            if unpack(s['glosses'])['en'].strip() == 'we 3+ (not you)':
                notes.append('The source specifies three or more referents and excludes the addressee.')
        if match and match[1] == 'Num': tags += ['num']
        if pos == ['adv.p']: notes.append('The source labels this an adverbial phrase.')
        claims = []
        for assignment in item['source_dialect_assignments']:
            label = labels[assignment['dialect_id']]
            assert label in {'English Loan', 'Hindi Loan', 'Possible Hindi Loan'}
            possible = label.startswith('Possible')
            tags += ['loanword'] + (['uncertain'] if possible else [])
            donor = 'English' if label == 'English Loan' else 'Hindi'
            notes.append(f'The source {"tentatively identifies" if possible else "identifies"} this as a borrowing from {donor}.')
            claims.append({'raw_label': label, 'donor_language': donor,
                           'certainty': 'possible' if possible else 'source-asserted',
                           'assignment': assignment, 'donor_form': None, 'graph_edge': None})
        forms = readings(lexeme); emitted = []
        locator = 'entry ' + eid + ', sense ' + s['id']
        if match: locator += ', prompt ' + match[1] + ' ' + match[2]
        if match and match[3]: locator += ', source note dated 2011-09-28'
        for n, form in enumerate(forms, 1):
            target = key + ':reading:' + str(n)
            rowtags = tags + (['multiword-expression'] if ' ' in form else [])
            rows.append(make_row(form, gloss, target, locator, rowtags, ' '.join(notes),
                                 key + ':reading:1' if n > 1 else ''))
            emitted.append(target)
        audit.append({'source_unit_key': item['entry_key'], 'section': 'dictionary',
            'raw': item, 'entry_keys': emitted, 'readings': forms, 'source_pos': pos,
            'semantic_domains': unpack(s['semantic_domains'], []), 'borrowing_claims': claims,
            'prompt': {'code': match[1], 'number': match[2], 'date': '2011-09-28' if match[3] else None} if match else None,
            'media': audio[eid], 'entry_url': 'https://livingdictionaries.app/kera-mundari/entry/' + eid,
            'media_url_status': 'Original URL resolved using the public project media-serving documentation; no media downloaded or playback derivative substituted.',
            'phonemic_status': 'Blank: upstream phonetic field empty; Roman lexeme retained literally without IPA claim.',
            'alternate_policy': 'Explicit ~ and optional parenthesis notation expanded; source lexeme preserved above.',
            'disposition': 'fully emitted'})
    # The about page is separately authored source context, not evidence copied from the cited book.
    about = json.loads((P / 'about-attestations.json').read_text())
    for a in about:
        key = SOURCE + ':about:' + a['key']; emitted = []
        if a['disposition'] in {'target', 'target-family-comparison'}:
            for n, form in enumerate(readings(a['form']), 1):
                target = key + ':reading:' + str(n)
                rows.append(make_row(form, a['gloss'], target, 'about page, ' + a['locator'], a['tags'] +
                    (['multiword-expression'] if ' ' in form else []), a['note'], key + ':reading:1' if n > 1 else '',
                    a.get('dialect_tag', TAG)))
                emitted.append(target)
        audit.append({'source_unit_key': key, 'section': 'about', 'raw': a, 'entry_keys': emitted,
                      'disposition': 'fully emitted' if emitted else 'audit-only control/context'})
    assert len({r[10] for r in rows}) == len(rows)
    assert all(len(r) == 15 and not r[5] for r in rows)
    keys = {r[10] for r in rows}
    assert all(not r[11] or r[11] in keys for r in rows)
    summary = {'status': 'whole pinned source proposal; independent audit and installation pending',
        'raw_entries': 446, 'raw_senses': 446, 'dictionary_rows': sum(len(a['entry_keys']) for a in audit if a['section'] == 'dictionary'),
        'about_units': len(about), 'about_rows': sum(len(a['entry_keys']) for a in audit if a['section'] == 'about'),
        'rows': len(rows), 'audit_units': len(audit), 'audio_records': sum(map(len, audio.values())),
        'source_pos': dict(collections.Counter(tuple(unpack(x['senses'][0]['parts_of_speech'], [])) for x in raw)),
        'notes_reviewed': sum(bool(x['entry']['notes']) for x in raw),
        'borrowing_labels': dict(collections.Counter(z['raw_label'] for a in audit for z in a.get('borrowing_claims', []))),
        'profile_policy': 'Literal NFC Roman spelling, no inferred IPA or phonological conversion. All Phonemic fields blank.',
        'source_sha256': PIN, 'snapshot_sha256': DB_PIN}
    summary['source_pos'] = {'+'.join(k) or '(absent)': v for k,v in summary['source_pos'].items()}
    return rows, audit, summary

def serialize(rows):
    out = io.StringIO(newline=''); csv.writer(out).writerows(rows); return out.getvalue()

def main():
    rows, audit, summary = prepare()
    (P / 'proposal.csv').write_bytes(serialize(rows).encode())
    (P / 'proposal-audit.jsonl').write_text(''.join(json.dumps(a, ensure_ascii=False) + '\n' for a in audit))
    chars = sorted(set(''.join(ud.normalize('NFD', r[2]) for r in rows)) - {' '})
    (P / 'proposal-profile.txt').write_text('Grapheme\tIPA\n' + ''.join(c + '\t' + c + '\n' for c in chars))
    summary['output_hashes'] = {f: sha(P / f) for f in ['proposal.csv', 'proposal-audit.jsonl', 'proposal-profile.txt']}
    (P / 'proposal-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
