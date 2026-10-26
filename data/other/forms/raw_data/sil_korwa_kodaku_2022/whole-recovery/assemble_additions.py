"""Assemble all current additions for review; never writes canonical assets."""
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[5]
sys.path.insert(0, str(DATA))
from dialects import dialect_tag

OUT = HERE / 'integration-proposal'
identity_path = HERE / 'identity-review/all-source-routing-proposal.json'
identity = json.loads(identity_path.read_text())
paths = [identity_path, OUT / 'comparison-rows.csv', OUT / 'comparison-routing-audit.jsonl',
         OUT / 'comparison-dialects.csv', HERE / 'interlinear-review/whole-expression-proposal.jsonl',
         HERE / 'interlinear-review/c5-column-proposal.jsonl',
         HERE / 'interlinear-review/continuous-semantic-proposal.jsonl',
         HERE / 'outside-review/outside-candidate.jsonl']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
pins = {str(p.relative_to(HERE)): sha(p) for p in paths}
read_jsonl = lambda p: [json.loads(line) for line in p.read_text().splitlines()]
rows = list(csv.reader(paths[1].open()))
audit = read_jsonl(paths[2])
dialects = {r['ID']: r for r in csv.DictReader(paths[3].open())}
existing = {r['ID']: r for r in csv.DictReader((DATA / 'cldf/dialects.csv').open())}
routes = {r['section']: dict(r) for r in identity['texts']}
routes['C.3']['dialect_id'] = 'sil-kodaku-kanda'
kanda = {'ID': 'sil-kodaku-kanda', 'Tag': dialect_tag('Kodaku', 'sil-kodaku-kanda', 'Kanda Kodaku'),
         'Language_ID': 'Kodaku', 'Source_Language_ID': 'Kanda', 'Name': 'Kanda Kodaku',
         'Glottocode': '', 'Latitude': '', 'Longitude': '', 'Clade': 'Munda',
         'Location': 'Kanda, Surguja, Chhattisgarh; Behera Table 7, printed p. 17', 'Quality': ''}
assert kanda['ID'] not in existing
dialects[kanda['ID']] = kanda
new_language = dict(identity['proposed_language_registry_row'])
assert not new_language['Latitude'] and not new_language['Longitude']
new_language['Quality'] = ''
for path in paths[4:7]:
    for candidate in read_jsonl(path):
        record = dict(candidate)
        record['input_file'] = str(path.relative_to(HERE))
        record['accepted_for_installation'] = False
        if candidate['role'] in {'unaligned-source-span', 'unresolved-internal-alignment-span'}:
            record['integration_disposition'] = 'held-internal-word-alignment; target retained in whole expression'
            audit.append(record)
            continue
        route = routes[candidate['section']]
        assert route['language'] == candidate['language']
        dialect = dialects.get(route['dialect_id'], existing.get(route['dialect_id']))
        assert dialect and dialect['Language_ID'] == candidate['language']
        tags = list(candidate.get('tags', []))
        if ' ' in candidate['form'] and 'multiword-expression' not in tags:
            tags.append('multiword-expression')
        if candidate.get('typed_uncertainty') and 'uncertain' not in tags:
            tags.append('uncertain')
        tags.append(dialect['Tag'])
        if candidate['role'] in {'whole-target-question', 'source-target-answer'}:
            locator = 'question ' + candidate['question_number']
            if candidate['role'] == 'source-target-answer':
                locator += ', answer'
        else:
            labels = candidate.get('printed_subnumbers', [candidate['printed_number']])
            locator = 'no. ' + ', '.join(labels)
        source = ';'.join(f'behera2022korwakodaku[Appendix {candidate["section"]}, p. {p-10}, {locator}]'
                          for p in candidate['source_pages'])
        row = [candidate['language'], '', candidate['form'], candidate['gloss'], '', '', '',
               source, '', '', candidate['source_unit_key'], '', '', '', ' '.join(dict.fromkeys(tags))]
        rows.append(row)
        record.update(integration_disposition='text-review-candidate', proposed_csv_row=row,
                      dialect_id=dialect['ID'], notes_policy='Source review/analysis notes retained in this audit')
        audit.append(record)
for candidate in read_jsonl(paths[7]):
    record = dict(candidate)
    record['accepted_for_installation'] = False
    if candidate['disposition'] != 'eligible-source-attributed-lexical':
        record['integration_disposition'] = candidate['disposition']
        audit.append(record)
        continue
    row = [candidate['language'], '', candidate['form'], candidate['gloss'], '', '', '',
           candidate['source'], '', candidate.get('etymology', candidate.get('Etymology', '')),
           candidate['entry_key'], '', '', '', ' '.join(candidate['tags'])]
    assert row[9], 'Qualified source claim must survive in Etymology'
    rows.append(row)
    record.update(integration_disposition='outside-review-candidate', proposed_csv_row=row)
    audit.append(record)
assert all(len(r) == 15 and r[2] for r in rows)
assert len(rows) == len({r[10] for r in rows})
assert pins == {str(p.relative_to(HERE)): sha(p) for p in paths}, 'Input changed during assembly'

def write_csv(path, records):
    with path.open('w', newline='') as handle:
        csv.writer(handle).writerows(records)

write_csv(OUT / 'all-additions.csv', rows)
fields = list(kanda)
write_csv(OUT / 'all-new-dialects.csv', [fields] + [[r[f] for f in fields] for r in dialects.values()])
fields = list(new_language)
write_csv(OUT / 'new-language.csv', [fields, [new_language[f] for f in fields]])
(OUT / 'all-additions-audit.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in audit))
report = {'status': 'Whole-report review assembly; not installed or accepted', 'inputs': pins,
          'rows': len(rows), 'languages': dict(Counter(r[0] for r in rows)),
          'audit_records': len(audit),
          'dispositions': dict(Counter(r['integration_disposition'] for r in audit)),
          'new_dialects': len(dialects), 'new_languages': 1,
          'canonical_changes': False,
          'pending': ['Independent text/alignment corrections', 'Final source-wide audit',
                      'Actual combined parser/registry/reference checks', 'Guarded installation']}
(OUT / 'all-additions-status.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'inputs'}, indent=2))
