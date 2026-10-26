"""Assemble review-only comparison routing and registry drafts; never install."""
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

candidate_path = HERE / 'comparisons/comparison-reviewed-candidate.jsonl'
routes_path = HERE / 'identity-review/all-source-routing-proposal.json'
language_path = DATA / 'cldf/languages.csv'
dialect_path = DATA / 'cldf/dialects.csv'
candidates = [json.loads(s) for s in candidate_path.read_text().splitlines()]
identity = json.loads(routes_path.read_text())
routes = {r['source_code']: r for r in identity['wordlists']}
languages = {r['ID']: r for r in csv.DictReader(language_path.open())}
existing = {r['ID']: r for r in csv.DictReader(dialect_path.open())}
new_language = identity['proposed_language_registry_row']
assert new_language['ID'] not in languages
languages[new_language['ID']] = new_language
assert len(candidates) == 1616
assert len({r['source_unit_key'] for r in candidates}) == len(candidates)
out = HERE / 'integration-proposal'
out.mkdir(exist_ok=True)
drafts = {}
rows = []
audit = []
grammar_glosses = {
    202: ('I (1st sg)', 'I', {'pron', '1sg', 'sg'}),
    203: ('you (2nd sg, informal)', 'you', {'pron', '2sg', 'sg', 'informal'}),
    204: ('you (2nd sg, formal)', 'you', {'pron', '2sg', 'sg', 'formal'}),
    205: ('he (3rd sg, masculine)', 'he', {'pron', '3sg', 'sg', 'm'}),
    206: ('she (3rd sg, feminine)', 'she', {'pron', '3sg', 'sg', 'f'}),
    207: ('we (1st pl, inclusive)', 'we', {'pron', '1pl', 'pl', 'inclusive'}),
    208: ('we (1st pl, exclusive)', 'we', {'pron', '1pl', 'pl', 'exclusive'}),
    209: ('you (2nd pl)', 'you', {'pron', '2pl', 'pl'}),
    210: ('they (3rd pl)', 'they', {'pron', '3pl', 'pl'}),
}
for candidate in candidates:
    code = candidate['source_code']
    route = routes[code]
    language = route['language']
    record = dict(candidate)
    record['routing_basis'] = route['basis']
    record['language_id'] = language
    record['accepted_for_installation'] = False
    gloss = candidate['gloss']
    if candidate['item'] in grammar_glosses:
        printed, lexical, required = grammar_glosses[candidate['item']]
        assert gloss == printed and required <= set(candidate['tags'])
        gloss = lexical
        record['gloss_field_decision'] = {
            'printed': printed, 'lexical': lexical,
            'grammar_retained_in_tags': sorted(required),
        }
    if language is None:
        assert code == 'T'
        record['integration_disposition'] = 'held-language-identity-conflict'
        audit.append(record)
        continue
    assert language in languages
    dialect_id = route['dialect_id']
    tag = dialect_tag(language, dialect_id, route['display'])
    draft = {
        'ID': dialect_id, 'Tag': tag, 'Language_ID': language,
        'Source_Language_ID': code, 'Name': route['display'],
        'Glottocode': '', 'Latitude': '', 'Longitude': '',
        'Clade': languages[language]['Clade'], 'Location': route['locality'],
        'Quality': '',
    }
    # No coordinate estimate is made; a quality grade would imply one.
    assert dialect_id not in existing
    if dialect_id in drafts:
        assert drafts[dialect_id] == draft
    drafts[dialect_id] = draft
    tags = list(candidate['tags']) + [tag]
    if candidate['typed_uncertainties'] and 'uncertain' not in tags:
        tags.append('uncertain')
    citations = ';'.join(
        f'behera2022korwakodaku[Appendix B.5, p. {page}, item {candidate["item"]}, site {code}]'
        for page in candidate['printed_pages'].split(';'))
    row = [language, '', candidate['form'], gloss, '', '', '',
           citations, '', '', candidate['source_unit_key'], '', '', '', ' '.join(tags)]
    assert len(row) == 15 and row[2]
    rows.append(row)
    record.update(integration_disposition='routed-review-candidate',
                  dialect_id=dialect_id, dialect_tag=tag, proposed_csv_row=row)
    audit.append(record)

def csv_text(records):
    buffer = io.StringIO(newline='')
    csv.writer(buffer).writerows(records)
    return buffer.getvalue()

(out / 'comparison-rows.csv').write_text(csv_text(rows), newline='')
fields = list(next(iter(existing.values())))
(out / 'comparison-dialects.csv').write_text(
    csv_text([fields] + [[r[f] for f in fields] for r in drafts.values()]), newline='')
(out / 'comparison-routing-audit.jsonl').write_text(
    ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in audit))
report = {
    'status': 'Review-only assembly; independent literal and identity acceptance pending',
    'input_sha256': {str(p.relative_to(DATA)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (candidate_path, routes_path, language_path, dialect_path)},
    'all_candidate_records': len(candidates), 'routed_rows': len(rows),
    'by_language': dict(Counter(r[0] for r in rows)),
    'identity_held_rows': len(candidates) - len(rows),
    'dialect_drafts': len(drafts),
    'uncertain_rows': sum('uncertain' in r[14].split() for r in rows),
    'grammar_glosses_cleaned': sum('gloss_field_decision' in r for r in audit),
    'canonical_changes': False,
    'pending': ['Independent correction/sample review', 'Birjia registry acceptance',
                'Final transcription/symbol review', 'Actual importer parser checks',
                'Combined whole-report audit and installation'],
}
(out / 'comparison-routing-status.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
