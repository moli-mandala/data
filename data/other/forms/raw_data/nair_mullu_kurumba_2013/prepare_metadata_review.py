"""Source-local identity/Unicode/reference policy, no canonical writes."""
import collections
import csv
import hashlib
import json
import unicodedata
from pathlib import Path
from pybtex.database import parse_file

HERE = Path(__file__).resolve().parent
DATA = next(p for p in HERE.parents if (p / 'source_meta.py').exists())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows = list(csv.reader((HERE / 'proposed.csv').open(newline='')))
first = [json.loads(s) for s in (HERE / 'physical-units-first-reading.jsonl').read_text().splitlines()]
languages = list(csv.DictReader((DATA / 'cldf/languages.csv').open(newline='')))
routes = {k: next(r for r in languages if r['ID'] == k) for k in ('Mullu Kurumba', 'Malayalam')}
assert routes['Mullu Kurumba']['Glottocode'] == 'mull1244'
bib = parse_file(str(HERE / 'proposed-references.bib'))
assert set(bib.entries) == {'nair2013mullukurumba', 'robertsathyajoseph1982mullukurumba'}
assert bib.entries['nair2013mullukurumba'].fields['ocr'] == 'No'
symbols = collections.Counter(c for r in rows for c in r[2])
assert not any(0xE000 <= ord(c) <= 0xF8FF for c in symbols)
section_inventory = {}
for unit in first:
    chars = collections.Counter(c for w in unit['source_witnesses'] for c in w['literal_first_reading'])
    section_inventory[unit['source_unit_key']] = {f'U+{ord(c):04X}': n for c, n in sorted(chars.items())}
report = dict(status='Source-local complete metadata/profile proposal; independent acceptance pending',
    inputs={name: sha(HERE/name) for name in ('physical-units-first-reading.jsonl', 'proposed.csv', 'proposal-audit.jsonl', 'proposed-settings.yaml', 'proposed-references.bib')},
    source_routes=routes, registry_action='Reuse existing two base languages; no dialect or coordinate additions.',
    lect_scope='Nair discusses Wayanad/Nilgiris and quotes the reported Nilgiris study but does not locate each printed example. No new source lect is inferred.',
    existing_source_overlap='Jeba2018 numeral witnesses remain distinct; no numeral forms are present here.',
    main_source='nair2013mullukurumba', auxiliary='robertsathyajoseph1982mullukurumba',
    reference_scope='Auxiliary is cited background only, never a directly collated lexical citation. All lexical witnesses cite Nair printed pages.',
    profile=dict(convert=False, preserve_hyphens=True, Native='blank', Phonemic='blank',
        display_policy='Exact literal Roman spelling; no invented IPA or normalization of drawn lowline into a phonemic consonant.',
        symbol_inventory=[dict(character=c, codepoint=f'U+{ord(c):04X}', name=unicodedata.name(c,'unnamed'), occurrences=n) for c,n in sorted(symbols.items())],
        section_inventory=section_inventory,
        notation={'U+0323': 'Separate round dot in original Symbol font; literal underdot, no IPA claim.',
            'U+0303': 'Visible original upper tilde, literal nasal-shaped sign, no automatic phonemic claim.',
            'U+0332': 'Separate drawn lower horizontal line beneath l on p55 and n on p56, distinct from round underdot.',
            'U+2013': 'Printed en dash in several bound forms, retained distinctly from hyphen-minus.'},
        PUA_remaining=0, dedicated_transcription_profile='Not applicable with source-level convert:false; Unicode inventory and parser retention are tested instead.'),
    pending=['Full independent original glyph/scope reading', 'fresh actual-output audit and graph review', 'root guarded installation approval', 'coordinated compiled/full-build/browser gates'],
    canonical_changes=False)
(HERE / 'metadata-profile-review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'reference_records': 2, 'registered_routes': sorted(routes), 'PUA': 0, 'canonical_changes': False}))
