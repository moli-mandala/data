"""Scoped installed-source checks; no CLDF/database generation."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import pybtex
from pybtex.database import parse_file

HERE = Path(__file__).resolve().parent
DATA = next(p for p in HERE.parents if (p / 'tags.py').exists())
sys.path.insert(0, str(DATA))
import make_cldf
from source_meta import SourceMeta

canonical = DATA / 'data/other/forms/20260927-nair-betta-kurumba.csv'
settings = canonical.with_suffix('.yaml')
receipt = json.loads((HERE / 'installation-receipt.json').read_text())
raw = list(csv.reader(canonical.open()))
assert canonical.read_bytes() == (HERE / 'proposed.csv').read_bytes()
assert settings.read_bytes() == (HERE / settings.name).read_bytes()
errors = []
parsed, stats = make_cldf.parse_file(str(canonical), errors)
assert len(parsed) == len(raw) == 206 and not errors and stats['converted'] == 0
by = {r[10]: r for r in raw}
assert {r.entry_key for r in parsed} == set(by)
for row in parsed:
    r = by[row.entry_key]
    assert row.form == row.old_form == r[2] and not row.ipa
    assert row.variant_of_key == r[11]
meta = SourceMeta([settings])
assert meta.sources['nair2013bettakurumba']['reference']['etymology_provenance'] == 'none'
registered = parse_file(str(DATA / 'cldf/sources.bib')).entries
proposed = parse_file(str(HERE / 'proposed-reference.bib')).entries
formatted = {}
for key, entry in proposed.items():
    assert registered[key] == entry
    formatted[key] = pybtex.PybtexEngine().format_from_string(entry.to_string('bibtex'), 'plain', output_backend='markdown')[3:].strip()
    assert formatted[key]
languages = list(csv.DictReader((DATA / 'cldf/languages.csv').open()))
assert sum(r['ID'] == 'BettaKurumba' and r['Glottocode'] == 'bett1235' for r in languages) == 1
identity_sha = hashlib.sha256((DATA / 'data/form-identities.csv').read_bytes()).hexdigest()
assert identity_sha == receipt['durable_identity_sha256']
audit = [json.loads(s) for s in (HERE / 'installed-audit.jsonl').read_text().splitlines()]
assert len(audit) == 225 and all(a['accepted_for_source_stage'] for a in audit)
assert {k for a in audit for k in a['proposed_entry_keys']} == set(by)
assert sum(a['disposition'] == 'audit-only' for a in audit) == 22
report = {'status': 'passed', 'rows': 206, 'physical_units': 225, 'audit_only': 22, 'variant_edges': 3,
          'csv_sha256': hashlib.sha256(canonical.read_bytes()).hexdigest(),
          'audit_sha256': hashlib.sha256((HERE / 'installed-audit.jsonl').read_bytes()).hexdigest(),
          'durable_identity_sha256': identity_sha, 'formatted_references': formatted,
          'parsed_errors': errors, 'converted': stats['converted'],
          'deferred': ['full CLDF/DB build', 'compiled identities/graphs/references', 'full suite', 'browser QA']}
(HERE / 'installed-validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
