"""Prove a file-default routing proposal without changing canonical settings."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import yaml

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[5]
sys.path.insert(0, str(DATA))
from source_meta import SourceMeta, importer_commands

KEY = 'behera2022korwakodaku'
old_yaml = DATA / 'data/other/forms/20260828-sil-korwa-kodaku.yaml'
old_csv = old_yaml.with_suffix('.csv')
new_stem = '20260927-behera-whole-additions'
out = HERE / 'routing-proposal'
out.mkdir(exist_ok=True)
old_settings = yaml.safe_load(old_yaml.read_text())
old_rules = old_settings['sources'][KEY].pop('transcription')
old_settings['sources'][KEY].setdefault('identity', {})['dedupe_by_entry_key'] = True
old_settings['sources'][KEY]['reference'] = {
    'editor': 'OpenAI Codex', 'ocr': False, 'etymology_provenance': 'source'}
assert 'transcription' not in old_settings.get('defaults', {})
old_settings.setdefault('defaults', {})['transcription'] = old_rules
importer = {'commands': [[
    'data/other/forms/raw_data/sil_korwa_kodaku_2022/whole-recovery/installation/install_source.py',
    '--install']],
    'note': 'Install or verify the exact accepted whole-report assets through the guarded source transaction.'}
old_settings['defaults']['importer'] = importer
old_proposal = out / old_yaml.name
old_proposal.write_text(yaml.safe_dump(old_settings, allow_unicode=True, sort_keys=False))
new_proposal = out / f'{new_stem}.yaml'
new_proposal.write_text(yaml.safe_dump({
    'file': f'{new_stem}.csv',
    'defaults': {'transcription': {'convert': False, 'preserve_hyphens': True},
                 'importer': importer},
}, sort_keys=False))
before = SourceMeta([old_yaml])
after = SourceMeta([old_proposal, new_proposal])
rows = list(csv.reader(old_csv.open()))
assert len(rows) == 4458 and all(len(r) == 15 for r in rows)
for row in rows:
    assert before.transcription_rule(KEY, old_csv, row[0]) == after.transcription_rule(KEY, old_csv, row[0])
    assert before.source(KEY).get('forms') == after.source(KEY).get('forms')
    assert before.source(KEY).get('notes') == after.source(KEY).get('notes')
assert after.flag(KEY, 'reference', 'editor') == 'OpenAI Codex'
assert after.flag(KEY, 'reference', 'ocr') is False
assert after.flag(KEY, 'reference', 'etymology_provenance') == 'source'
for language in ('kw', 'Kodaku', 'Sadri', 'Birjia', 'mu', 'Asuri'):
    assert after.transcription(KEY, f'{new_stem}.csv', language) == (None, False)
assert list(after.sources) == [KEY]
assert importer_commands(after, old_yaml.stem) == importer['commands']
assert importer_commands(after, new_stem) == importer['commands']
report = {
    'status': 'Routing mechanism verified in isolation; transcription decision not accepted',
    'old_rows_with_identical_transcription_rule': len(rows),
    'source_key_declared_once': True,
    'old_form_and_note_settings_preserved': True,
    'new_file_literal_route': True,
    'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (old_csv, old_yaml)},
    'proposal_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (old_proposal, new_proposal)},
    'not_proven': ['Final symbol coverage', 'Final corpus field/parser behavior',
                  'Accepted sound-profile decision', 'Canonical installation'],
}
(out / 'routing-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
