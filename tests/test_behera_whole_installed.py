"""Installed Behera whole-report source assets; compiled CLDF remains a separate gate."""
import csv
import hashlib
import io
import json
from pathlib import Path

import make_cldf
import source_meta
from pybtex.database import parse_file

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'data/other/forms/raw_data/sil_korwa_kodaku_2022/whole-recovery'
FORMS = ROOT / 'data/other/forms'
KEY = 'behera2022korwakodaku'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_eight_asset_installation_and_receipt():
    receipt = json.loads((HERE / 'installation-receipt.json').read_text())
    approval = HERE / 'installation/root-acceptance.json'
    accepted = json.loads(approval.read_text())
    assert receipt['approval_sha256'] == sha(approval)
    assert set(receipt['installed_sha256']) == set(accepted['canonical_baselines'])
    assert len(receipt['installed_sha256']) == 8
    assert {name for name, expected in receipt['installed_sha256'].items()
            if sha(ROOT / name) != expected} == set()
    assert receipt['durable_identity_sha256'] == sha(ROOT / 'data/form-identities.csv')
    assert (FORMS / '20260828-sil-korwa-kodaku.csv').read_bytes() == (
        HERE / 'legacy-review/legacy-repaired-candidate.csv').read_bytes()
    assert (FORMS / '20260927-behera-whole-additions.csv').read_bytes() == (
        HERE / 'integration-proposal/all-additions.csv').read_bytes()
    assert (ROOT / 'data/other/forms/raw_data/sil_korwa_kodaku_2022/whole-source-audit.jsonl').read_bytes() == (
        HERE / 'installation/accepted-audit.jsonl').read_bytes()
    assert (ROOT / 'cldf/sources.bib').read_bytes() == (HERE / 'integration-proposal/sources.bib').read_bytes()


def test_installed_rows_parse_and_keep_source_keys():
    rows = []
    for stem, count in (('20260828-sil-korwa-kodaku', 4459),
                        ('20260927-behera-whole-additions', 2405)):
        path = FORMS / f'{stem}.csv'
        with path.open() as stream:
            source_rows = list(csv.reader(stream))
        assert len(source_rows) == count
        errors = io.StringIO()
        parsed, _ = make_cldf.parse_file(str(path), errors, name=stem)
        assert not errors.getvalue() and len(parsed) == count
        assert [row.entry_key for row in parsed] == [row[10] for row in source_rows]
        assert all(row.source.split('[', 1)[0] == KEY for row in parsed)
        rows += parsed
    assert len({row.entry_key for row in rows}) == 6864
    meta = source_meta.load()
    assert {make_cldf.source_entry_dedupe_key(row, meta) for row in rows} == {row.entry_key for row in rows}


def test_registered_dialects_and_references():
    with (ROOT / 'cldf/languages.csv').open() as stream:
        languages = {row['ID'] for row in csv.DictReader(stream)}
    with (ROOT / 'cldf/dialects.csv').open() as stream:
        dialects = {row['Tag']: row['Language_ID'] for row in csv.DictReader(stream)}
    assert 'Birjia' in languages
    for stem in ('20260828-sil-korwa-kodaku', '20260927-behera-whole-additions'):
        with (FORMS / f'{stem}.csv').open() as stream:
            for row in csv.reader(stream):
                assert row[0] in languages
                for tag in row[14].split():
                    if tag.startswith('dialect:'):
                        assert dialects[tag] == row[0]
    bib = parse_file(ROOT / 'cldf/sources.bib')
    assert {KEY, 'abraham1995sadri', 'ctdp2002hillkorwa',
            'powercity_korba_ndc', 'singh1994scheduledtribes'} <= set(bib.entries)


def test_audit_reconciles_every_installed_key_and_preserves_holds():
    audit = [json.loads(line) for line in (
        ROOT / 'data/other/forms/raw_data/sil_korwa_kodaku_2022/whole-source-audit.jsonl').read_text().splitlines()]
    assert len(audit) == len({row['audit_key'] for row in audit}) == 6823
    assert all(row['accepted_for_installation'] is True for row in audit)
    installed = set()
    for stem in ('20260828-sil-korwa-kodaku', '20260927-behera-whole-additions'):
        with (FORMS / f'{stem}.csv').open() as stream:
            installed.update(row[10] for row in csv.reader(stream))
    assert {key for row in audit for key in row['proposed_entry_keys']} == installed
    held = {key for row in audit for key in row.get('held_candidate_keys', [])}
    assert held.isdisjoint(installed) and len(held) == 332
