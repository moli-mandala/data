"""Installed complete chapter; compiled gates remain separate."""
import csv
import importlib.util
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA / 'data/other/forms/raw_data/nair_mullu_kurumba_2013'


def test_installed_audit_preserves_every_witness_and_explicit_reuse():
    rows = list(csv.reader((DATA / 'data/other/forms/20260927-nair-mullu-kurumba.csv').open(newline='')))
    assert rows == list(csv.reader((PACKAGE / 'proposed.csv').open(newline='')))
    assert len(rows) == len({r[10] for r in rows}) == 135
    assert sum(r[0] == 'Mullu Kurumba' for r in rows) == 110
    assert sum(r[0] == 'Malayalam' for r in rows) == 25
    audit = [json.loads(s) for s in (PACKAGE / 'installed-audit.jsonl').read_text().splitlines()]
    raw = [json.loads(s) for s in (PACKAGE / 'proposal-audit.jsonl').read_text().splitlines()]
    assert len(audit) == 65 and [a['source_record'] for a in audit] == raw
    assert all(a['accepted_for_installation'] for a in audit)
    assert sum(len(a['source_record']['witness_dispositions']) for a in audit) == 164
    references = [k for a in audit for k in a['entry_keys']]
    assert len(references) == 141 and set(references) == {r[10] for r in rows}
    assert sum(bool(r[11]) for r in rows) == 11


def test_actual_canonical_parser_metadata_and_guarded_noop():
    path = PACKAGE / 'installation/check_canonical.py'
    spec = importlib.util.spec_from_file_location('mullu_canonical_checks', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.main()
    assert report['parser_monkeypatch'] is False
    assert report['checks']['NFC']['initial_dedupe_keys'] == 135
    assert report['checks']['NFD']['all15fields_exact']
    assert report['all_prior_bibliography_semantics_unchanged']
    assert report['prior_bibliography_entries'] == 630
