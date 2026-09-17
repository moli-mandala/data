import csv
import importlib.util
import io
import sys
import unicodedata
from pathlib import Path

import pytest
from segments import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'data/other/forms/raw_data/yoshioka_cleanup.py'
spec = importlib.util.spec_from_file_location('yoshioka_cleanup_tests', SCRIPT)
cleanup = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cleanup
spec.loader.exec_module(cleanup)


@pytest.fixture(scope='module')
def parsed():
    records = cleanup.entry_records(cleanup.load_snapshot())
    rows, audits = cleanup.compile_records(records)
    return records, rows, audits


def test_snapshot_accounts_for_every_original_key_and_page(parsed):
    _, rows, audits = parsed
    assert len(audits) == 3267
    assert {a['Entry_Key'] for a in audits if a['Entry_Key'].startswith('yoshioka-entry-')} == {
        f'yoshioka-entry-{i}' for i in range(1, 3213)
    }
    assert len({a['Entry_Key'] for a in audits}) == len(audits)
    assert {int(a['PDF_Page']) for a in audits} == set(range(505, 619))
    assert sum(a['Status'] == 'excluded' for a in audits) == 9
    assert all(a['Review'] for a in audits if a['Status'] != 'installed')
    assert len({r[10] for r in rows}) == len(rows)
    assert all(len(r) == 15 and r[0] == 'Bur' and r[2] for r in rows)
    assert all('�' not in ''.join(r) and all(unicodedata.is_normalized('NFC', c) for c in r) for r in rows)


def test_plural_labels_apply_to_plural_forms_and_keep_their_noun_class(parsed):
    by = {r[10]: r for r in parsed[1]}
    assert by['yoshioka-entry-3'][2:4] == ['aalú', 'potato']
    assert 'pl' not in by['yoshioka-entry-3'][14].split()
    plural = by['yoshioka-entry-3:inflection:1']
    assert plural[2] == 'aaloínc'
    assert {'noun', 'Burushaski-class-X', 'pl'} <= set(plural[14].split())
    assert plural[11] == 'yoshioka-entry-3'


def test_pronoun_paradigm_and_verbal_argument_codes_do_not_make_headwords_nouns(parsed):
    by = {r[10]: r for r in parsed[1]}
    pron = by['yoshioka-entry-43']
    assert pron[3] == 'so-and-so, something'
    assert set(pron[14].split()) == {'pron', 'dialect:Bur:Yoshioka-EB:Eastern%20Burushaski'}
    assert any(r[2] == 'aléstiŋ' and {'pron', 'pl', 'Burushaski-class-H', 'Burushaski-class-X'} <= set(r[14].split())
               for r in parsed[1] if r[11] == pron[10])
    verb = by['yoshioka-entry-565']
    assert verb[3] == 'give' and 'noun' not in verb[14].split()
    assert 'Y.SG.OBJ' in verb[6] and 'Y.SG.OBJ' not in verb[3]
    assert 'pl' not in by['yoshioka-entry-385'][14].split()
    assert any(r[2] == 'bum' and {'pfv', 'participle'} <= set(r[14].split()) for r in parsed[1])


def test_dialectal_head_variant_does_not_inherit_preceding_plural_suffix_tag(parsed):
    by = {r[10]: r for r in parsed[1]}
    dialectal = [r for r in parsed[1] if r[11] == 'yoshioka-entry-2534' and r[2] == 'ṣinc̣']
    assert len(dialectal) == 1
    assert 'dialect:Bur:Yoshioka-GA:Ganish' in dialectal[0][14].split()
    assert 'pl' not in dialectal[0][14].split()
    for key in ['yoshioka-entry-89', 'yoshioka-entry-874']:
        assert {'sg', 'pl'} <= set(by[key][14].split())
    assert 'dialect:Bur:Yoshioka-NG:Nager' not in by['yoshioka-entry-1282'][14].split()


def test_source_grammar_and_class_specific_plural_analysis_are_preserved(parsed):
    by = {r[10]: r for r in parsed[1]}
    for audit in parsed[2]:
        if audit['Status'] == 'installed' and audit['Morphology']:
            for key in audit['Emitted_Keys'].split('|'):
                assert 'Source grammar: ' + audit['Morphology'] in by[key][6]
    assert 'pl' in by['yoshioka-entry-1568'][14].split()  # children: H PL
    assert {'sg', 'pl'} <= set(by['yoshioka-entry-38'][14].split())
    assert {'sg', 'Burushaski-class-X'} <= set(by['yoshioka-entry-3049'][14].split())
    plural = by['yoshioka-entry-3049:inflection:1']
    assert {'pl', 'Burushaski-class-Y'} <= set(plural[14].split())
    assert not {'sg', 'Burushaski-class-X'} & set(plural[14].split())
    assert 'prox' in by['yoshioka-entry-1731'][14].split()
    assert 'Burushaski-class-Z' in by['yoshioka-entry-930'][14].split()
    assert by['yoshioka-entry-577'][3] == 'bunch (of grapes), head (of wheat, barley)'
    second = next(r for r in parsed[1] if r[2] == 'čhu' and r[3].startswith('head (of polostick'))
    assert 'Burushaski-class-Y' in second[14].split() and 'PL -míŋ' in second[6]
    assert by['yoshioka-entry-840'][3].startswith('X bowl')
    stem = next(r for r in parsed[1] if r[11] == 'yoshioka:p258:y201308'
                and r[2] == 'd-@-̈phirkan-')
    assert 'dialect:Bur:Yoshioka-NG:Nager' in stem[14].split()
    assert 'ipfv' not in stem[14].split()
    assert {'sg', 'pl'} <= set(by['yoshioka-entry-278'][14].split())
    assert 'pl' in by['yoshioka-entry-117'][14].split()
    assert 'double-plural' in by['yoshioka-entry-182:inflection:1'][14].split()
    assert by['yoshioka-entry-1088'][3] == 'trousers, slacks, breeches'
    assert any(r[2] == 'gurpáltiŋ' for r in parsed[1] if r[11] == 'yoshioka-entry-1088')


def test_source_tags_and_bibliographic_keys_are_registered(parsed):
    import pybtex.database
    from make_refs import create_short_ref
    with (ROOT / 'cldf/dialects.csv').open() as stream:
        dialects = {r['Tag']: r for r in csv.DictReader(stream)}
    bib = pybtex.database.parse_file(str(ROOT / 'cldf/sources.bib')).entries
    assert create_short_ref(bib['ilcaa1967']) == 'AA1967'
    assert create_short_ref(bib['yoshioka2012']) == 'Y2012'
    for row in parsed[1]:
        for tag in row[14].split():
            if tag.startswith('dialect:'):
                assert tag.startswith('dialect:Bur:')
                assert dialects[tag]['Language_ID'] == 'Bur'
        assert all(c.split('[', 1)[0] in bib for c in row[7].split(';'))


def test_generated_review_uses_yoshioka_evidence_and_reports_pending_full_build():
    import audit_source_ingestions as audit
    uid = '20260726-yoshioka-eastern-burushaski'
    path = ROOT / 'data/other/forms' / (uid + '.csv')
    importers, audits, tests = audit.infer_related_files(path, uid)
    assert all('yoshioka' in p for p in importers)
    assert any(p.endswith('yoshioka_2026/audit.csv') for p in audits)
    assert all('magar' not in p and 'gujari' not in p for p in importers + audits + tests)
    assert audit.UNIT_PRIMARY_SOURCES[uid] == {'yoshioka2012'}
    assert audit.UNIT_EVIDENCE_OVERRIDES[uid]['12. Install and run the full data pipeline'][0] is False


def test_complete_reference_notes_and_true_crossreferences_survive(parsed):
    by = {r[10]: r for r in parsed[1]}
    assert 'berger[p. 10, abáat, cited by Yoshioka]' in by['yoshioka-entry-2'][7]
    assert 'ilcaa1967[item 520, cited by Yoshioka]' in by['yoshioka-entry-3'][7]
    assert by['yoshioka-entry-666'][2] == 'd-@-́c-'
    assert by['yoshioka-entry-666'][11] == 'yoshioka-entry-2574'
    assert by['yoshioka-entry-666'][3] == by['yoshioka-entry-2574'][3]
    assert all(not r[3].startswith('see ') for r in parsed[1])
    assert all(a['Reference_Note'] in a['Raw_Text'] or 'shared-sense-reference' in a['Review']
               for a in parsed[2] if a['Reference_Note'])
    assert all('uncertain' in r[14].split() and 'See ' in r[6]
               for r in parsed[1] if not r[3])


def test_affricates_and_recovered_font_symbols_remain_distinct(parsed):
    profile = Tokenizer(str(ROOT / 'conversion/yoshioka.txt'))
    assert profile('c č c̣ ch čh c̣h', column='IPA').split() == ['ʦ', '#', 'c', '#', 'ʦ̣', '#', 'ʦʰ', '#', 'cʰ', '#', 'ʦ̣ʰ']
    assert all('�' not in profile(r[2], column='IPA') for r in parsed[1])


def test_existing_cognate_evidence_keys_and_all_variant_targets_survive(parsed):
    keys = {r[10] for r in parsed[1]}
    assert all(not r[11] or r[11] in keys for r in parsed[1])
    assert not any(r[1] or r[12] for r in parsed[1])
    with (ROOT / 'data/burushaski_cognates.csv').open() as stream:
        needed = {k for row in csv.DictReader(stream) for k in row['Evidence_Keys'].split('|')
                  if k.startswith('yoshioka-')}
    assert needed <= keys


def test_actual_source_build_preserves_rows_transcription_and_durable_ids(parsed, tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    import make_cldf
    from assign_form_ids import assign_ids
    from source_key_aliases import apply_source_key_aliases
    monkeypatch.setattr(make_cldf, 'tqdm', lambda iterable, **kw: iterable)
    directory = tmp_path / 'other/forms'
    directory.mkdir(parents=True)
    path = directory / '20260726-yoshioka-eastern-burushaski.csv'
    with path.open('w', newline='') as stream:
        csv.writer(stream).writerows(parsed[1])
    errors = io.StringIO()
    compiled, _ = make_cldf.parse_file(str(path), errors, name='yoshioka', file_num='yoshioka-check')
    assert errors.getvalue() == ''
    assert len(compiled) == len(parsed[1])
    assert {r.entry_key for r in compiled} == {r[10] for r in parsed[1]}
    assert next(r for r in compiled if r.entry_key == 'yoshioka-entry-3').old_form == 'aalú'
    from tags import extract_tags
    from form_note_policy import apply_form_note_policy
    expected_notes = {r[10]: r[6] for r in parsed[1]}
    for row in compiled:
        assert row.notes == expected_notes[row.entry_key]
        _, public_notes = extract_tags(row.notes, language_id='Bur')
        assert public_notes == row.notes
        assert apply_form_note_policy(public_notes, row.source, row.etymology)[0] == row.notes
    forms = [{'ID': r.id, 'Language_ID': r.lang, 'Form': r.form, 'Original': r.old_form,
              'Gloss': r.gloss, 'Source': r.source, 'Status': 'unlinked', 'Redirect': ''} for r in compiled]
    with (ROOT / 'data/form-identities.csv').open() as stream:
        previous = [r for r in csv.DictReader(stream) if r['Source_Key'].startswith('yoshioka-')]
    source_keys = {r.id: r.entry_key for r in compiled}
    mapping, registry = assign_ids(forms, previous, source_keys)
    old_active = {r['Source_Key']: r['Form_ID'] for r in previous if r['Status'] == 'active'}
    new_active = {r['Source_Key']: r['Form_ID'] for r in registry if r['Status'] == 'active'}
    assert all(new_active[k] == old_active[k] for k in old_active.keys() & new_active.keys())
    cleanup.write_outputs(tmp_path / 'output', parsed[1], parsed[2])
    aliases = {}
    apply_source_key_aliases(aliases, previous, registry, set(mapping.values()), [tmp_path / 'output/form-aliases.csv'])
    retired = old_active.keys() - new_active.keys()
    assert all(old_active[k] in aliases for k in retired)


def test_aliases_reject_active_collisions_and_retarget_existing_urls(tmp_path):
    from source_key_aliases import apply_source_key_aliases
    path = tmp_path / 'aliases.csv'
    path.write_text('Retired_Source_Key,Target_Source_Key,Reason\nold,new,printed continuation\n')
    previous = [{'Source_Key': 'old', 'Form_ID': 'f_old'}]
    current = [{'Source_Key': 'new', 'Form_ID': 'f_new', 'Status': 'active'}]
    aliases = {'legacy-url': 'f_old'}
    assert apply_source_key_aliases(aliases, previous, current, {'f_new'}, [path]) == 1
    assert aliases == {'legacy-url': 'f_new', 'f_old': 'f_new'}
    with pytest.raises(ValueError, match='active ID'):
        apply_source_key_aliases({}, previous, current, {'f_old', 'f_new'}, [path])
