"""Acquisition and font-decoding checks, not an ingestion acceptance audit."""
import hashlib
import importlib.util
import json
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / 'data/other/forms/raw_data/dhakal_darai_2011'


def decoder():
    spec = importlib.util.spec_from_file_location('dhakal_decode', RAW / 'decode.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_chapter_evidence_and_font_coverage():
    meta = json.loads((RAW / 'evidence/snapshot.json').read_text())
    assert meta['pdf_pages'] == 486
    assert [p['printed_page'] for p in meta['pages']] == list(range(42, 77))
    assert sum(p['glyph_count'] for p in meta['pages']) == 40425
    decode = decoder()
    for page in meta['pages']:
        for name, digest in page['files'].items():
            assert hashlib.sha256((RAW / 'evidence' / name).read_bytes()).hexdigest() == digest
        chars = json.loads((RAW / 'evidence' / f"p{page['printed_page']:03}-glyphs.json").read_text())
        assert len(chars) == page['glyph_count']
        assert decode.content_order_text(chars)


def test_content_order_preserves_nasal_attachment():
    decode = decoder()
    chars = json.loads((RAW / 'evidence/p046-glyphs.json').read_text())
    text = decode.content_order_text(chars)
    # Visually checked against PDF p.68: positional text extraction gets these wrong.
    assert '/ãkʰi/' in text and '/hõco/' in text
    assert '/ak̃' not in text and '/hoc̃' not in text
    assert '/ciũṭa/' in text and '/dzʰə̃krija/' in text


def test_font_mapping_is_scoped_and_rejects_unknown_symbols():
    import pytest
    decode = decoder()
    assert decode.decode_char({'text': '\uf04e', 'fontname': 'SILDoulosIPA93Regular'}) == 'ŋ'
    with pytest.raises(ValueError):
        decode.decode_char({'text': '\uf04e', 'fontname': 'UnrelatedFont'})


def inventory_module():
    import sys
    spec = importlib.util.spec_from_file_location('dhakal_inventory', RAW / 'inventory.py')
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(RAW))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_quote_inventory_reproduces_and_preserves_mixed_delimiters():
    from collections import Counter
    module = inventory_module()
    records = module.build()
    assert records == json.loads((RAW / 'inventory.json').read_text())
    assert len({r['key'] for r in records}) == len(records) == 495
    assert Counter(r['kind'] for r in records) == {
        'slash-delimited-form': 444, 'needs-context-review': 51}
    assert all(r['status'] == 'unreviewed' for r in records)
    # Quotes inside possessives/contractions must not split glosses.
    assert any(r['gloss_candidate'] == 'husband’s younger sister' for r in records)
    assert any(r['gloss_candidate'] == "Don't speak much." for r in records)
    assert any(r['gloss_candidate'] == 'ONO, the manner one walks' for r in records)
    assert any(r['gloss_candidate'] == 'cord for tethering beast' for r in records)


def test_inventory_keeps_repeated_attestations_and_conflicting_source_glosses():
    rows = inventory_module().build()
    bird = [r for r in rows if r['form_candidate'] == 'ɡeutʰəli']
    assert {(r['printed_page'], r['gloss_candidate']) for r in bird} == {
        (47, 'skylark'), (53, 'sparrow')}
    assert len({r['key'] for r in bird}) == 2
    prose = [r for r in rows if r['gloss_candidate'] == 'cluster']
    assert len(prose) == 1 and prose[0]['kind'] == 'needs-context-review'
    assert not prose[0]['form_candidate']


def test_every_context_case_has_an_explicit_disposition():
    from collections import Counter
    rows = inventory_module().build()
    dispositions = json.loads((RAW / 'context-dispositions.json').read_text())
    assert {r['key'] for r in dispositions} == {
        r['key'] for r in rows if r['kind'] == 'needs-context-review'}
    assert Counter(r['status'] for r in dispositions) == {
        'lexical-candidate': 34, 'excluded-sentence': 16, 'excluded': 1}
    assert next(r for r in dispositions if r['key'] == 'dhakal2011:p73:g452')['form'] == 'ro-'
    exception = json.loads((RAW / 'inventory-exceptions.json').read_text())[0]
    assert exception['form'] == 'caralə' and exception['source_gloss'] == 'graze-PST'
    chars = json.loads((RAW / 'evidence/p043-glyphs.json').read_text())
    assert exception['anchor'] in decoder().content_order_text(chars)


def test_proposal_accounts_for_every_span_and_extra_record():
    import sys
    spec = importlib.util.spec_from_file_location('dhakal_proposal', RAW / 'proposal.py')
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(RAW))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    proposal = module.build()
    assert proposal == json.loads((RAW / 'proposal.json').read_text())
    assert len(proposal['candidates']) == 485 and len(proposal['excluded']) == 17
    index = {r['key']: r for r in proposal['candidates']}
    assert sum(bool(r.get('variant_of_key')) for r in index.values()) == 4
    assert sum('loanword' in r['tags'] for r in index.values()) == 2
    assert index['dhakal2011:p74:g621']['form'] == 'naimarəi'
    assert index['dhakal2011:p74:g697']['tags'] == ['neg', '1pl']
    for relation in json.loads((RAW / 'reviewed-relations.json').read_text()):
        assert relation['child'] in index and relation['parent'] in index
        assert relation['child'] != relation['parent']
    originals = {r['key'] for r in json.loads((RAW / 'inventory.json').read_text())}
    assert originals <= set(index) | {r['key'] for r in proposal['excluded']}


def test_grammar_separates_source_labels_without_mangling_definitions():
    spec = importlib.util.spec_from_file_location('dhakal_grammar', RAW / 'grammar.py')
    grammar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grammar)
    assert grammar.parse('graze.IMP', 'cər') == ('graze', ['impv'])
    assert grammar.parse('cut. ABS', 'kaṭi') == ('cut', ['abs'])
    assert grammar.parse('you.HH', 'pau') == ('you', ['honorific', 'high-honorific'])
    assert grammar.parse('buy. PROS', 'kin.la.rə') == ('buy', ['prospective'])
    assert grammar.parse('one-CLF', 'ek.ṭa') == ('one', ['classifier'])
    assert grammar.parse('PROS', '-larʰə') == ('', ['prospective', 'suffix'])
    for gloss in ['daughter-in-law', 'rest-house', 'Kumal (an ethnic group)',
                  'half moon that usually occurs in the month of Bhadra']:
        assert grammar.parse(gloss, 'example') == (gloss, [])


def test_proposal_grammar_and_corrected_dots_use_registered_tags():
    import tags
    proposal = json.loads((RAW / 'proposal.json').read_text())
    index = {r['key']: r for r in proposal['candidates']}
    assert index['dhakal2011:p59:g201']['form'] == 'cʰəṭ.pə.ṭi'
    assert index['dhakal2011:p59:g201']['extracted_form'] == 'cʰət.̣pə.ṭi'
    assert index['dhakal2011:p68:g503']['form'] == 'pəṭ.ka'
    assert not any('.̣' in r['form'] for r in index.values())
    frontend = (RAW.parents[4].parent / 'jambu-static/src/lib/tags.ts').read_text()
    for tag in ['prospective', 'non-past', 'classifier', 'high-honorific']:
        assert tag in tags.GRAMMATICAL_TAGS
        assert f"'{tag}'" in frontend


def test_transcription_preserves_source_contrasts_and_literal_originals():
    import csv
    import unicodedata
    from segments.tokenizer import Tokenizer
    from profile_policy import house_output
    spec = importlib.util.spec_from_file_location('dhakal_transcription', RAW / 'transcription.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.build()
    with (RAW / 'transcription-preview.csv').open() as stream:
        assert [{**r, 'Page': str(r['Page'])} for r in rows] == list(csv.DictReader(stream))
    assert len(rows) == 485
    index = {r['Source_Key']: r for r in rows}
    assert index['dhakal2011:p61:g177']['Original'] == 'ɡen ṭʰa'
    assert index['dhakal2011:p61:g177']['Form'] == 'genṭʰa'
    assert index['dhakal2011:p58:g806']['Form'] == 'kapṭike'
    assert index['dhakal2011:p58:g59']['Form'] == 'ṭʰyai ṭʰyai'
    tokenizer = Tokenizer(str(module.PROFILE))
    convert = lambda s: unicodedata.normalize('NFC', tokenizer(s, column='IPA').replace(' ', ''))
    assert convert('ə a c dz j w ɡ'.replace(' ', '')) == 'əaʦʣyvg'
    assert convert('ãkʰi') == convert(unicodedata.normalize('NFD', 'ãkʰi')) == 'ãkʰi'
    assert convert('-tə') == '-tə'
    assert all('ā' not in r['Form'] for r in rows)
    with module.PROFILE.open() as stream:
        rules = {r['Grapheme']: r['IPA'] for r in csv.DictReader(stream, delimiter='\t')}
    for grapheme, output in rules.items():
        assert house_output(grapheme, output, rules, 'dhakal-darai') == output


def test_historical_chapter3_baseline_preserves_accepted_evidence():
    import csv
    import sys
    from collections import Counter
    spec=importlib.util.spec_from_file_location('dhakal_legacy_import', RAW/'import_chapter3_legacy.py')
    module=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(RAW))
    try:
        spec.loader.exec_module(module)
        expected,audit=module.build()
    finally:
        sys.path.pop(0)
    path=RAW/'legacy-before-whole-recovery.csv'
    with path.open() as stream:
        assert list(csv.reader(stream))==expected
    assert len(expected)==485 and len(audit)==502
    assert Counter(a['status'] for a in audit)=={'proposed':485,'excluded':17}
    result=json.loads((RAW/'acceptance-results.json').read_text())
    assert result['material_errors']==0 and len(result['review'])==20
    assert hashlib.sha256(path.read_bytes()).hexdigest()==result['sample']['csv_sha256']
    for relative,digest in result['input_hashes'].items():
        target=RAW/'legacy-before-whole-recovery-profile.txt' if relative=='conversion/dhakal-darai.txt' else module.ROOT/relative
        assert hashlib.sha256(target.read_bytes()).hexdigest()==digest


def test_installed_whole_darai_exact_output_actual_parser_and_local_graph():
    import csv
    import io
    import sys
    from make_cldf import parse_file
    spec=importlib.util.spec_from_file_location('dhakal_whole_import',RAW/'import_source.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    expected,audit=module.build()
    assert len(expected)==8373 and len(audit)==11304
    rows={row[10]:row for row in expected}
    parsed_all=[]
    for route,stem in module.ROUTES.items():
        path=module.ROOT/f'data/other/forms/{stem}.csv'
        assert path.read_bytes()==(RAW/f'whole-proposed-{route}.csv').read_bytes()
        errors=io.StringIO();parsed,stats=parse_file(str(path),errors=errors)
        assert not errors.getvalue()
        parsed_all.extend(parsed)
        for row in parsed:
            original=rows[row.entry_key]
            assert row.old_form==original[2] and row.ipa==original[5]
            assert row.source==original[7]
            assert set(original[14].split())<=set(row.tags.split())
            assert row.notes==original[6]
            assert not row.native and not row.param
            assert '\ufffd' not in row.form
            if route=='literal-preservation':assert row.form==original[2] and not row.ipa
    assert {r.entry_key for r in parsed_all}==rows.keys()
    with (RAW/'legacy-before-whole-recovery.csv').open() as stream:
        old=list(csv.reader(stream))
    assert len(old)==485
    for row in old:
        assert all(rows[row[10]][i]==row[i] for i in [2,5,8,9,10,11,12,13])
    assert sum(bool(r[11]) for r in expected)==12
    assert sum(bool(r[13]) for r in expected)==18
    assert all(parent in rows for row in expected for col in [11,12,13] for parent in row[col].split('|') if parent)
    assert (RAW/'whole-installed-audit.jsonl').read_bytes()==(RAW/'whole-proposed-audit.jsonl').read_bytes()
    result=json.loads((RAW/'root-whole-independent-review-20260927206.json').read_text())
    assert result['material_errors']==0 and result['reviewed']==20


def test_darai_metadata_registry_and_formatted_bibliography():
    import csv
    import pybtex
    from pybtex.database import parse_file
    root = RAW.parents[4]
    metadata = json.loads((RAW / 'metadata-review.json').read_text())
    with (root / 'cldf/dialects.csv').open() as stream:
        dialects = {r['ID']: r for r in csv.DictReader(stream)}
    with (RAW/'whole-dialects-proposed.csv').open() as stream:
        metadata['dialects'] += list(csv.DictReader(stream))
    for dialect in metadata['dialects']:
        assert dialects[dialect['ID']] == dialect
        assert not dialect['Latitude'] and not dialect['Longitude']
    with (root / 'cldf/languages.csv').open() as stream:
        language = next(r for r in csv.DictReader(stream) if r['ID'] == 'Darai')
    assert language['Clade'] == 'Bihari' and language['Glottocode'] == 'dara1250'
    installed = parse_file(str(root / 'cldf/sources.bib')).entries
    proposed = parse_file(str(RAW / 'source.bib')).entries
    assert len(proposed) == 7
    proposed.update(parse_file(str(RAW / 'whole-source-proposed.bib')).entries)
    for key, entry in proposed.items():
        assert installed[key] == entry
        assert pybtex.PybtexEngine().format_from_string(entry.to_string('bibtex'), 'plain', output_backend='markdown')


def test_compiled_darai_source_survival():
    import csv
    expected = set()
    for route in ['author-transcription','literal-preservation']:
        with (RAW/f'whole-proposed-{route}.csv').open() as stream:
            expected.update(r[10] for r in csv.reader(stream))
    with (RAW.parents[4] / 'cldf/form-source-keys.csv').open() as stream:
        found = {r['Source_Key'] for r in csv.DictReader(stream) if r['Source_Key'] in expected}
    assert found == expected, f'{len(found)}/{len(expected)} Darai source keys survived the full build'
