"""Source-extraction regressions; these do not certify an installed ingestion."""
import importlib.util
import json
import unicodedata as ud
from pathlib import Path

from segments.tokenizer import Tokenizer
from tags import GRAMMATICAL_TAGS

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/other/forms/raw_data/cuj_asur_2020'


def load(name):
    spec = importlib.util.spec_from_file_location('cuj_asur_' + name, RAW / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXTRACT = load('extract')
PARSE = load('parse')


def records():
    return [json.loads(line) for line in (RAW / 'entries.jsonl').open()]


def candidates():
    return [PARSE.parse(record) for record in records()]


def test_complete_snapshot_and_physical_identity():
    rows = records()
    assert len(rows) == len({r['entry_key'] for r in rows}) == 2005
    assert {r['printed_page'] for r in rows} == set(range(1, 106))
    assert all(r['pdf_page'] == r['printed_page'] + 5 for r in rows)
    assert all(r['tokens'] and r['tokens'][0]['font'] == PARSE.NATIVE for r in rows)
    assert rows[0]['entry_key'] == 'cujasur2020:p1:c1:y144.5'
    assert rows[-1]['entry_key'] == 'cujasur2020:p105:c2:y113.6'


def test_native_font_repair_keeps_raw_glyph_evidence():
    source = {r['entry_key']: r for r in records()}
    parsed = {r['entry_key']: r for r in candidates()}
    assert parsed['cujasur2020:p1:c2:y190.8']['native'] == 'अखरि'
    token = source['cujasur2020:p1:c2:y190.8']['tokens'][0]
    assert token['raw'] == 'अखरिर'
    assert any(g['text'] == 'रि' and g['width'] == 2.619 for g in token['glyphs'])
    assert parsed['cujasur2020:p2:c1:y273.1']['native'] == 'अंगुर'
    assert parsed['cujasur2020:p33:c1:y81.8']['native'] == 'चुरिङ'
    assert parsed['cujasur2020:p43:c2:y315.5']['native'] == 'ठेर्रा'
    assert parsed['cujasur2020:p75:c1:y498.5']['native'] == 'बिलई कूल'
    assert parsed['cujasur2020:p84:c1:y147.7']['native'] == 'मचिस'
    assert parsed['cujasur2020:p39:c1:y442.1']['native'] == 'झरिया'


def test_full_width_heading_bands_and_bottom_lines():
    rows = candidates()
    index = {r['entry_key']: r for r in rows}
    assert [s['printed_sense'] for s in index['cujasur2020:p39:c1:y208.7']['senses']] == ['1', '2', '3']
    rain = index['cujasur2020:p39:c1:y490.7']['senses']
    assert len(rain) == 1
    assert rain[0]['english_gloss'] == 'continuous rain for more than a week'
    assert index['cujasur2020:p60:c1:y528.0']['senses'][0]['english_gloss'] == 'a kind of snake'
    assert any(r['entry_key'] == 'cujasur2020:p4:c1:y547.3' for r in rows)


def test_gloss_stops_before_usage_and_cross_reference_labels():
    index = {r['entry_key']: r for r in candidates()}
    assert index['cujasur2020:p1:c1:y375.6']['senses'][0]['english_gloss'] == 'straw hook'
    assert index['cujasur2020:p2:c1:y321.6']['senses'][0]['english_gloss'] == 'finger'
    assert index['cujasur2020:p2:c1:y370.5']['senses'][0]['english_gloss'] == 'pickle'
    assert index['cujasur2020:p9:c1:y81.8']['senses'][0]['english_gloss'] == 'POSS'


def test_homonyms_and_crossrefs_are_not_flattened_or_invented():
    rows = candidates()
    assert len([r for r in rows if r['native'] == 'अच्छा']) == 2
    assert {r['homonym'] for r in rows if r['native'] == 'अच्छा'} == {'1', '2'}
    xref = next(r for r in rows if r['entry_key'] == 'cujasur2020:p2:c1:y402.3')
    assert xref['kind'] == 'cross-reference' and xref['relation_label'] == 'fr. var.'
    assert not xref['ipa'] and xref['relationship_body'] == 'fr. var. of अचाएर'


def test_candidate_accounting_is_explicitly_unreviewed():
    rows = candidates()
    assert rows == [json.loads(line) for line in (RAW / 'candidates.jsonl').open()]
    assert sum(r['kind'] == 'cross-reference' for r in rows) == 200
    assert sum(bool(r['ipa']) for r in rows) == 1714
    assert sum(len(r['senses']) for r in rows) == 2107
    assert sum('unresolved-native-glyph' in r['review_flags'] for r in rows) == 9
    assert all(r['status'] == 'candidate-unreviewed' for r in rows)


def test_reference_prefixes_and_scientific_identifications_survive():
    index = {r['entry_key']: r for r in candidates()}
    prefixed = index['cujasur2020:p58:c2:y80.8']
    assert prefixed['kind'] == 'cross-reference'
    assert any(r['kind'] == 'variant-of' and r['targets'][0]['native'] == 'लोलो'
               for r in prefixed['references'])
    assert index['cujasur2020:p64:c1:y509.8']['scientific_name_candidates'] == ['Lantana camara']
    assert index['cujasur2020:p93:c2:y504.4']['native'] == 'साकाअ◌ो'
    assert index['cujasur2020:p92:c2:y81.8']['ipa'] == 'loaː'
    nested = index['cujasur2020:p22:c1:y540.0']['references']
    assert [(r['kind'], r['scope']) for r in nested] == [
        ('complex-form-of', 'entry'), ('variant-of', 'nested-reference')]


def test_audit_accounts_for_placeholders_and_unresolved_targets():
    module = load('audit')
    rows = module.audit(candidates(), {r['entry_key']: r for r in records()})
    saved = json.loads((RAW / 'audit.json').read_text())
    assert rows == saved['records']
    assert len(rows) == 2005
    assert all(r['installed_rows'] == 0 for r in rows)
    index = {r['entry_key']: r for r in rows}
    assert index['cujasur2020:p84:c1:y164.4']['classification'] == 'printed-bare-headword'
    assert index['cujasur2020:p16:c1:y265.5']['classification'] == 'printed-undefined-transcribed-headword'
    assert index['cujasur2020:p64:c1:y509.8']['classification'] == 'scientific-identification-without-English-definition'
    missing = [t for r in rows for ref in r['references'] for t in ref['targets'] if t['status'] == 'missing']
    assert {t['native'] for t in missing} == {'सुकुल'}


def proposal():
    return load('import_source').build(candidates(), json.loads((RAW / 'audit.json').read_text()))


def test_rich_proposal_separates_grammar_scripts_and_source_damage():
    rows, audit = proposal()
    assert len(rows) == len({r[10] for r in rows}) == 2106
    assert len(audit['records']) == 2005
    assert sum(r['decision'] == 'excluded-corrupt-head' for r in audit['records']) == 1
    assert all(len(r) == 15 and r[2] and r[0] == 'Asuri' for r in rows)
    assert all(set(r[14].split()) <= GRAMMATICAL_TAGS for r in rows)
    index = {r[10]: r for r in rows}
    poss = index['cujasur2020:p9:c1:y81.8:sense:0']
    assert 'poss' in poss[14].split() and poss[3] == 'possessive marker'
    native = index['cujasur2020:p84:c1:y164.4:sense:0']
    assert native[2] == native[4] == 'मटर दाइल' and not native[5]
    assert all('cid:' not in r[2] and 'cid:' not in r[4] for r in rows)


def test_embedded_Hindi_definitions_and_explicit_compound_members():
    rows, _ = proposal()
    index = {r[10]: r for r in rows}
    millet = index['cujasur2020:p46:c2:y160.2:sense:0']
    assert millet[3] == 'black finger millet'
    assert 'काला रागी' in millet[6] and 'p. 44' in millet[7]
    compound = index['cujasur2020:p66:c1:y95.3:sense:0']
    assert compound[3] == 'soil used to paint walls'
    assert compound[13].split('|') == ['cujasur2020:p65:c2:y473.1:sense:0',
                                       'cujasur2020:p102:c1:y453.1:sense:0']
    assert index['cujasur2020:p102:c1:y453.1:sense:0'][3] == 'soil'
    assert not millet[13]  # "unspec. comp. form" is not a component assertion.
    parsed = {r['entry_key']: r for r in candidates()}
    target = parsed['cujasur2020:p19:c2:y210.4']['references'][0]['targets'][0]
    assert target['printed_sense'] == '1' and not target['homonym']


def test_profile_covers_every_proposed_form_and_preserves_uncertain_stops():
    tokenizer = Tokenizer(str(ROOT / 'conversion/cuj-asur.txt'))
    def convert(form):
        return ud.normalize('NFC', tokenizer(form, column='IPA').replace(' ', '').replace('#', ' '))
    assert convert('jom y w ʈʰaːɽ') == 'jom y v ṭʰāṛ'
    assert convert('ãː mː oɽeʔᵍ kʰokᵏro') == 'ā̃ mm oṛeʔᵍ kʰokᵏro'
    assert convert('-आए:') == '-आए:'  # Native colon is not blindly treated as IPA length.
    rows, _ = proposal()
    for row in rows:
        assert '�' not in convert(row[2])
        assert convert(ud.normalize('NFD', row[2])) == convert(ud.normalize('NFC', row[2]))


def test_proposed_variant_links_resolve_without_cycles():
    rows, audit = proposal()
    index = {r[10]: r for r in rows}
    assert sum(bool(r[11]) for r in rows) == 185
    assert len(audit['pending_relationships']) == 16
    for row in rows:
        seen, current = set(), row[10]
        while current:
            assert current in index and current not in seen
            seen.add(current)
            current = index[current][11]


def test_nested_reference_after_target_does_not_attach_to_head():
    parsed = {r['entry_key']: r for r in candidates()}
    refs = parsed['cujasur2020:p6:c2:y190.0']['references']
    assert [r['scope'] for r in refs] == ['entry', 'nested-reference']
    assert parsed['cujasur2020:p9:c1:y233.9']['references'][0]['targets'][0]['native'] == '-लाङ'
    rows, _ = proposal()
    index = {r[10]: r for r in rows}
    assert {'first-person', 'du', 'inclusive'} <= set(index['cujasur2020:p9:c1:y233.9:sense:0'][14].split())
    assert 'Musca domestica' not in index['cujasur2020:p89:c2:y232.7:sense:1'][6]
    assert 'Musca domestica' in index['cujasur2020:p89:c2:y232.7:sense:2'][6]


def test_compiled_source_survival():
    import csv
    expected = {r[10] for r in proposal()[0]}
    with (ROOT / 'cldf/form-source-keys.csv').open() as stream:
        found = {r['Source_Key'] for r in csv.DictReader(stream) if r['Source_Key'] in expected}
    assert found == expected, 'Full build pending: installed CUJ Asur source keys are missing from compiled CLDF'


def test_installed_rows_and_actual_pipeline_layers():
    import csv
    import io
    from make_cldf import parse_file
    rows, _ = proposal()
    path = ROOT / 'data/other/forms/20260921-cuj-asur.csv'
    assert rows == list(csv.reader(path.open()))
    error = io.StringIO()
    parsed, stats = parse_file(str(path), errors=error)
    assert not error.getvalue()
    assert len(parsed) == 2106
    original = {r[10]: r for r in rows}
    for row in parsed:
        source = original[row.entry_key]
        assert row.old_form == source[2] and row.ipa == source[5]
        assert row.variant_of_key == source[11]
        assert row.derivation_parent_keys == source[13]
    assert len({r.entry_key for r in parsed}) == 2106
