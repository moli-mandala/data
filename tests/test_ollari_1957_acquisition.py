"""Source-review and preview integrity; final compiled ingestion remains pending."""
import csv
import hashlib
import json
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / 'data/other/forms/raw_data/bhattacharya_ollari_1957'


def test_complete_prose_review_preserves_evidence_order_and_scope():
    from collections import Counter
    import unicodedata
    manifest = json.loads((RAW / 'manifest.json').read_text())['comparison_prose_review']
    counts = Counter()
    records = []
    for page in range(48, 78):
        lexical_path = RAW / f'reviewed-p{page}.json'
        lexical = json.loads(lexical_path.read_text())
        review = json.loads((RAW / f'comparison-reviewed-p{page}.json').read_text())
        assert review['printed_page'] == page and review['pdf_page'] == page + 11
        assert review['lexical_record_sha256'] == hashlib.sha256(lexical_path.read_bytes()).hexdigest()
        assert [r['entry_key'] for r in review['records']] == [r['entry_key'] for r in lexical]
        for row in review['records']:
            passages = row['passages']
            assert [p['position'] for p in passages] == list(range(1, len(passages) + 1))
            assert row['status'] == ('visually-transcribed-prose' if passages else 'visually-checked-no-prose')
            for passage in passages:
                assert passage['text'].strip() == passage['text']
                assert unicodedata.is_normalized('NFC', passage['text'])
                counts[passage['kind']] += 1
        records.extend(review['records'])
    assert len(records) == len({r['entry_key'] for r in records}) == 657
    assert sum(counts.values()) == 509
    assert dict(counts) == manifest['passage_counts']
    assert manifest['reviewed_pages'] == list(range(48, 78))
    assert manifest['physical_records'] == 657 and manifest['remaining_physical_records'] == 0
    assert sum(bool(r.get('reading_uncertainty')) for r in records) == manifest['typed_reading_uncertainties']
    sun = next(r for r in records if r['source_head'] == 'VĒLE')
    assert any('different source' in p['text'] for p in sun['passages'])
    assert 'Do not infer cognacy' in sun['relationship_review']
    mouse = next(r for r in records if r['source_head'] == 'SIREL')
    assert mouse['passages'][0]['kind'] == 'source-derivation'
    assert 'Poya sire, pl. sirel' in mouse['passages'][1]['text']


def test_prose_reading_review_preserves_corrections_and_residual_uncertainty():
    audit = json.loads((RAW / 'audits/prose-reading-review.json').read_text())
    assert audit['resolved'] == 4 and audit['retained_uncertain'] == 2
    assert len(audit['records']) == 6
    for item in audit['records']:
        page = json.loads((RAW / f"comparison-reviewed-p{item['printed_page']}.json").read_text())
        current = next(r for r in page['records'] if r['entry_key'] == item['entry_key'])
        assert current == item['after']
        assert 'reading_uncertainty' in item['before']
        assert bool(current.get('reading_uncertainty')) == (item['result'] == 'retained-uncertain')
    corrected = next(x for x in audit['records'] if x['printed_page'] == 60)
    assert corrected['before']['passages'][0]['text'] == 'Go. A. ciṗre'
    assert corrected['after']['passages'][0]['text'] == 'Go. A. cipṛe'


def test_prose_auxiliary_citations_are_passage_specific_and_resolve():
    from pybtex.database import parse_file
    references = json.loads((RAW / 'explicit-reference-resolution.json').read_text())['records']
    assert len(references) == 3
    auxiliary = parse_file(str(RAW / 'auxiliary-references.bib'))
    registry = parse_file(str(RAW.parents[4] / 'cldf/sources.bib'))
    assert set(auxiliary.entries) == {'roy1912mundas', 'burrow1953parji'}
    with (RAW / 'preview/20260921-bhattacharya-ollari-entry-texts.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    cited = {(r['Entry_Key'], int(r['Position'])): r for r in rows if ';' in r['Source']}
    assert len(cited) == 3
    for reference in references:
        assert reference['registry_key'] in auxiliary.entries or reference['registry_key'] in registry.entries
        row = cited[(reference['entry_key'], reference['position'])]
        assert row['Source'].endswith(';' + reference['citation'])
    # Language labels alone must not add a book citation.
    assert sum('burrow1953parji' in r['Source'] for r in rows) == 1
    assert sum('LSI[' in r['Source'] for r in rows) == 1


def test_internal_relationship_review_covers_claims_and_protects_homonyms():
    ledger = json.loads((RAW / 'internal-relationship-review.json').read_text())
    kinds = {'cross-reference', 'source-derivation', 'source-analysis', 'source-etymology'}
    evidence = {(r['entry_key'], p['position']): p
                for file in sorted(RAW.glob('comparison-reviewed-p*.json'))
                for r in json.loads(file.read_text())['records']
                for p in r['passages'] if p['kind'] in kinds}
    claims = ledger['claims']
    assert len(claims) == len(evidence) == 51
    assert {(c['entry_key'], c['position']) for c in claims} == set(evidence)
    units = {r['entry_key']: r for r in map(json.loads, (RAW / 'draft/lexical-units.jsonl').read_text().splitlines())}
    for claim in claims:
        passage = evidence[(claim['entry_key'], claim['position'])]
        assert claim['source_text'] == passage['text'] and claim['kind'] == passage['kind']
        assert claim['emitted_edges'] == []
        for candidate in claim.get('target_candidates', []):
            target = units[candidate['entry_key']]
            assert (candidate['form'], candidate['gloss']) == (target['form'], target['gloss'])
    crossrefs = {c['source_head']: c for c in claims if c['kind'] == 'cross-reference'}
    assert len(crossrefs) == 34
    assert crossrefs['BER']['target_candidates'][0]['gloss'] == 'big (m.)'
    assert crossrefs['ĀM']['target_candidates'][0]['gloss'] == 'to yawn'
    assert crossrefs['BĀR-']['proposed_relation']['kind'] == 'variant'
    assert crossrefs['MUTAM SĪKAṬ']['disposition'] == 'candidate-transcription-difference'
    assert 'proposed_relation' not in crossrefs['PUL']


def test_complete_vocabulary_page_scaffold_is_pinned_and_nonempty():
    ledger = json.loads((RAW / 'ocr/pages.json').read_text())
    assert [r['printed_page'] for r in ledger] == list(range(48, 78))
    assert [r['pdf_page'] for r in ledger] == list(range(59, 89))
    assert len({r['text_sha256'] for r in ledger}) == 30
    for page in ledger:
        stem = RAW / 'ocr' / f"p{page['printed_page']:02d}"
        for suffix, field in [('txt', 'text_sha256'), ('tsv', 'tsv_sha256')]:
            assert hashlib.sha256(stem.with_suffix('.' + suffix).read_bytes()).hexdigest() == page[field]
        with stem.with_suffix('.tsv').open() as stream:
            words = [r for r in csv.DictReader(stream, delimiter='\t') if r['level'] == '5' and r['text'].strip()]
        assert len(words) > 50
        assert all(int(r['width']) > 0 and int(r['height']) > 0 for r in words)
        assert page['status'] == 'unreviewed-OCR-not-installable'


def test_manifest_distinguishes_scan_edition_and_unavailable_alternatives():
    manifest = json.loads((RAW / 'manifest.json').read_text())
    assert manifest['year'] == 1957 and '1956' in manifest['series']
    assert manifest['text_layer_characters'] == 0
    assert manifest['pdf_pages'] == 93
    assert manifest['vocabulary_pdf_pages'] == [59, 88]
    assert manifest['baseline']['canonical_language'] == 'OllariGadaba'
    assert manifest['ocr']['OMP_THREAD_LIMIT'] == 1
    assert manifest['status'] == 'canonical source inputs installed; full compilation and validation pending'
    assert manifest['installation']['rows'] == 880


def test_reviewed_pages_remain_attached_to_physical_evidence():
    candidates = {
        r['entry_key']: r for r in map(json.loads, (RAW / 'candidate-entries.jsonl').read_text().splitlines())
    }
    reviewed = []
    for page, count in [(48, 13), (49, 22), (50, 21), (51, 23), (52, 24), (53, 21), (54, 20), (55, 16), (56, 18), (57, 19), (58, 15), (59, 22), (60, 28), (61, 31), (62, 25), (63, 20), (64, 20), (65, 21), (66, 23), (67, 22), (68, 30), (69, 34), (70, 16), (71, 19), (72, 22), (73, 23), (74, 16), (75, 20), (76, 27), (77, 26)]:
        records = json.loads((RAW / f'reviewed-p{page}.json').read_text())
        assert len(records) == count
        assert {r['entry_key'] for r in records} == {
            k for k, r in candidates.items() if r['printed_page'] == page
        }
        for record in records:
            evidence = candidates[record['entry_key']]
            assert record['raw_ocr'] == '\n'.join(line['text'] for line in evidence['lines'] if line.get('origin') != 'manual-scan-transcription')
            assert record['pdf_page'] == page + 11
            assert record['comparison_review'].startswith('pending;')
        reviewed.extend(records)
    keys = {r['entry_key'] for r in reviewed}
    assert len(keys) == 657
    assert all(not r['explicit_derivation_parent'] or r['explicit_derivation_parent'] in keys for r in reviewed)


def test_reviewed_contrasts_homonyms_and_alternates_are_not_flattened():
    records = json.loads((RAW / 'reviewed-p49.json').read_text())
    forms = {r['transcribed_form']: r for r in records}
    assert {'arg-', 'argil', 'aṛ-', 'aṛtol', 'asaṛ', 'āṭe', 'āta'} <= forms.keys()
    homonyms = [r for r in records if r['transcribed_form'] == 'ām']
    assert {(r['source_pos'], r['gloss']) for r in homonyms} == {('sb.', 'yawn'), ('pron.', 'we')}
    assert len({r['entry_key'] for r in homonyms}) == 2
    assert forms['aṛup-, aṛut-']['alternate_forms'] == ['aṛup-', 'aṛut-']
    assert forms['āya, aya']['alternate_forms'] == ['āya', 'aya']
    assert forms['ān']['morphology'] == [{'label': 'obl. stem', 'raw': 'an-', 'kind': 'stem'}]


def test_reviewed_boundary_repairs_and_new_diacritics():
    import runpy
    records = runpy.run_path(str(RAW / 'segment.py'))['extract']()
    pinned = [json.loads(line) for line in (RAW / 'candidate-entries.jsonl').read_text().splitlines()]
    assert records == pinned
    assert len([r for r in records if r['printed_page'] == 69]) == 34
    p50 = json.loads((RAW / 'reviewed-p50.json').read_text())
    assert [r['transcribed_form'] for r in p50[5:7]] == ['ɔlɔken', 'ɔssa']
    assert [(r['source_pos'], r['gloss']) for r in p50 if r['transcribed_form'] == 'inḍi'] == [('num.', 'two (n.)'), ('adv.', 'now; this time')]
    p51 = json.loads((RAW / 'reviewed-p51.json').read_text())
    assert p51[20]['alternate_forms'] == ['unḍup-', 'unḍut-', 'unḍuk-']


def test_skewed_page_column_order_and_uncertainties():
    records = [json.loads(line) for line in (RAW / 'candidate-entries.jsonl').read_text().splitlines()]
    p52 = [r for r in records if r['printed_page'] == 52]
    assert len(p52) == 24
    assert all(line['column'] == r['column'] for r in p52 for line in r['lines'])
    assert [r['top'] for r in p52[-4:]] == [3455, 3684, 3834, 4376]
    review52 = json.loads((RAW / 'reviewed-p52.json').read_text())
    assert review52[8]['lexical_review'] == 'uncertain-head-character'
    assert review52[8]['uncertainties'][0]['type'] == 'ambiguous-glyph'
    review53 = json.loads((RAW / 'reviewed-p53.json').read_text())
    assert {'org-', 'oṛg-', 'oṛ-, oṛt-'} <= {r['transcribed_form'] for r in review53}
    assert review53[2]['uncertainties'][0]['type'] == 'ambiguous-example-symbol'


def test_printed_homonym_numbers_are_metadata_not_phonetic_forms():
    records = json.loads((RAW / 'reviewed-p55.json').read_text())
    for form, glosses in [('karke', {'the month of caitra (March–April)', 'unripe mango'}), ('kākal', {'brinjal', 'crow'})]:
        pair = [r for r in records if r['transcribed_form'] == form]
        assert len(pair) == 2
        assert {r['homonym_number'] for r in pair} == {1, 2}
        assert {r['gloss'] for r in pair} == glosses
        assert len({r['entry_key'] for r in pair}) == 2
        assert {r['source_head'][-1] for r in pair} == {'¹', '²'}
    p54 = json.loads((RAW / 'reviewed-p54.json').read_text())
    assert p54[13]['morphology'] == [{'label': 'pl.', 'raw': 'kaṇul', 'kind': 'full-form'}]
    assert {'kaṭ-', 'kat-', 'kaṇ', 'kaṅar'} <= {r['transcribed_form'] for r in p54}
    assert p54[18]['transcribed_form'] == 'kanīr'
    assert p54[18]['review_history'][0]['resolved_uncertainties'][0]['type'] == 'ambiguous-diacritic'


def test_nasal_length_context_and_thigh_boundary_are_preserved():
    p56 = json.loads((RAW / 'reviewed-p56.json').read_text())
    assert p56[1]['transcribed_form'] == 'kã·j-'
    assert p56[0]['transcribed_form'] == 'kākor'
    assert p56[0]['usage_context'] == 'nīr'
    assert {(r['source_pos'], r['gloss']) for r in p56 if r['transcribed_form'] == 'ki'} == {('sb.', 'hand'), ('adv.', 'or')}
    p57 = json.loads((RAW / 'reviewed-p57.json').read_text())
    thigh = p57[7]
    assert thigh['entry_key'] == 'ollari1957:p57:c1:y3913'
    assert thigh['transcribed_form'] == 'kuyug'
    assert thigh['morphology'] == [{'label': 'pl.', 'raw': 'kuyugul', 'kind': 'full-form'}]
    assert 'KuyuG' not in p57[6]['raw_ocr']


def test_entry_pronunciation_notes_and_length_contrast_survive():
    p58 = json.loads((RAW / 'reviewed-p58.json').read_text())
    p59 = json.loads((RAW / 'reviewed-p59.json').read_text())
    for record in [p58[4], p59[19]]:
        assert record['pronunciation_overrides'] == [{'source': 'j', 'value': 'z', 'scope': 'entry'}]
        assert 'j' in record['transcribed_form']
    forms = {r['transcribed_form']: r for r in p59}
    assert forms['kor']['gloss'] == 'fowl'
    assert forms['kōr']['gloss'] == 'horn'
    assert forms['gāṭi']['gloss'] == 'many'
    assert forms['gã·ti']['gloss'] == 'joint'
    assert p58[5]['alternate_forms'] == ['kelmaṅ', 'kelman']


def test_non_ocr_entry_is_explicit_and_source_pos_is_retained():
    p60 = json.loads((RAW / 'reviewed-p60.json').read_text())
    missing = p60[9]
    assert missing['entry_key'] == 'ollari1957:p60:c1:manual-goler'
    assert missing['raw_ocr'] == ''
    assert missing['alternate_forms'] == ['gōler-', 'gōlen-']
    assert missing['manual_scan_evidence'][0]['words'] == []
    assert 'Approximate' in missing['manual_scan_evidence'][0]['coordinate_note']
    assert p60[15]['pronunciation_overrides'][0]['value'] == 'ts'
    assert p60[20]['pronunciation_overrides'][0]['value'] == 'dz'
    p61 = json.loads((RAW / 'reviewed-p61.json').read_text())
    assert p61[23]['source_pos'] == 'sb.'
    assert p61[23]['gloss'] == 'to swallow'
    assert p61[0]['alternate_forms'] == ['ṭuṅ', 'pelṭuṅ']


def test_optional_segment_and_partial_ocr_omission_are_preserved():
    p62 = json.loads((RAW / 'reviewed-p62.json').read_text())
    assert p62[0]['transcribed_form'] == 'tite'
    assert p62[7]['transcribed_form'] == 'tīte'
    assert p62[10]['transcribed_form'] == 'tuñ(g)-'
    assert p62[10]['alternate_forms'] == ['tuñ-', 'tuñg-']
    assert p62[22]['source_head'] == 'TŌṬP, TŌṬT-'
    p63 = json.loads((RAW / 'reviewed-p63.json').read_text())
    restored = p63[9]
    assert restored['entry_key'] == 'ollari1957:p63:c1:manual-nagup'
    assert 'make to laugh' in restored['raw_ocr']
    assert 'NAGUP' not in restored['raw_ocr']
    assert restored['manual_scan_evidence'][0]['text'] == 'NAGUP-, NAGUT-,'
    assert restored['explicit_derivation_parent'] == p63[7]['entry_key']


def test_new_numbered_homonyms_and_gender_forms_remain_distinct():
    p65 = json.loads((RAW / 'reviewed-p65.json').read_text())
    for form, glosses in [('panḍ-', {'to become tired', 'to be able'}), ('par-', {'to fall', 'to receive'})]:
        pair = [r for r in p65 if r['transcribed_form'] == form]
        assert len(pair) == 2
        assert {r['homonym_number'] for r in pair} == {1, 2}
        assert {r['gloss'] for r in pair} == glosses
        assert len({r['entry_key'] for r in pair}) == 2
    p64 = json.loads((RAW / 'reviewed-p64.json').read_text())
    assert [(r['transcribed_form'], r['gloss']) for r in p64[3:6]] == [('niya', 'good'), ('niyaṭe', 'good (f. n.)'), ('niyaṭonḍ', 'good (m.)')]
    assert p65[3]['transcribed_form'] == 'pañgil'


def test_length_retroflex_contrasts_and_cross_reference_only_record():
    p66 = json.loads((RAW / 'reviewed-p66.json').read_text())
    forms = {r['transcribed_form']: r for r in p66}
    assert [(forms[f]['gloss']) for f in ['pal', 'pāl', 'pinḍe', 'pinde']] == ['tooth', 'milk', 'verandah', 'insect']
    assert forms['paṛṅ(g)-']['alternate_forms'] == ['paṛṅ-', 'paṛṅg-']
    p67 = json.loads((RAW / 'reviewed-p67.json').read_text())
    crossref = p67[13]
    assert crossref['transcribed_form'] == 'pul'
    assert crossref['gloss'] == crossref['source_pos'] == ''
    assert crossref['cross_reference']['target_form'] == 'ber-pul'
    assert crossref['cross_reference']['relation'] == 'see'
    assert p67[8]['morphology'][0]['raw'] == 'ev'
    assert p67[8]['morphology'][0]['expanded_form'] == 'punev'
    assert p67[8]['morphology'][0]['kind'] == 'ending-replacement'
    assert p67[8]['morphology'][0]['evidence']['printed_page'] == 18
    pair = [r for r in p67 if r['transcribed_form'] == 'pun-']
    assert {r['homonym_number'] for r in pair} == {1, 2}
    assert len({r['entry_key'] for r in pair}) == 2


def test_cross_reference_target_and_source_omissions_are_preserved():
    p67 = json.loads((RAW / 'reviewed-p67.json').read_text())
    p68 = json.loads((RAW / 'reviewed-p68.json').read_text())
    p69 = json.loads((RAW / 'reviewed-p69.json').read_text())
    assert p67[13]['cross_reference']['target_entry_key'] == p69[25]['entry_key']
    assert p69[25]['transcribed_form'] == 'ber pul'
    assert p68[19]['source_pos'] == ''
    assert p68[19]['gloss'] == 'cheek'
    assert p68[21]['senses'] == [{'source_pos': 'postpos.', 'gloss': 'on, upon'}, {'source_pos': 'sb.', 'gloss': 'top of something'}]
    assert p68[5]['alternate_forms'] == ['pēpal', 'pēpɔl']
    assert p69[5]['transcribed_form'] == 'bākɔs'
    assert p69[0]['pronunciation_overrides'] == [{'source': 'j', 'value': 'z', 'scope': 'entry'}]
    assert {r['homonym_number'] for r in p69 if r['transcribed_form'] == 'bābu'} == {1, 2}


def test_explicit_causatives_and_nasal_spelling_have_source_evidence():
    p70 = json.loads((RAW / 'reviewed-p70.json').read_text())
    p71 = json.loads((RAW / 'reviewed-p71.json').read_text())
    p72 = json.loads((RAW / 'reviewed-p72.json').read_text())
    assert p70[4]['explicit_derivation_parent'] == p70[3]['entry_key']
    assert p70[4]['source_pos'] == 'vb. cs.'
    assert p72[8]['explicit_derivation_parent'] == p72[10]['entry_key']
    assert p72[8]['source_pos'] == 'vb. cs.'
    assert p71[3]['transcribed_form'] == 'mã·jik'
    assert p71[4]['alternate_forms'] == ['māyṅ-', 'māyṅg-']
    assert {r['gloss'] for r in p71 if r['transcribed_form'] == 'māl'} == {'daughter', 'wine'}
    assert p71[0]['explicit_derivation_parent'] == ''
    assert 'probably' in p71[0]['source_note']


def test_printed_semantic_uncertainty_and_retroflex_contrast_survive():
    p73 = json.loads((RAW / 'reviewed-p73.json').read_text())
    assert p73[6]['gloss'] == 'palate (tongue ?)'
    assert p73[1]['gloss'] == 'month of jyaiṣṭha (April–May)'
    assert p73[5]['alternate_forms'] == ['vaṅ-', 'vaṅg-']
    assert p73[7]['transcribed_form'] == 'vaṭ-'
    assert p73[13]['transcribed_form'] == 'vat'
    assert p73[12]['transcribed_form'] == 'vanḍdan magginḍ'
    assert p73[20]['morphology'] == [{'label': 'pl.', 'raw': 'vāṅgusul', 'kind': 'full-form'}]


def test_final_pages_preserve_verb_contrasts_and_optional_retroflex():
    p75 = json.loads((RAW / 'reviewed-p75.json').read_text())
    p76 = json.loads((RAW / 'reviewed-p76.json').read_text())
    assert p75[2]['alternate_forms'] == ['sanḍup-', 'sanḍut-']
    assert p75[7]['alternate_forms'] == ['sandup-', 'sandut-']
    assert p75[7]['explicit_derivation_parent'] == p75[6]['entry_key']
    assert p75[13]['transcribed_form'] == 'salñiḍ'
    assert p76[2]['explicit_derivation_parent'] == p76[1]['entry_key']
    assert p76[15]['alternate_forms'] == ['sirṅaṭonḍ', 'sirṅaṭṭonḍ']
    assert p76[21]['alternate_forms'] == ['sī-', 'sīn-', 'siy-', 'sīd-']
    assert p76[24]['pronunciation_overrides'] == [{'source': 'j', 'value': 'z', 'scope': 'entry'}]


def test_complete_page_coverage_and_explicit_cross_reference_endpoints():
    records = [r for f in sorted(RAW.glob('reviewed-p*.json')) for r in json.loads(f.read_text())]
    by_key = {r['entry_key']: r for r in records}
    assert {r['printed_page'] for r in records} == set(range(48, 78))
    assert len(records) == len(by_key) == 657
    for r in records:
        target = r.get('cross_reference', {}).get('target_entry_key')
        if target:
            assert target in by_key
    p77 = json.loads((RAW / 'reviewed-p77.json').read_text())
    assert p77[2]['morphology'] == [{'label': 'pl.', 'raw': '-til', 'kind': 'suffix'}]
    assert p77[6]['cross_reference']['target_entry_key'] == p77[9]['entry_key']
    assert p77[17]['morphology'] == [{'label': 'pl.', 'raw': 'soṭiṭev', 'kind': 'full-form'}]


def test_draft_expansion_reproduces_all_units_and_holds_ambiguous_scope(tmp_path):
    import runpy
    module = runpy.run_path(str(RAW / 'expand_lexical_units.py'))
    summary = module['build'](tmp_path)
    assert (summary['physical_records'], summary['lexical_units'], summary['printed_alternates'], summary['inflections'], summary['pending_morphology']) == (657, 880, 123, 100, 2)
    for name in ['lexical-units.jsonl', 'expansion-audit.jsonl', 'expansion-summary.json']:
        assert (tmp_path / name).read_bytes() == (RAW / 'draft' / name).read_bytes()
    units = [json.loads(line) for line in (tmp_path / 'lexical-units.jsonl').read_text().splitlines()]
    by_key = {r['entry_key']: r for r in units}
    assert by_key['ollari1957:p67:c2:y250:inflection:1']['form'] == 'punev'
    assert by_key['ollari1957:p58:c1:y1917:inflection:1']['form'] == 'kerjil'
    assert by_key['ollari1957:p58:c1:y1917:inflection:1']['pronunciation_overrides'][0]['value'] == 'z'
    assert by_key['ollari1957:p54:c2:y3663:inflection:1']['form'] == 'kanīrtil'
    for key in ['ollari1957:p49:c2:y4059', 'ollari1957:p70:c2:y2780']:
        assert key + ':inflection:1' not in by_key
    audit = [json.loads(line) for line in (tmp_path / 'expansion-audit.jsonl').read_text().splitlines()]
    assert len(audit) == 657
    assert sum(len(r['emitted_keys']) for r in audit) == len(units)
    assert {k for r in audit for k in r['emitted_keys']} == set(by_key)


def test_draft_expansion_rejects_unreviewed_notation_and_bad_replacements():
    import copy
    import runpy
    import pytest
    expand = runpy.run_path(str(RAW / 'expand_lexical_units.py'))['expand']
    source = json.loads((RAW / 'reviewed-p67.json').read_text())[8]
    bad = copy.deepcopy(source)
    bad['morphology'][0]['expanded_form'] = 'punedev'
    with pytest.raises(ValueError, match='Unverified ending replacement'):
        expand([bad])
    bad = copy.deepcopy(source)
    bad['transcribed_form'] = 'pun(ed)'
    with pytest.raises(ValueError, match='Unexpanded source notation'):
        expand([bad])


def test_draft_profile_covers_corpus_and_scopes_pronunciation(tmp_path):
    import runpy
    import unicodedata
    module = runpy.run_path(str(RAW / 'convert_draft.py'))
    report = module['build'](tmp_path)
    assert report['draft_units'] == 880 and report['unmapped_forms'] == 0
    for name in ['converted-units.jsonl', 'conversion-report.json']:
        assert (tmp_path / name).read_bytes() == (RAW / 'draft' / name).read_bytes()
    convert = module['convert']
    assert convert('kã·j-')['display'] == 'kā̃j-'
    assert convert('neliṅ')['display'] == 'neliŋ'
    assert convert('ɔlɔken')['display'] == 'ɔlɔken'
    assert convert('jir er-', [{'scope': 'entry', 'source': 'j', 'value': 'dz'}])['display'] == 'ʣir er-'
    assert convert('tanḍ jir', [{'scope': 'entry', 'source': 'j', 'value': 'z'}])['display'] == 'tanḍ zir'
    assert convert('jir')['display'] == 'jir'
    rows = [json.loads(line) for line in (RAW / 'draft/lexical-units.jsonl').read_text().splitlines()]
    for row in rows:
        assert convert(unicodedata.normalize('NFD', row['form']), row['pronunciation_overrides']) == convert(row['form'], row['pronunciation_overrides'])


def test_rich_preview_preserves_source_layers_through_actual_parser(tmp_path, monkeypatch):
    import io
    import runpy
    import sys
    root = RAW.parents[4]
    monkeypatch.chdir(root)
    monkeypatch.syspath_prepend(str(RAW))
    monkeypatch.syspath_prepend(str(root))
    import source_meta
    import make_cldf
    module = runpy.run_path(str(RAW / 'preview_import.py'))
    output = tmp_path / 'other/forms/preview'
    summary = module['build'](output)
    assert summary['rows'] == 880 and summary['physical_records'] == 657
    assert summary['variants'] == 123 and summary['derivations'] == 110
    for name in ['20260921-bhattacharya-ollari.csv', '20260921-bhattacharya-ollari.yaml', 'rich-row-audit.jsonl', 'physical-record-audit.jsonl', 'rich-preview-summary.json']:
        assert (output / name).read_bytes() == (RAW / 'preview' / name).read_bytes()
    meta = source_meta.SourceMeta([output / '20260921-bhattacharya-ollari.yaml'])
    monkeypatch.setattr(source_meta, 'load', lambda: meta)
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(output / '20260921-bhattacharya-ollari.csv'), errors)
    assert errors.getvalue() == '' and len(parsed) == stats['converted'] == 880
    audits = [json.loads(line) for line in (output / 'rich-row-audit.jsonl').read_text().splitlines()]
    for row, audit in zip(parsed, audits):
        assert row.old_form == audit['proposed_row'][2]
        assert row.ipa == audit['proposed_row'][5]
        assert row.form == audit['expected_display']
    raw = [a['proposed_row'] for a in audits]
    by_key = {r[10]: r for r in raw}
    assert len(by_key) == 880 and all(len(r) == 15 for r in raw)
    assert all(not r[11] or r[11] in by_key for r in raw)
    assert all(not r[13] or r[13] in by_key for r in raw)
    assert by_key['ollari1957:p67:c2:y1342'][3] == ''
    plural = by_key['ollari1957:p67:c2:y250:inflection:1']
    assert plural[2] == 'punev' and 'pl' in plural[14].split()
    assert all('raw OCR' not in r[6] for r in raw)
    # The actual parser's generated IDs, not guessed row positions, bind prose.
    from entry_text_sources import read_entry_text_sources
    prose = runpy.run_path(str(RAW / 'prose_preview.py'))['build'](output)
    assert prose['passages'] == 509 and prose['entries_with_prose'] == 466
    assert prose['uncertain_entries'] == 2
    forms_path = output / 'parsed-forms.csv'
    with forms_path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ID', 'Entry_Key'])
        writer.writerows((r.id, r.entry_key) for r in reversed(parsed))
    sidecar = output / '20260921-bhattacharya-ollari-entry-texts.csv'
    blocks = list(read_entry_text_sources([sidecar], forms_path))
    with sidecar.open(newline='') as stream:
        source_blocks = list(csv.DictReader(stream))
    ids = {r.entry_key: r.id for r in parsed}
    assert len(blocks) == 509
    for block, source_block in zip(blocks, source_blocks):
        assert block[0] == ids[source_block['Entry_Key']]
        assert block[4] == source_block['Content']
        assert ':variant:' not in source_block['Entry_Key']
        assert ':inflection:' not in source_block['Entry_Key']
    for name in ['20260921-bhattacharya-ollari-entry-texts.csv', 'prose-row-audit.jsonl', 'prose-preview-summary.json']:
        assert (output / name).read_bytes() == (RAW / 'preview' / name).read_bytes()
    integrated = output / 'integrated'
    integration = runpy.run_path(str(RAW / 'integration_preview.py'))
    combined = integration['build'](integrated)
    assert combined['variants'] == 124 and combined['rows'] == 880
    assert combined['prose']['passages'] == 509
    assert combined['reviewed_unexpanded_suffix_scopes'] == combined['pending_suffix_scopes'] == 2
    final, stats = make_cldf.parse_file(str(integrated / '20260921-bhattacharya-ollari.csv'), errors)
    assert len(final) == 880 and errors.getvalue() == ''
    final_keys = {r.entry_key: r for r in final}
    child = final_keys['ollari1957:p69:c1:y3359']
    parent = final_keys['ollari1957:p66:c1:y3821']
    assert child.variant_of_key == parent.entry_key
    assert child.gloss == parent.gloss == 'to sing'
    assert not child.borrowed_from_key and not child.derivation_parent_keys
    plural = final_keys['ollari1957:p54:c1:y1900']
    assert plural.derivation_parent_keys == 'ollari1957:p54:c1:y867'
    assert 'pl' in plural.tags.split() and plural.gloss == 'they'
    assert not plural.variant_of_key and combined['derivations'] == 111
    for name in ['20260921-bhattacharya-ollari.csv', 'rich-row-audit.jsonl', 'rich-preview-summary.json', 'physical-record-audit.jsonl']:
        assert (integrated / name).read_bytes() == (RAW / 'integration-preview' / name).read_bytes()


def test_morphology_dispositions_require_complete_current_evidence(monkeypatch):
    import copy
    import runpy
    import pytest
    monkeypatch.syspath_prepend(str(RAW))
    review = runpy.run_path(str(RAW / 'integration_preview.py'))['review_morphology']
    audits = [json.loads(line) for line in (RAW / 'preview/physical-record-audit.jsonl').read_text().splitlines()]
    decisions = json.loads((RAW / 'morphology-dispositions.json').read_text())['decisions']
    assert review(audits, decisions) == 2
    assert sum(len(a['morphology_dispositions']) for a in audits) == 2
    with pytest.raises(ValueError, match='Unreviewed morphology'):
        review(audits, decisions[:1])
    changed = copy.deepcopy(decisions)
    changed[0]['candidate_bases'] = ['aya']
    with pytest.raises(ValueError, match='Morphology evidence changed'):
        review(audits, changed)


def test_reviewed_relation_rejects_stale_or_conflicting_evidence(monkeypatch):
    import copy
    import runpy
    import pytest
    monkeypatch.syspath_prepend(str(RAW))
    apply = runpy.run_path(str(RAW / 'integration_preview.py'))['apply_relations']
    relation = json.loads((RAW / 'reviewed-relations.json').read_text())['relations'][0]
    with (RAW / 'preview/20260921-bhattacharya-ollari.csv').open(newline='') as stream:
        rows = list(csv.reader(stream))
    evidence = {(relation['child_key'], relation['evidence_position']): relation['evidence_text']}
    for column, value in [(2, 'different'), (3, 'different meaning'), (11, relation['parent_key'])]:
        bad = copy.deepcopy(rows)
        next(r for r in bad if r[10] == relation['child_key'])[column] = value
        with pytest.raises(ValueError):
            apply(bad, [relation], evidence)
    with pytest.raises(ValueError, match='Missing relation endpoint'):
        apply([r for r in rows if r[10] != relation['parent_key']], [relation], evidence)
    with pytest.raises(ValueError, match='source evidence changed'):
        apply(copy.deepcopy(rows), [relation], {next(iter(evidence)): 'see pār-'})


def test_seeded_lexical_audit_is_reproducible_and_bound_to_preview():
    import runpy
    sampler = runpy.run_path(str(RAW / 'sample_audit.py'))['sample']
    sample_path = RAW / 'audits/lexical-sample-2026092101.json'
    report = json.loads(sample_path.read_text())
    current = sampler(2026092101, 20)
    correction = json.loads((RAW / 'audits/headword-correction-2026092102.json').read_text())
    assert all(r['source_record']['entry_key'] != correction['entry_key'] for r in report['records'])
    # Preserve the historical review, while explicitly accounting for the later
    # correction outside that sample rather than rewriting its original evidence.
    assert report['input_sha256']['reviewed-p71.json'] == correction['before_sha256']['reviewed-p71.json']
    assert current['input_sha256']['reviewed-p71.json'] == correction['after_sha256']['reviewed-p71.json']
    assert report['preview_sha256'] == correction['before_sha256']['preview/rich-row-audit.jsonl']
    later = json.loads((RAW / 'audits/corrections-2026092103.json').read_text())
    assert later['before_preview_sha256'] == correction['after_sha256']['preview/rich-row-audit.jsonl']
    assert current['preview_sha256'] == later['after_preview_sha256']
    current['input_sha256']['reviewed-p71.json'] = report['input_sha256']['reviewed-p71.json']
    current['preview_sha256'] = report['preview_sha256']
    assert current == report
    results = json.loads((RAW / 'audits/lexical-results-2026092101.json').read_text())
    assert hashlib.sha256(sample_path.read_bytes()).hexdigest() == results['sample_sha256']
    assert results['physical_records_reviewed'] == len(report['records']) == 20
    assert results['preview_rows_reviewed'] == sum(len(r['preview_rows']) for r in report['records']) == 27
    assert [r['entry_key'] for r in results['results']] == [r['source_record']['entry_key'] for r in report['records']]
    assert results['status'] == 'lexical-sample-reviewed; not final ingestion audit'


def test_integrated_audit_retains_failure_and_corrects_headword_length():
    report_path = RAW / 'audits/integration-sample-2026092102.json'
    report = json.loads(report_path.read_text())
    results = json.loads((RAW / 'audits/integration-results-2026092102.json').read_text())
    assert results['sample_sha256'] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert results['physical_records_reviewed'] == len(report['records']) == 20
    assert results['material_errors'] == 1
    correction = json.loads((RAW / 'audits/headword-correction-2026092102.json').read_text())
    failed = [r for r in results['results'] if r['result'] == 'material-error']
    assert [r['entry_key'] for r in failed] == [correction['entry_key']]
    source = next(r for r in json.loads((RAW / 'reviewed-p71.json').read_text()) if r['entry_key'] == correction['entry_key'])
    assert source['transcribed_form'] == 'mutam sīkaṭ'
    assert source['cross_reference']['target_form'] == 'sikaṭ'
    with (RAW / 'integration-preview/20260921-bhattacharya-ollari.csv').open(newline='') as stream:
        row = next(r for r in csv.reader(stream) if r[10] == correction['entry_key'])
    assert row[2:4] == ['mutam sīkaṭ', 'fog']
    assert not any(row[11:14])


def test_bibliography_review_preserves_secondary_sources_and_discrepancies():
    from pybtex.database import parse_file
    ledger = json.loads((RAW / 'bibliography-review.json').read_text())
    assert len(ledger['entries']) == 13
    assert all(r['printed_page'] == 78 and r['pdf_page'] == 89 for r in ledger['entries'])
    assert ledger['entries'][6]['author_as_printed'] == 'Gundert, F.'
    assert ledger['entries'][6]['resolved_preview_keys'] == ['keed-m']
    assert ledger['entries'][7]['existing_registry_candidate'] == 'Kitt-Kannada'
    assert ledger['entries'][9]['existing_registry_candidate'] == 'Tr-Gondi'
    assert ledger['entries'][12]['year_as_printed'] == '1924–39'
    assert ledger['entries'][12]['resolved_preview_keys'] == ['madras-tamil-lexicon']
    assert ledger['entries'][2]['year_as_printed'] == '1909; 1936'
    assert ledger['entries'][2]['resolved_preview_keys'] == ['bray1909brahui', 'bray1934brahui']
    bib = parse_file(str(RAW / 'source-reference.bib'))
    assert set(bib.entries) == {'bhattacharya1957ollari'}
    source = bib.entries['bhattacharya1957ollari']
    assert source.fields['year'] == '1957'
    assert '1956' in source.fields['note']
    assert source.fields['ocr'] == 'Yes'
    assert source.fields['included'].startswith('Preview only, not installed')
    from pybtex import PybtexEngine
    registry = parse_file(str(RAW.parents[4] / 'cldf/sources.bib')).entries
    auxiliary = parse_file(str(RAW / 'auxiliary-references.bib')).entries
    bibliography_only = parse_file(str(RAW / 'bibliography-only-references.bib')).entries
    assert len(bibliography_only) == 9
    assert not (set(bibliography_only) & set(auxiliary))
    for key, entry in bibliography_only.items():
        if key in registry:
            assert registry[key] == entry
    available = set(registry) | set(auxiliary) | set(bibliography_only)
    for item in ledger['entries']:
        assert set(item['resolved_preview_keys']) <= available
    for entry in [*bib.entries.values(), *auxiliary.values(), *bibliography_only.values()]:
        formatted = PybtexEngine().format_from_string(entry.to_string('bibtex'), 'plain', output_backend='markdown')
        assert formatted.strip() and entry.fields['year'].replace('--', '–') in formatted


def test_second_integration_audit_fixes_diacritics_usage_and_gender():
    sample_path = RAW / 'audits/integration-sample-2026092103.json'
    sample = json.loads(sample_path.read_text())
    results = json.loads((RAW / 'audits/integration-results-2026092103.json').read_text())
    assert results['sample_sha256'] == hashlib.sha256(sample_path.read_bytes()).hexdigest()
    assert len(sample['records']) == results['physical_records_reviewed'] == 20
    assert results['material_errors'] == sum(r['result'] == 'material-error' for r in results['results']) == 5
    audit = json.loads((RAW / 'audits/corrections-2026092103.json').read_text())
    current = {r['entry_key']: r for f in RAW.glob('comparison-reviewed-p*.json')
               for r in json.loads(f.read_text())['records']}
    assert len(audit['corrections']) == 5
    for correction in audit['corrections']:
        assert current[correction['entry_key']] == correction['after']
        assert correction['before'] != correction['after']
    with (RAW / 'integration-preview/20260921-bhattacharya-ollari.csv').open(newline='') as stream:
        row = next(r for r in csv.reader(stream) if r[10] == 'ollari1957:p56:c1:y850')
    assert row[3] == 'blind' and 'mn' in row[14].split()
    for key in ['ollari1957:p73:c2:y2167', 'ollari1957:p74:c2:y3080']:
        assert current[key]['passages'][0] == {'position': 1, 'kind': 'usage', 'text': 'usually pl.'}


def test_wider_comparison_review_is_bound_to_current_pages():
    audit = json.loads((RAW / 'audits/comparative-second-pass.json').read_text())
    pages = [r['printed_page'] for r in audit['reviewed_pages']]
    assert sorted(pages + audit['remaining_pages']) == list(range(48, 78))
    for page in audit['reviewed_pages']:
        path = RAW / f"comparison-reviewed-p{page['printed_page']}.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == page['after_sha256']
        records = json.loads(path.read_text())['records']
        assert [r['entry_key'] for r in records] == page['entry_keys']
        assert sum(len(r['passages']) for r in records) == page['passages']
    for correction in audit['corrections']:
        assert correction['before'] != correction['after']
        page_number = int(correction['entry_key'].split(':')[1][1:])
        records = json.loads((RAW / f'comparison-reviewed-p{page_number}.json').read_text())['records']
        assert next(r for r in records if r['entry_key'] == correction['entry_key']) == correction['after']
    corrected = {r['entry_key']: r['after'] for r in audit['corrections']}
    assert 'Tu. tumbilụ,' in corrected['ollari1957:p62:c1:y3759']['passages'][0]['text']
    assert 'Ta. acanku ' in corrected['ollari1957:p48:c2:y3871']['passages'][0]['text']
    assert corrected['ollari1957:p51:c1:y567']['passages'][1]['text'] == 'see oguṛe and kuṛve below'
    assert 'Hindī kəbāṛi' in corrected['ollari1957:p54:c2:y4105']['passages'][0]['text']
    daughter_in_law = corrected['ollari1957:p59:c1:y3746']['passages'][0]['text']
    for form in ['Naik. koraḷ, Kol. koral', 'Kui kōṛu', 'Go. A. koṛs-']:
        assert form in daughter_in_law


def test_fresh_acceptance_audit_and_last_comparison_corrections():
    sample_path = RAW / 'audits/integration-sample-2026092105.json'
    sample = json.loads(sample_path.read_text())
    results = json.loads((RAW / 'audits/integration-results-2026092105.json').read_text())
    assert results['sample_sha256'] == hashlib.sha256(sample_path.read_bytes()).hexdigest()
    assert results['physical_records_reviewed'] == len(sample['records']) == 20
    assert results['preview_rows_reviewed'] == 24
    assert results['prose_passages_reviewed'] == 16
    assert results['material_errors'] == 0
    assert all(r['result'] == 'pass' for r in results['results'])
    for path, digest in sample['input_sha256'].items():
        assert hashlib.sha256((RAW / path).read_bytes()).hexdigest() == digest
    corrections = json.loads((RAW / 'audits/corrections-2026092104.json').read_text())
    current = {r['entry_key']: r for f in RAW.glob('comparison-reviewed-p*.json')
               for r in json.loads(f.read_text())['records']}
    for correction in corrections['corrections']:
        assert current[correction['entry_key']] == correction['after']
    assert 'Burushaski mɛl' in current['ollari1957:p71:c1:y3634']['passages'][0]['text']
    assert 'Brah. khīsun' in current['ollari1957:p60:c2:y2009']['passages'][0]['text']


def test_installed_ollari_matches_audited_outputs():
    import yaml
    root = RAW.parents[4]
    stem = '20260921-bhattacharya-ollari'
    forms = root / 'data/other/forms'
    assert (forms / f'{stem}.csv').read_bytes() == (RAW / 'integration-preview' / f'{stem}.csv').read_bytes()
    assert (root / 'data/other/entry_texts' / f'{stem}.csv').read_bytes() == (RAW / 'integration-preview' / f'{stem}-entry-texts.csv').read_bytes()
    settings = yaml.safe_load((forms / f'{stem}.yaml').read_text())
    assert settings['defaults']['identity']['append_order'] == 32
    assert settings['defaults']['transcription']['input'] == 'phonemic'
    assert settings['sources']['bhattacharya1957ollari']['identity']['dedupe_by_entry_key']
