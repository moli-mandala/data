"""Preserve original evidence and review boundaries during whole-thesis recovery."""
import hashlib
import json
import importlib.util
import csv
from pathlib import Path

P = Path(__file__).resolve().parents[1] / 'data/other/forms/raw_data/dhakal_darai_2011'


def load(name):
    return [json.loads(line) for line in (P / name).read_text().splitlines()]


def test_layout_eligibility_preserves_ordinary_font_target_words():
    old = {r['source_unit_key']: r for r in load('chapter4-interlinear-alignment-candidates.jsonl')}
    new = {r['source_unit_key']: r for r in load('chapter4-interlinear-layout-v2-candidates.jsonl')}
    assert old.keys() <= new.keys()
    assert {(new[k]['form'], new[k]['candidate_gloss']) for k in new.keys() - old.keys()} == {
        ('kami-se', 'ironsmith-ABL'), ('mana', 'a unit'), ('caur', 'rice')}
    # Eligibility changes must not silently rewrite any previously extracted form.
    assert all(new[k]['form'] == row['form'] for k, row in old.items())
    assert new['dhakal2011:p110:ch4:surface:l46:word:l49:w3']['candidate_gloss'] == 'one'
    assert new['dhakal2011:p105:ch4:surface:l5:word:l10:w1']['candidate_gloss'] == 'some'


def test_review_keeps_printed_ambiguity_instead_of_inventing_alignment():
    rows = load('chapter4-interlinear-adjudicated-candidates.jsonl')
    group = next(r for r in rows if r['form'] == 'kʰəna tə u-rə')
    assert group['candidate_gloss'] == 'PART he-[OBL]GEN'
    assert group['needs_manual_alignment'] and len(group['source_candidate_keys']) == 3
    assert 'do not invent' in group['uncertainty']
    assert not any(r['form'] == 'tə' and r['parent_expression_key'] == group['parent_expression_key'] for r in rows)
    nasal = next(r for r in rows if r['form'] == 'ṭʰaũ-jə̃')
    assert nasal['printed_page'] == 108
    discrepancy = next(r for r in rows if r.get('source_semantic_discrepancy'))
    assert discrepancy['form'] == 'məntinne' and discrepancy['candidate_gloss'] == 'knowledge'
    assert any(r['review_status'].startswith('Unreviewed') for r in rows)


def test_reviewed_derivative_preserves_all_first_reading_keys():
    original = sum((load(f'chapter4-{kind}-first-reading.jsonl') for kind in ('expressions', 'tables', 'prose')), [])
    reviewed = load('chapter4-adjudicated-records.jsonl')
    assert len(original) == len(reviewed) == 624
    assert {r['source_unit_key'] for r in original} == {r['source_unit_key'] for r in reviewed}
    report = json.loads((P / 'chapter4-adjudication-reconciliation-20260927.json').read_text())
    assert report['output_sha256'] == hashlib.sha256((P / 'chapter4-adjudicated-records.jsonl').read_bytes()).hexdigest()


def test_complete_interlinear_review_preserves_source_conflicts_and_glyph_positions():
    rows = {r['source_unit_key']: r for r in load('chapter4-interlinear-reviewed.jsonl')}
    assert len(rows) == 704
    assert rows['dhakal2011:p120:ch4:surface:l21:word:l23:w1']['form'] == 'kaṭ-ikun'
    assert rows['dhakal2011:p123:ch4:surface:l79:word:l81:w1']['form'] == 'pəile-hĩ-se'
    group = rows['dhakal2011:p78:ch4:surface:l93:word:l100:w1']
    assert group['form'] == 'u-ke apʰnə' and group['candidate_gloss'] == 'he-[OBL] DATself'
    assert len(group['complete_review_input']) == 2 and group['needs_manual_alignment']
    conflict = rows['dhakal2011:p99:ch4:surface:l31:word:l33:w1']
    assert conflict['candidate_gloss'] == 'we'
    assert '1pl' in conflict['source_interpretation_uncertainty']
    assert all('independent review pending' in r['review_status'] for r in rows.values())


def test_interlinear_labels_do_not_erase_lexical_hyphens_or_resolve_source_conflicts():
    spec = importlib.util.spec_from_file_location('darai_interlinear_grammar', P/'interlinear_grammar.py')
    grammar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grammar)
    assert grammar.parse('daughter-in-law-GEN')['gloss'] == 'daughter-in-law'
    assert grammar.parse('I-[OBL]DAT')['gloss'] == 'I'
    assert grammar.parse('all-day')['gloss'] == 'all-day'
    assert grammar.parse('COP.PST-3.SG')['tags'] == ['copula','pret','3sg','verb']
    conflict = grammar.parse('remain-3SG-NPST-1PL', conflicting=True)
    assert conflict['gloss'] == 'remain' and conflict['labels'] == ['3SG','NPST','1PL']
    assert not {'3sg','1pl'} & set(conflict['tags'])
    assert grammar.parse('feel-PVF')['unmapped_source_labels'] == ['PVF']
    assert grammar.parse('he-[OBL] DATself', ambiguous=True)['tags'] == []


def test_chapter5_original_review_preserves_controls_ambiguity_and_column_alignment():
    inputs = load('chapter5-first-reading.jsonl') + load('chapter5-interlinear-layout-v2-candidates.jsonl')
    reviewed = load('chapter5-reviewed-records.jsonl')
    assert {r['source_unit_key'] for r in inputs} == {
        key for r in reviewed for key in r.get('source_candidate_keys', [r['source_unit_key']])}
    keyed = {r['source_unit_key']: r for r in reviewed}
    assert sum(r.get('disposition') == 'English-gloss-control' for r in reviewed) == 6
    base = 'dhakal2011:p144:ch5:surface:l41:word:l42:'
    assert [keyed[base+f'w{i}']['candidate_gloss'] for i in range(1,4)] == ['bathe-SIM','while','one-CLF']
    unknown = keyed['dhakal2011:p141:ch5:surface:l76:word:l78:w2']
    assert len(unknown['review_input']) == 2
    assert 'F060' in unknown['form'] and unknown['proposed_phonemic'] == ''
    assert 'encoding_uncertainty' in unknown
    assert keyed['dhakal2011:p141:ch5:surface:l60:word:l67:w2']['form'] == 'cʰə̃uḍi'
    assert '⟦' not in keyed['dhakal2011:p141:ch5:surface:l60']['form']


def test_chapter4_metadata_preserves_lect_and_reviewed_uncertainty_boundaries():
    rows = {r['source_unit_key']:r for r in load('chapter4-metadata-proposal.jsonl')}
    assert all(r.get('lect') for r in rows.values() if r['disposition']=='proposed-attestation')
    tanahun = rows['dhakal2011:p90:ch4:prose:l88:a1']
    assert tanahun['lect']=='Tanahun' and '2pl' in tanahun['proposed_grammar']['tags']
    assert 'excludes it from Chitwan' in tanahun['proposed_notes']
    assert rows['dhakal2011:p113:ch4:prose:l36:a1']['disposition']=='audit-only'
    nasal = rows['dhakal2011:p97:ch4:prose:l62:a1']
    assert nasal['form']=='-ĩ' and 'uncertain' not in nasal['proposed_grammar']['tags']
    shifted = rows['dhakal2011:p92:ch4:prose:l31:a1']
    assert shifted['form']=='-otạ' and 'uncertain' in shifted['proposed_grammar']['tags']
    assert 'under' in shifted['proposed_notes']
    assert all('awaits original' not in r.get('proposed_notes','') for r in rows.values())


def test_chapter4_review_export_preserves_uncertain_literal_without_phonemic_claim():
    with (P/'chapter4-proposed.csv').open(newline='') as handle:
        rows = {row[10]: row for row in csv.reader(handle)}
    assert len(rows) == 1282
    shifted = rows['dhakal2011:p92:ch4:prose:l31:a1']
    assert shifted[2]=='-otạ' and shifted[5]=='' and 'uncertain' in shifted[14]
    accented = rows['dhakal2011:p78:ch4:surface:l43:word:l50:w1']
    assert accented[2]=='u-hí̽' and accented[5]==''
    assert 'dhakal2011:p113:ch4:prose:l36:a1' not in rows
    assert rows['dhakal2011:p90:ch4:prose:l88:a1'][14].startswith('dialect:Darai:darai_tanahun:Tanahun;')


def test_chapter6_review_preserves_missing_gloss_controls_and_split_word_evidence():
    inputs = load('chapter6-first-reading.jsonl') + load('chapter6-interlinear-layout-v2-candidates.jsonl')
    reviewed = load('chapter6-reviewed-records.jsonl')
    assert {r['source_unit_key'] for r in inputs} == {
        key for r in reviewed for key in r.get('source_candidate_keys', [r['source_unit_key']])}
    keyed = {r['source_unit_key']: r for r in reviewed}
    assert len(reviewed) == 1523
    missing = keyed['dhakal2011:p181:ch6:surface:l41:word:l46:w1']
    assert missing['form'] == 'məi' and missing['candidate_gloss'] == ''
    assert missing['review_input']
    group = keyed['dhakal2011:p185:ch6:surface:l7:word:l15:w1']
    assert group['form'] == 'detaĩ -kan-səb'
    assert len(group['source_candidate_keys']) == len(group['review_input']) == 2
    assert sum(r.get('disposition') == 'English-gloss-control' for r in reviewed) == 6
    assert keyed['dhakal2011:p188:ch6:surface:l9:word:l10:w1']['disposition'] == 'unacceptable-source-control'
    assert keyed['dhakal2011:p174:ch6:surface:l11']['gloss'] == 'We eat.'
    assert keyed['dhakal2011:p174:ch6:surface:l11']['source_reference'] == 'Darai 1996:38'


def test_chapter7_review_separates_gloss_columns_without_regularizing_source_errors():
    rows = {r['source_unit_key']: r for r in load('chapter7-reviewed-records.jsonl')}
    assert len(rows) == 441
    base = 'dhakal2011:p214:ch7:surface:l53:word:l55:'
    assert rows[base+'w1']['candidate_gloss'] == 'nearly'
    assert rows[base+'w2']['candidate_gloss'] == 'die-PST'
    error = rows['dhakal2011:p213:ch7:surface:l29:word:l42:w1']
    assert error['form'] == 'ete' and error['candidate_gloss'] == 'come-IMP.H'
    assert error['source_interpretation_uncertainty']
    glyph = rows['dhakal2011:p210:ch7:list:l19']
    assert 'F032' in glyph['form'] and glyph['proposed_phonemic'] == ''
    assert len([r for r in rows.values() if r.get('kind') == 'pronominal-adverb-table']) == 16
    assert rows['dhakal2011:p215:ch7:table:l94']['source_role'] == 'relative'


def test_chapter8_review_preserves_particle_restrictions_and_repairs_gloss_columns():
    rows = {r['source_unit_key']: r for r in load('chapter8-reviewed-records.jsonl')}
    inputs = load('chapter8-first-reading.jsonl') + load('chapter8-interlinear-layout-v2-candidates.jsonl')
    assert set(rows) == {r['source_unit_key'] for r in inputs}
    assert len(rows) == 332
    assert rows['dhakal2011:p232:ch8:surface:l22:word:l23:w1']['candidate_gloss'] == 'EXCL'
    assert rows['dhakal2011:p232:ch8:surface:l22:word:l24:w1']['candidate_gloss'] == 'big'
    assert rows['dhakal2011:p229:ch8:surface:l42:word:l46:w1']['candidate_gloss'] == 'SIM'
    for suffix in ('a1', 'a2'):
        assert 'between two nouns or NPs' in rows['dhakal2011:p224:ch8:prose:l6:'+suffix]['note']
    assert rows['dhakal2011:p225:ch8:prose:l85:a1']['disposition'] == 'foreign-comparison-control'
    assert rows['dhakal2011:p228:ch8:surface:l48:word:l49:w1']['source_interpretation_uncertainty']


def test_chapter9_review_preserves_blank_gloss_and_source_semantic_conflict():
    rows = {r['source_unit_key']: r for r in load('chapter9-reviewed-records.jsonl')}
    assert len(rows) == 248
    assert rows['dhakal2011:p239:ch9:surface:l20:word:l23:w1']['candidate_gloss'] == ''
    conflict = rows['dhakal2011:p241:ch9:surface:l19:word:l22:w1']
    assert conflict['form'] == 'bʰai-m' and conflict['candidate_gloss'] == 'buffalo-POSS1SG'
    assert conflict['source_interpretation_uncertainty']
    nasal = rows['dhakal2011:p242:ch9:surface:l47:word:l49:w1']
    assert nasal['form'] == 'pəile-hĩ-se' and nasal['review_input'][0]['form'] == 'pəile-hi-̃se'
    assert rows['dhakal2011:p241:ch9:prose:l63:a1']['form'] == '-kə'


def test_chapter10_review_preserves_all_evidence_and_rejects_starred_constituents():
    reviewed = load('chapter10-reviewed-records.jsonl')
    inputs = load('chapter10-first-reading.jsonl') + load('chapter10-interlinear-layout-v2-candidates.jsonl')
    assert len(reviewed) == 875
    assert {r['source_unit_key'] for r in inputs} == {
        key for r in reviewed for key in r.get('source_candidate_keys', [r['source_unit_key']])}
    rows = {r['source_unit_key']: r for r in reviewed}
    for key in ['dhakal2011:p282:ch10:surface:l12',
                'dhakal2011:p282:ch10:surface:l12:word:l13:w1',
                'dhakal2011:p282:ch10:surface:l12:word:l14:w1']:
        assert rows[key]['disposition'] == 'unacceptable-source-control'
    grouped = rows['dhakal2011:p274:ch10:surface:l25:word:l28:w1']
    assert grouped['form'] == 'nai kər-ihĩ'
    assert grouped['candidate_gloss'] == 'NEG-do-1PL'
    assert len(grouped['source_candidate_keys']) == len(grouped['review_input']) == 2
    assert rows['dhakal2011:p283:ch10:surface:l5:word:l7:w2']['disposition'] == 'foreign-language-quotation-control'


def test_chapter10_review_repairs_columns_but_retains_printed_analysis_conflicts():
    rows = {r['source_unit_key']: r for r in load('chapter10-reviewed-records.jsonl')}
    assert rows['dhakal2011:p283:ch10:surface:l5:word:l7:w1']['candidate_gloss'] == 'language-LOC'
    assert rows['dhakal2011:p283:ch10:surface:l5:word:l7:w3']['candidate_gloss'] == 'we-[OBL]GEN'
    assert rows['dhakal2011:p283:ch10:surface:l20:word:l23:w1']['candidate_gloss'] == 'and'
    disputed = rows['dhakal2011:p268:ch10:surface:l44:word:l45:w1']
    assert disputed['form'] == 'dza-tə-m' and disputed['candidate_gloss'] == 'NEG-go-1SG'
    assert disputed['source_interpretation_uncertainty']
    assert rows['dhakal2011:p273:ch10:surface:l16']['source_interpretation_uncertainty']
    assert rows['dhakal2011:p278:ch10:surface:l48:word:l49:w2']['candidate_gloss'] == 'bathe-PST'


def test_chapter11_recovery_accounts_for_layout_controls_merges_and_splits():
    raw = load('chapter11-first-reading.jsonl') + load('chapter11-interlinear-layout-v2-candidates.jsonl')
    reviewed = load('chapter11-reviewed-records.jsonl')
    assert len(raw) == 944 and len(reviewed) == 943
    assert {r['source_unit_key'] for r in raw} == {
        key for r in reviewed for key in r.get('source_candidate_keys', [r['source_unit_key']])}
    rows = {r['source_unit_key']: r for r in reviewed}
    base = 'dhakal2011:p306:ch11:surface:l34:word:'
    assert rows[base+'l43:w2']['form'] == 'kər-lə'
    assert rows[base+'l43:w2']['candidate_gloss'] == 'do-PST'
    assert sum(r.get('disposition') == 'non-Darai-English-gloss-control' for r in reviewed) == 6
    graze = rows['dhakal2011:p311:ch11:surface:l6:word:l9:w1']
    assert graze['form'] == 'cəri kʰər-ike' and graze['candidate_gloss'] == 'graze-INF'
    assert len(graze['source_candidate_keys']) == 2
    split = 'dhakal2011:p316:ch11:surface:l49:word:l51:w1'
    assert [(rows[split+f':part{i}']['form'], rows[split+f':part{i}']['candidate_gloss']) for i in (1,2)] == [
        ('məhina','time'), ('bʰəi-ne','become-COND')]


def test_chapter11_preserves_source_conflicts_private_glyph_and_foreign_scope():
    rows = {r['source_unit_key']: r for r in load('chapter11-reviewed-records.jsonl')}
    private = rows['dhakal2011:p320:ch11:surface:l12']
    assert 'U+F03E' in private['form'] and private['proposed_phonemic'] == ''
    assert 'interlinear ŋ' in private['review_decisions'][0]['decision']
    foreign = rows['dhakal2011:p297:ch11:surface:l75:word:l77:w1']
    assert foreign['form'] == 'nuwarən' and foreign['disposition'] == 'foreign-language-quotation-control'
    boat = rows['dhakal2011:p314:ch11:surface:l53:word:l54:w1']
    assert boat['form'] == 'məi' and boat['candidate_gloss'] == 'boat'
    assert boat['source_interpretation_uncertainty']
    omitted = rows['dhakal2011:p297:ch11:surface:l52:word:l56:w2']
    assert omitted['form'] == 'ki' and omitted['candidate_gloss'] == ''


def test_chapter12_review_preserves_all_candidates_and_translation_scopes():
    raw = load('chapter12-first-reading.jsonl') + load('chapter12-interlinear-layout-v2-candidates.jsonl')
    reviewed = load('chapter12-reviewed-records.jsonl')
    assert len(raw) == 386 and len(reviewed) == 385
    assert {r['source_unit_key'] for r in raw} == {
        key for r in reviewed for key in r.get('source_candidate_keys', [r['source_unit_key']])}
    rows = {r['source_unit_key']: r for r in reviewed}
    proverb = rows['dhakal2011:p327:ch12:surface:l28']
    assert len(proverb['source_translation_groups']) == 2
    assert 'indecisive' in proverb['source_translation_groups'][1]['raw']
    riddle = rows['dhakal2011:p329:ch12:surface:l21']
    assert 'axe' in riddle['source_solution']['raw'] and 'axe' not in riddle['gloss']
    queen = rows['dhakal2011:p329:ch12:surface:l41:word:l60:w1']
    assert queen['form'] == 'cəmpa rani' and len(queen['source_candidate_keys']) == 2
    assert sum(r.get('disposition') == 'non-Darai-English-gloss-control' for r in reviewed) == 10
    assert rows['dhakal2011:p343:ch12:surface:l58:word:l59:w2']['candidate_gloss'] == 'he'


def test_chapter12_retains_original_semantic_conflicts_and_nasal_marks():
    rows = {r['source_unit_key']: r for r in load('chapter12-reviewed-records.jsonl')}
    for key, gloss in [
        ('dhakal2011:p339:ch12:surface:l6:word:l9:w1', 'towards the evening'),
        ('dhakal2011:p341:ch12:surface:l44:word:l47:w1', 'next-ERG')]:
        assert rows[key]['candidate_gloss'] == gloss
        assert rows[key]['source_interpretation_uncertainty']
    nasal = rows['dhakal2011:p328:ch12:surface:l56:word:l58:w1']
    assert nasal['form'] == 'ṭʰaũ' and 'U+F0E2' in nasal['review_input'][0]['form']
    assert rows['dhakal2011:p339:ch12:surface:l65:word:l77:w2']['form'] == 'dʰuk-lə'
    assert rows['dhakal2011:p345:ch12:surface:l39:word:l49:w1']['form'] == 'ḍʰuk-lə'


def test_appendix_paradigms_preserve_cross_page_cells_and_literal_variation():
    a = {r['source_unit_key']: r for r in load('appendix3a-reviewed-records.jsonl')}
    b = {r['source_unit_key']: r for r in load('appendix3b-reviewed-records.jsonl')}
    assert len(a) == 113 and len(b) == 187
    assert a['dhakal2011:p367:appendix3a:l115']['form'] == 'ro-larə'
    assert a['dhakal2011:p367:appendix3a:l111']['gloss'] == 'cry'
    assert b['dhakal2011:p368:appendix3b:l273:root']['gloss'] == 'wash'
    continued = b['dhakal2011:p368:appendix3b:l322:col7']
    assert continued['form'] == 'kʰasa-i-t'
    assert continued['source_locations'] == [
        {'printed_page': 368, 'line': 322}, {'printed_page': 369, 'line': 13}]
    assert continued['source_grammar'] == 'non-past 3SG'
    unusual = b['dhakal2011:p368:appendix3b:l190:col10']
    assert unusual['form'] == 'əi-ta--hat(-səb)' and unusual['transcription_uncertainty']


def test_appendix_agreement_roles_and_absence_are_not_invented_forms():
    rows = load('appendix3c1-table-reviewed-records.jsonl')
    absent = [r for r in rows if r['disposition'] == 'source-absence-control']
    assert len(rows) == 126 and len(absent) == 78
    assert all(r['form'] == r['gloss'] == '' for r in absent)
    dual = next(r for r in rows if r['source_subject'] == '1SG' and
                r['source_object'] == '2SG' and r['source_tense'] == 'PST')
    assert dual['form'] == 'de-hal-mi-r(s)'
    polite = next(r for r in rows if r['source_subject'] == '2SG.H' and
                  r['source_object'] == '3SG' and r['source_tense'] == 'PST')
    assert polite['form'] == 'de-hale-u-k'


def test_text_appendix_font_and_wrapped_column_repairs_preserve_targets():
    rows = {r['source_unit_key']: r for r in load('appendix5-reviewed-records.jsonl')}
    prefix = 'dhakal2011:appendix5:'
    assert rows[prefix + 'HtMW.SD.013']['form'].endswith('bənit.')
    become = rows[prefix + 'HtMW.SD.013:word:p398:l79:w1']
    assert (become['form'], become['candidate_gloss']) == ('bən-i-t', 'become -3SG -NPST')
    conditional = rows[prefix + 'HtMW.SD.013:word:p398:l57:w1']
    assert (conditional['form'], conditional['candidate_gloss']) == ('rakʰ-ne', 'put -COND')
    assert rows[prefix + 'HtMW.SD.022']['gloss'] == 'I tell (you) how the beer is made.'
    aux = rows[prefix + 'HtMW.SD.022:word:p400:l52:w1']
    assert (aux['form'], aux['candidate_gloss']) == ('aṭə-m', 'AUX.NPST -1SG')
    initial = rows[prefix + 'HtMW.SD.024:word:p400:l97:w1']
    final = rows[prefix + 'HtMW.SD.024:word:p400:l102:w2']
    assert (initial['form'], initial['candidate_gloss']) == ('makai-ke', 'maize -DAT')
    assert (final['form'], final['candidate_gloss']) == ('pis-ike', 'grind -INF')
    assert {w['line'] for w in final['analysis_evidence']} == {113, 114}


def test_text_appendix_foreign_quotes_and_review_coverage_are_explicit():
    reviewed = load('appendix5-reviewed-records.jsonl')
    original = load('appendix5-text-first-reading.jsonl') + load('appendix5-interlinear-column-candidates.jsonl')
    assert {r['source_unit_key'] for r in reviewed} == {r['source_unit_key'] for r in original}
    rows = {r['source_unit_key']: r for r in reviewed}
    quoted = rows['dhakal2011:appendix5:DR.CND.001:word:p388:l12:w1']
    assert quoted['form'] == 'krija' and quoted['disposition'] == 'foreign-language-quotation-control'
    assert rows['dhakal2011:appendix5:DR.CND.001']['form']
    for row in reviewed:
        if row.get('owner_review_pages') and row.get('kind') == 'whole-expression':
            assert {line['printed_page'] for line in row['source_lines']} <= set(row['owner_review_pages'])


def test_complete_text_review_preserves_boundaries_and_source_defects():
    reviewed = load('appendix5-reviewed-records.jsonl')
    assert len(reviewed) == 2665
    assert all(r.get('owner_review_pages') for r in reviewed)
    rows = {r['source_unit_key']: r for r in reviewed}
    prefix = 'dhakal2011:appendix5:'
    assert rows[prefix + 'JaH.SLD.017:word:p425:l105:w1']['form'] == 'kʰodz-i'
    assert rows[prefix + 'JaH.SLD.017:word:p425:l105:w2']['form'] == '-laləi'
    today = rows[prefix + 'JaH.SLD.014:word:p424:l156:w1']
    assert today['form'] == 'adzu'
    assert [x['text'] for x in today['source_analysis_extra_fragments']['analysis']] == ['tə', '-s']
    assert rows[prefix + 'JaH.SLD.062:word:p434:l48:w1']['candidate_gloss'] == 'fear -EMP'
    assert rows[prefix + 'JaH.SLD.062:word:p434:l49:w1']['candidate_gloss'] == 'run away -SIM'
    assert rows[prefix + 'PtP.CND.005:word:p435:l65:w2']['form'] == 'de-ike'
    assert rows[prefix + 'PtP.CND.005:word:p435:l66:w1']['form'] == 'məi-rə'
    damaged = rows[prefix + 'PtP.CND.045:word:p443:l86:w1']
    assert damaged['candidate_gloss'] == 'ork for ot'
    assert damaged['source_analysis_uncertainty']
    wrong_analysis = rows[prefix + 'PtP.CND.063:word:p447:l25:w1']
    assert wrong_analysis['source_surface_form'] == 'dukʰ'
    assert (wrong_analysis['form'], wrong_analysis['candidate_gloss']) == ('dui', 'two')
    assert wrong_analysis['source_analysis_uncertainty']
    grouped = rows[prefix + 'In.UN.047']
    assert len(grouped['source_expression_groups']) == 2


def test_chapter3_expansion_preserves_surface_and_segmented_evidence():
    rows = load('chapter3-reviewed-expression-records.jsonl')
    words = [r for r in rows if r['kind'] == 'aligned-word']
    assert len(words) == 53
    by_key = {r['source_unit_key']: r for r in rows}
    prefix = 'dhakal2011:p73:ch3:surface:l60'
    assert by_key[prefix]['form'].split() == ['uhĩ', 'bʰaike', 'cʰiulə']
    assert by_key[prefix + ':word:l63:w1']['form'] == 'u-ĩ'
    assert by_key[prefix + ':word:l63:w1']['gloss'] == 'he-ERG'
    assert by_key['dhakal2011:p75:ch3:surface:l53:word:l59:w1']['form'] == 'kami-se'
    assert by_key['dhakal2011:p75:ch3:surface:l53:word:l59:w1']['gloss'] == 'Kami-ABL'


def test_whole_index_retains_all_installed_keys_and_exact_index_targets():
    rows = load('whole-evidence-index.jsonl')
    by_key = {r['source_unit_key']: r for r in rows}
    assert len(by_key) == len(rows)
    with (P / 'legacy-before-whole-recovery.csv').open(newline='') as stream:
        canonical = list(csv.reader(stream))
    assert len(canonical) == 485
    assert {r[10] for r in canonical} <= by_key.keys()
    for row in load('front-index-attestations-reviewed.jsonl'):
        target = by_key[row['reuse_source_unit_key']]
        assert (target['form'], target['printed_page']) == (row['form'], row['source_crossreference_page'])
    # Source controls remain available for audit; a blank form must not vanish.
    assert any(r['kind'] == 'paradigm-absence-control' and not r['form'] for r in rows)


def test_later_chapter_labels_remain_metadata_not_english_gloss():
    spec = importlib.util.spec_from_file_location('darai_grammar_later', P / 'interlinear_grammar.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.parse('go-PRF')['tags'] == ['perfect', 'verb']
    assert module.parse('sit-PROG')['gloss'] == 'sit'
    assert 'progressive' in module.parse('sit-PROG')['tags']
    held = module.parse('say-INSTAN-PVF')
    assert held['gloss'] == 'say'
    assert held['unmapped_source_labels'] == ['INSTAN', 'PVF']
    assert 'pfv' not in held['tags']  # Do not repair a printed label by resemblance.
    assert module.parse('I')['gloss'] == 'I'
    assert module.parse('I')['tags'] == []


def test_whole_notation_expansion_retains_ambiguous_groups_and_distinct_glosses():
    rows = {r['source_unit_key']: r for r in load('whole-expanded-metadata-review.jsonl')}
    base = 'dhakal2011:p140:ch5:surface:l36:word:l39:w1'
    assert rows[base]['form'] == 'ɡə̃idakot-se'
    assert rows[base]['proposed_grammar']['gloss'] == 'Gaindako'
    assert rows[base + ':alternate:2']['form'] == 'kəte'
    assert rows[base + ':alternate:2']['proposed_grammar']['gloss'] == 'than'
    base = 'dhakal2011:p370:appendix3c1:l22'
    assert [rows[k]['form'] for k in (base, base + ':alternate:2')] == ['de-ta-mi-r', 'de-ta-mi-s']
    assert rows[base]['raw_group_form'] == 'de-ta-mi-r(s)'
    held = rows['dhakal2011:p376:appendix3d:l87']
    assert held['form'] == 'kər-lə-hai/kʰ-ai-səb'
    assert held['source_notation_decision']['split_alternates'] is False
    assert held['phonemic_policy'].startswith('blank; literal')
    assert not any(k.startswith(held['source_unit_key'] + ':alternate:') for k in rows)


def test_two_statements_under_one_text_id_keep_exact_word_scope():
    rows = {r['source_unit_key']: r for r in load('whole-expanded-metadata-review.jsonl')}
    base = 'dhakal2011:appendix5:In.UN.047'
    first, second = rows[base], rows[base + ':statement:2']
    assert first['form'].endswith('baṭikanhun.')
    assert second['form'].startswith('u radzakə')
    assert first['physical_source_unit_key'] == second['physical_source_unit_key'] == base
    assert second['source_claims']['text_id'] == 'In.UN.047'
    assert second['source_claims']['speaker_turn_label'] == 'A'
    assert second['source_gloss'].startswith('A:')
    assert second['proposed_grammar']['gloss'] == 'He did not even have a child.'
    children = [r for k, r in rows.items() if k.startswith(base + ':word:')]
    assert len(children) == 16
    assert sum(r['parent_expression_key'] == base for r in children) == 8
    assert sum(r['parent_expression_key'] == base + ':statement:2' for r in children) == 8
    assert rows[base + ':word:p412:l22:w1']['parent_expression_key'] == base + ':statement:2'


def test_whole_agreement_roles_do_not_conflate_subject_object_or_case():
    rows = {r['source_unit_key']: r for r in load('whole-expanded-metadata-review.jsonl')}
    row = rows['dhakal2011:p370:appendix3c1:l22']
    assert row['source_agreement_roles']['subject']['source_label'] == '1SG'
    assert row['source_agreement_roles']['object']['source_label'] == '2SG'
    assert 'non-past' in row['proposed_grammar']['tags']
    assert not {'1sg', '2sg'} & set(row['proposed_grammar']['tags'])
    copula = rows['dhakal2011:p377:appendix3e:l29']
    assert copula['source_agreement_roles']['agreement']['alternatives'][0]['honorific']
    assert not {'nom', 'erg'} & set(copula['proposed_grammar']['tags'])


def test_conflicting_labels_remain_literal_without_confident_false_tags():
    rows = {r['source_unit_key']: r for r in load('whole-expanded-metadata-review.jsonl')}
    came = rows['dhakal2011:appendix5:In.UN.077:word:p416:l99:w1']
    assert 'CAUS' in came['source_gloss']
    assert 'pret' in came['proposed_grammar']['tags']
    assert not {'caus', 'third-person'} & set(came['proposed_grammar']['tags'])
    modal = rows['dhakal2011:p263:ch10:surface:l59:word:l64:w2']
    assert modal['source_gloss'] == 'plough-POSS-2SG'
    assert modal['proposed_grammar']['tags'] == ['2sg','uncertain']
    assert 'poss' in modal['conflicting_tag_decision']['withhold_confident_tags']
    giver = rows['dhakal2011:p373:appendix3c1:aligned-word:l28']
    assert '1sg' not in giver['proposed_grammar']['tags']
    assert '2sg' in giver['proposed_grammar']['tags']
    # A conflicting lexical definition does not erase a clear possessive person.
    sister = rows['dhakal2011:appendix5:PtP.CND.041:word:p442:l114:w1']
    assert '2sg' in sister['proposed_grammar']['tags']


def test_written_out_table_headers_are_grammar_not_lexical_glosses():
    spec = importlib.util.spec_from_file_location('darai_headers', P / 'interlinear_grammar.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.parse_source_grammar('non-past 3PL.H')['tags']) == {'non-past','3pl','honorific','verb'}
    assert set(module.parse_source_grammar('Past/perfect')['tags']) == {'pret','perfect','verb'}
    assert module.parse_source_grammar('negative prefix')['tags'] == ['neg','prefix']
    assert module.parse_source_grammar('2sg non-honorific')['tags'] == ['2sg']
    assert module.parse_source_grammar('2sg high-honorific')['tags'] == ['2sg','honorific','high-honorific']
    assert module.parse_source_grammar('ADJV')['tags'] == ['adjectivizer']


def test_historical_witness_and_recording_locality_do_not_become_default_dialects():
    rows = {r['source_unit_key']: r for r in load('whole-expanded-metadata-review.jsonl')}
    historical = rows['dhakal2011:p145:ch5:comparison:l10']['source_attribution']
    assert historical['lect'] is None
    assert 'Hodgson' in historical['witness']
    wine = rows['dhakal2011:appendix5:HtMW.SD.002']['source_attribution']
    assert wine['lect'] == 'Chitwan'
    assert wine['corpus_context']['recording_locality'] == 'Kathmandu'
    assert wine['recording_location_is_not_a_dialect_assignment']
    assert rows['dhakal2011:p175:ch6:prose:l57:a1']['source_attribution']['lect'] == 'Tanahun'
    assert rows['dhakal2011:p3:ch12:l21:a3']['source_attribution']['lect'] is None
    assert rows['dhakal2011:p37:ch12:l8:a1']['source_attribution']['lect'] is None
    spec = importlib.util.spec_from_file_location('darai_attribution', P / 'whole_attribution.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture = [
        {'source_unit_key':'grandchild','source_claims':{},'disposition':'candidate','parent_expression_key':'child'},
        {'source_unit_key':'child','source_claims':{},'disposition':'candidate','parent_expression_key':'root'},
        {'source_unit_key':'root','source_claims':{'lect':'Tanahun'},'disposition':'candidate'},
    ]
    assert all(r['source_attribution']['lect'] == 'Tanahun' for r in module.apply_attribution(fixture))
    fixture = [
        {'source_unit_key':'modern','source_claims':{'witness':'Word list today'},'disposition':'candidate','parent_expression_key':'historical'},
        {'source_unit_key':'historical','source_claims':{'witness':"Hodgson's (1985) list, as cited by Dhakal"},'disposition':'candidate'},
    ]
    result = module.apply_attribution(fixture)
    assert result[0]['source_attribution']['lect'] == 'Chitwan'
    assert result[1]['source_attribution']['lect'] is None


def test_exact_source_function_tags_are_registered_without_prose_guessing():
    from tags import GRAMMATICAL_TAGS, extract_tags
    expected = {'habitual','hortative','hearsay','inferential','mirative',
                'desiderative','dubitative','obligation','possibility','purposive',
                'sequential-converb','simultaneous-converb','counterfactual',
                'adjectivizer','adverbializer','nominalizer',
                'imminent','frustrative','hypothetical','clitic'}
    assert expected <= GRAMMATICAL_TAGS
    frontend = (P.parents[5] / 'jambu-static/src/lib/tags.ts').read_text()
    registry = frontend.split('export const GRAMMATICAL_TAGS = [',1)[1].split('];',1)[0]
    assert all("'"+tag+"'" in registry for tag in expected)
    prose = 'The possibility of rain was reported as hearsay.'
    assert extract_tags(prose) == ('', prose)
    assert extract_tags('habitual visitor') == ('', 'habitual visitor')
    assert extract_tags('discussion of a clitic') == ('', 'discussion of a clitic')
    rows = {r['source_unit_key']: r for r in load('whole-expanded-metadata-review.jsonl')}
    marker = rows['dhakal2011:p384:appendix4:l57']
    assert {'adjectivizer','suffix'} <= set(marker['proposed_grammar']['tags'])
    assert 'adj' not in marker['proposed_grammar']['tags']
    habitual = rows['dhakal2011:p367:appendix3a:l20']
    assert 'habitual' in habitual['proposed_grammar']['tags']


def test_whole_transcription_coverage_preserves_canonical_and_unknown_marks():
    import unicodedata
    from segments.tokenizer import Tokenizer
    from profile_policy import house_output
    routes = {r['source_unit_key']: r for r in load('whole-transcription-proposal.jsonl')}
    with (P / 'legacy-before-whole-recovery.csv').open(newline='') as stream:
        old = list(csv.reader(stream))
    assert len(old) == 485
    assert all(routes[r[10]]['proposed_phonemic'] == r[5] for r in old)
    unknown = routes['dhakal2011:p141:ch5:surface:l76:word:l78:w2']
    assert unknown['route'] == 'literal-unresolved'
    assert unknown['proposed_phonemic'] == '' and unknown['transcription_decisions']
    tokenizer = Tokenizer(str(P / 'whole-profile-proposed.txt'))
    convert = lambda text: unicodedata.normalize('NFC', tokenizer(text, column='IPA').replace(' ', ''))
    for row in routes.values():
        text = row['proposed_phonemic']
        if not text:
            continue
        converted = convert(text)
        assert '\ufffd' not in converted, row['source_unit_key']
        assert converted == convert(unicodedata.normalize('NFD', text)), row['source_unit_key']
    with (P / 'whole-profile-proposed.txt').open() as stream:
        rules = {r['Grapheme']: r['IPA'] for r in csv.DictReader(stream, delimiter='\t')}
    assert rules['g'] == 'g'
    for grapheme, output in rules.items():
        assert house_output(grapheme, output, rules, 'dhakal-darai') == output


def test_explicit_category_columns_preserve_roles_without_semantic_guessing():
    rows = {r['source_unit_key']: r for r in load('whole-expanded-metadata-review.jsonl')}
    where = rows['dhakal2011:p215:ch7:table:l70']
    assert {'spatial','interr'} <= set(where['proposed_grammar']['tags'])
    assert 'dir' not in where['proposed_grammar']['tags']  # Not the direct case.
    temporal = rows['dhakal2011:p307:ch11:table:l14']
    assert 'temporal' in temporal['proposed_grammar']['tags']
    size = rows['dhakal2011:p307:ch11:table:l17']
    assert size['source_claims']['source_category'] == 'Size'
    assert 'adj' not in size['proposed_grammar']['tags']


def test_whole_csv_preview_preserves_old_graph_and_statement_citations():
    with (P/'whole-preview-author-transcription.csv').open(newline='') as stream:
        author=list(csv.reader(stream))
    with (P/'whole-preview-literal-preservation.csv').open(newline='') as stream:
        literal=list(csv.reader(stream))
    rows={r[10]:r for r in author+literal}
    assert len(rows)==10804 and len(literal)==12
    with (P/'legacy-before-whole-recovery.csv').open(newline='') as stream:
        for old in csv.reader(stream):
            assert all(rows[old[10]][i]==old[i] for i in [2,5,8,9,10,11,12,13])
    key='dhakal2011:appendix5:In.UN.047'
    assert rows[key][7]=='dhakal2011darai[p. 411];dhakal2011darai[p. 412]'
    assert rows[key+':statement:2'][7]=='dhakal2011darai[p. 412]'
    assert all(not r[5] and r[6] for r in literal)
    assert 'multiword-expression' in rows[key][14].split()
    output=rows['dhakal2011:p216:ch7:derivation:l38']
    assert output[13].split('|')==['dhakal2011:p216:ch7:derivation:l35','dhakal2011:p216:ch7:derivation:l37']
    assert {'adv','derived'} <= set(output[14].split())
    assert all(parent in rows for row in rows.values() for parent in row[13].split('|') if parent)
    assert rows['dhakal2011:p138:ch5:list:l18'][13]=='dhakal2011:p138:ch5:list:l16'
    assert rows['dhakal2011:p277:ch10:list:l35'][13]=='dhakal2011:p277:ch10:list:l33'
    assert not rows['dhakal2011:p277:ch10:list:l35'][11]


def test_all_source_notes_are_reconciled_and_scope_specific_functions_survive():
    rows = {r['source_unit_key']: r for r in load('whole-expanded-metadata-review.jsonl')}
    decisions=json.loads((P/'whole-note-editorial-decisions.json').read_text())
    assert decisions['reviewed_unique_notes']==539
    indexed={r['index']:r for r in decisions['decisions']}
    for i,tag in [(173,'imminent'),(215,'frustrative'),(200,'hypothetical'),(213,'hearsay')]:
        assert all(tag in rows[key]['proposed_grammar']['tags'] for key in indexed[i]['keys'])
    for decision in decisions['decisions']:
        for key in decision['keys']:
            assert rows[key]['editorial_source_note']==decision['output_note']
    with (P/'whole-preview-author-transcription.csv').open(newline='') as stream:
        for row in csv.reader(stream):
            assert 'awaits original' not in row[6]
            assert 'pending original' not in row[6]
            assert 'Higher-resolution independent review pending' not in row[6]


def test_proverb_senses_and_two_cited_statements_remain_distinct():
    rows = {r['source_unit_key']: r for r in load('whole-expanded-metadata-review.jsonl')}
    key='dhakal2011:p327:ch12:surface:l28'
    assert rows[key]['source_translation_type']=='literal'
    idiom=rows[key+':sense:idiomatic']
    assert idiom['proposed_grammar']['gloss']=='I was indecisive.'
    assert idiom['form']==rows[key]['form']
    assert 'figurative' in idiom['proposed_grammar']['tags']
    key='dhakal2011:p343:ch12:surface:l58'
    assert rows[key]['form'].endswith('boslə')
    assert rows[key+':statement:2']['form']=='əse sãdzʰ əilə'
    children=[r for k,r in rows.items() if k.startswith(key+':word:')]
    assert len(children)==9
    assert sum(r['parent_expression_key']==key for r in children)==6
    assert sum(r['parent_expression_key']==key+':statement:2' for r in children)==3
    assert rows[key+':word:l69:w1']['source_claims']['text_ids']==['SQ.BLD.133']


def test_whole_staging_accounts_every_occurrence_and_preserves_exact_reuse():
    audit = load('whole-proposed-audit.jsonl')
    rows = {}
    for route in ['author-transcription', 'literal-preservation']:
        with (P / ('whole-proposed-' + route + '.csv')).open(newline='') as stream:
            rows.update((r[10], r) for r in csv.reader(stream))
    assert len(audit) == 11304 and len(rows) == 8373
    assert sum(r['export']['disposition'] == 'reuse' for r in audit) == 2431
    assert sum(r['export']['disposition'] == 'audit-only' for r in audit) == 500
    for record in audit:
        export = record['export']
        if export['disposition'] == 'audit-only':
            continue
        target = rows[export['entry_key']]
        assert set(export['occurrence_citation'].split(';')) <= set(target[7].split(';'))
        if export['reason'].startswith('identical exported'):
            original = export['occurrence_row']
            assert all(original[i] == target[i] for i in [0,1,2,3,4,5,6,8,9])
            assert set(original[14].split()) == set(target[14].split())
    key = 'dhakal2011:p74:g621'
    assert rows[key][3] == 'not kill'
    assert 'will not kill' in rows[key][6]
    assert {'neg','3sg'} <= set(rows[key][14].split())


def test_reuse_fingerprint_keeps_existing_tags_and_exported_notes():
    import sys
    sys.path.insert(0, str(P))
    try:
        from prepare_whole_reuse_candidates import fingerprint
        from copy import deepcopy
        row = next(r for r in load('whole-expanded-metadata-review.jsonl')
                   if r['source_unit_key'] == 'dhakal2011:p74:g116:prose')
        changed = deepcopy(row)
        changed['proposed_grammar']['existing_tags'] = []
        assert fingerprint(row) != fingerprint(changed)
        a = [''] * 15
        b = a[:]
        b[6] = 'Different scholarly analysis.'
        assert fingerprint(row, a, 'author') != fingerprint(row, b, 'author')
        assert fingerprint(row, a, 'author') != fingerprint(row, a, 'literal')
    finally:
        sys.path.pop(0)


def test_whole_proposal_actual_parser_preserves_literals_notes_tags_and_citations(monkeypatch):
    import io
    import make_cldf
    from segments.tokenizer import Tokenizer
    meta = make_cldf.source_meta.SourceMeta([
        P/'whole-proposed-author-transcription.yaml',
        P/'whole-proposed-literal-preservation.yaml',
    ])
    monkeypatch.setattr(make_cldf.source_meta, 'load', lambda: meta)
    monkeypatch.setitem(make_cldf.convertors, 'dhakal-darai', Tokenizer(str(P/'whole-profile-proposed.txt')))
    for route in ['author-transcription', 'literal-preservation']:
        path = P / ('whole-proposed-' + route + '.csv')
        with path.open(newline='') as stream:
            expected = {r[10]: r for r in csv.reader(stream)}
        errors = io.StringIO()
        parsed, _ = make_cldf.parse_file(str(path), errors=errors)
        assert not errors.getvalue()
        assert {r.entry_key for r in parsed} == set(expected)
        for row in parsed:
            raw = expected[row.entry_key]
            assert row.old_form == raw[2] and row.ipa == raw[5]
            assert row.source == raw[7]
            assert set(raw[14].split()) <= set(row.tags.split())
            assert '\ufffd' not in row.form
            if route == 'literal-preservation':
                assert row.form == raw[2] and not row.ipa
            if row.entry_key == 'dhakal2011:p368:appendix3b:l70:root':
                assert row.old_form == 'uḍʰe-' and row.ipa == 'uḍʰe-'
                assert row.form.endswith('-')


def test_sampler_excludes_historical_key_schema_and_prior_whole_samples():
    import sys
    sys.path.insert(0, str(P))
    try:
        from sample_whole_proposal import source_keys, sample
        old = set(source_keys(json.loads((P/'preview/acceptance-sample.json').read_text())))
        assert 'dhakal2011:p59:g201' in old and len(old) >= 20
        chapters = set(source_keys(json.loads((P/'chapters12-independent-selection.json').read_text())))
        assert 'dhakal2011:p3:ch12:l21:a2' in chapters and len(chapters) >= 20
        result = sample(20260927203, 20)
        assert old | chapters <= set(result['excluded_keys'])
        assert not {r['source_unit_key'] for r in result['records']} & set(result['excluded_keys'])
    finally:
        sys.path.pop(0)


def test_source_headings_and_derivation_roles_are_not_lost_or_overgeneralized():
    rows = {r['source_unit_key']:r for r in load('whole-expanded-metadata-review.jsonl')}
    tags = lambda key: set(rows[key]['proposed_grammar']['tags'])
    assert 'adj' in tags('dhakal2011:p126:ch5:list:l24')
    marker = tags('dhakal2011:p178:ch6:table:l39')
    assert {'suffix','3sg','honorific'} <= marker
    assert not {'gen','dat','nom','erg'} & marker
    word = tags('dhakal2011:p220:ch7:surface:l33:word:l37:w1')
    assert {'adv','derived'} <= word and 'adverbializer' not in word
    assert 'adverbializer' in tags('dhakal2011:p384:appendix4:l60')
    assert 'adj' not in tags('dhakal2011:p384:appendix4:l57')
    assert 'clitic' in tags('dhakal2011:p222:ch8:prose:l29:a1')
    assert 'suffix' not in tags('dhakal2011:p222:ch8:prose:l29:a1')
    assert 'suffix' not in tags('dhakal2011:p158:ch6:prose:l37:a1')
    assert {'verb','intr','non-past'} <= tags('dhakal2011:p173:ch6:table:l12')
    assert 'relative' in tags('dhakal2011:p307:ch11:table:l14')
    assert 'correlative' in tags('dhakal2011:p307:ch11:table:l15')
    assert 'adj' not in tags('dhakal2011:p307:ch11:table:l17')
    with (P/'whole-proposed-author-transcription.csv').open(newline='') as stream:
        for row in csv.reader(stream):
            assert 'Printed interlinear analysis: .' not in row[6]


def test_all_paradigm_roles_and_literal_root_boundaries_are_preserved():
    rows = load('whole-expanded-metadata-review.jsonl')
    roots = [r for r in rows if r['kind'] == 'paradigm-root']
    assert len(roots) == 49
    assert all('verb' in r['proposed_grammar']['tags'] for r in roots)
    inflected = [r for r in rows if r['kind'] == 'paradigm-inflection' and not r['disposition'].startswith('audit-only')]
    assert len(inflected) == 555 and all('verb' in r['proposed_grammar']['tags'] for r in inflected)
    corrected = [r for r in roots if r.get('root_boundary_decision')]
    assert len(corrected) == 37
    assert all(r['form'] == r['root_boundary_decision']['before_form'] + '-' for r in corrected)
    by_key = {r['source_unit_key']: r for r in rows}
    assert by_key['dhakal2011:p376:appendix3d:root:l110']['form'] == 'pəḍʰə'
    for row in rows:
        if row['kind'] in {'bound-morpheme','bound-morpheme-repeat','bound-morphology','derivation-affix','comparative-tense-cell'} and not row['disposition'].startswith('audit-only'):
            assert 'verb' not in row['proposed_grammar']['tags']
        if row['kind'] == 'comparative-pronoun-cell' and not row['disposition'].startswith('audit-only'):
            assert 'pron' in row['proposed_grammar']['tags']
    our = by_key['dhakal2011:p145:ch5:comparison:l19']['proposed_grammar']['tags']
    assert {'pron','poss','1pl'} <= set(our) and 'adj' not in our


def test_reviewed_alignment_overrides_only_resolved_geometry_warnings():
    rows = {r['source_unit_key']:r for r in load('whole-expanded-metadata-review.jsonl')}
    verb = rows['dhakal2011:p214:ch7:surface:l53:word:l55:w2']['proposed_grammar']
    assert verb['gloss'] == 'die' and {'pret','verb'} <= set(verb['tags'])
    assert verb['alignment_adjudication'][0]['printed_gloss'] == 'die-PST'
    assert 'held_analysis' not in verb
    for key in ['dhakal2011:p78:ch4:surface:l93:word:l100:w1',
                'dhakal2011:p91:ch4:surface:l62:word:l67:w1']:
        assert rows[key]['proposed_grammar']['held_analysis']
        assert not rows[key]['proposed_grammar']['tags']
    anomaly=rows['dhakal2011:p190:ch6:surface:l28:word:l31:w1']
    assert anomaly['source_gloss']=='1SG plough'
    assert anomaly['proposed_grammar']['gloss']=='plough'
    assert 'uncertain' in anomaly['proposed_grammar']['tags']
    assert '1sg' not in anomaly['proposed_grammar']['tags']


def test_conditional_particles_and_residual_labels_preserve_grammatical_roles():
    import re
    spec=importlib.util.spec_from_file_location('darai_interlinear_grammar',P/'interlinear_grammar.py')
    grammar=importlib.util.module_from_spec(spec);spec.loader.exec_module(grammar)
    assert set(grammar.parse('COND.PART')['tags'])=={'conditional','part'}
    assert set(grammar.parse('become-COND')['tags'])=={'conditional','verb'}
    assert grammar.parse('English conditional particle')['tags']==[]
    rows=load('whole-expanded-metadata-review.jsonl')
    particles=[r for r in rows if r['source_gloss']=='COND.PART']
    assert len(particles)==23
    assert all(set(r['proposed_grammar']['tags'])=={'conditional','part'} for r in particles)
    for r in rows:
        tags=set(r['proposed_grammar']['tags'])
        if tags & {'part','prefix','suffix','affix','infix','clitic'}:
            assert 'verb' not in tags
    keyed={r['source_unit_key']:r for r in rows}
    assert keyed['dhakal2011:p146:ch6:pair:l30']['proposed_grammar']['gloss']=='throw'
    assert {'pret','verb'} <= set(keyed['dhakal2011:p146:ch6:pair:l30']['proposed_grammar']['tags'])
    assert keyed['dhakal2011:p184:ch6:prose:l46:a4']['proposed_grammar']['gloss']==''
    assert {'suffix','1sg'} <= set(keyed['dhakal2011:p184:ch6:prose:l46:a4']['proposed_grammar']['tags'])
    sentence=keyed['dhakal2011:p371:appendix3c1:whole-expression:l24']
    assert sentence['proposed_grammar']['gloss']=='I gave you.'
    assert not {'sg','2sg','honorific'} & set(sentence['proposed_grammar']['tags'])
    decisions=json.loads((P/'whole-interlinear-scope-decisions-20260927.json').read_text())['decisions']
    assert len(decisions)==100
    for r in rows:
        if r['disposition'].startswith('audit-only'):continue
        gloss=r['proposed_grammar']['gloss']
        if re.search(r'\b(?:[A-Z]{2,}|[123](?:SG|PL))\b',gloss):
            assert r['source_unit_key'] in {
                'dhakal2011:p78:ch4:surface:l93:word:l100:w1',
                'dhakal2011:p91:ch4:surface:l62:word:l67:w1',
            } or ' VS.' in gloss
