"""Preparation boundaries: complete source topology without claiming acceptance."""
import collections
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/bhat_belari_1971'

def records():
    return [json.loads(x) for x in (ROOT/'lexical-candidates.jsonl').read_text().splitlines()]

def test_every_inventoried_source_occurrence_has_a_stable_locator():
    rows=records()
    inventory=json.loads((ROOT/'section-inventory.json').read_text())
    expected={r['section']:r['source_form_occurrences'] for r in inventory['sections'] if r['source_form_occurrences']}
    assert dict(collections.Counter(r['section'] for r in rows))==expected
    assert len(rows)==len({r['entry_key'] for r in rows})==191
    assert {r['printed_page'] for r in rows}=={119,120,121,122,123}
    for row in rows:
        assert row['pdf_page']==row['printed_page']+7
        assert row['candidate_form'] and row['source_gloss']
        assert row['status']=='first-pass-visual-transcription; not accepted for emission'
        assert row['entry_key']==f"bhat1971belari:p{row['printed_page']}:s{row['section']}:c{row['column']}:r{row['row_in_section_column']}"

def test_syncretic_cells_and_source_anomalies_are_not_collapsed():
    rows=records()
    subj=[r for r in rows if r['section']=='7c']
    first=[r for r in subj if r['source_grammar'] in ('subjunctive I singular','subjunctive I plural')]
    assert [r['candidate_form'] for r in first]==['brave','bravo']
    imperative=[r for r in rows if r['section']=='7d' and r['candidate_form']=='balle']
    assert len(imperative)==2
    assert {r['source_grammar'] for r in imperative}=={'imperative feminine singular','imperative plural'}
    we=[r for r in rows if r['section']=='12' and r['source_gloss']=='we']
    assert len(we)==1 and we[0]['candidate_form']=='eŋkḷo'
    assert all('inclusive' not in r['source_gloss'] and 'exclusive' not in r['source_gloss'] for r in rows)
    chilly=[r for r in rows if r['source_gloss']=='chilly']
    assert len(chilly)==1 and any(s.startswith('gloss:') for s in chilly[0]['review'])

def test_review_corrections_do_not_overwrite_first_pass():
    review=[json.loads(x) for x in (ROOT/'glyph-review.jsonl').read_text().splitlines()]
    assert len(review)==191
    by_key={r['entry_key']:r for r in records()}
    assert all(by_key[r['entry_key']]['candidate_form']==r['first_pass'] for r in review)
    corrections=[r for r in review if r['reviewed_form']!=r['first_pass']]
    assert len(corrections)==10
    assert ('battigo','battɨgo') in [(r['first_pass'],r['reviewed_form']) for r in corrections]
    assert ('aḷi','aḷɨ') in [(r['first_pass'],r['reviewed_form']) for r in corrections]
    assert sum(bool(r.get('issues')) for r in review)==3

def test_grammatical_scope_and_reviewed_readings_survive_preparation():
    import importlib.util
    from tags import GRAMMATICAL_TAGS,GENDER_TAGS
    spec=importlib.util.spec_from_file_location('belari_analysis',ROOT/'prepare_analysis.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=module.prepare()
    assert rows==[json.loads(x) for x in (ROOT/'analysis-proposals.jsonl').read_text().splitlines()]
    assert len(rows)==192 and len({r['source_occurrence_key'] for r in rows})==191
    assert all(set(r['tags']) <= GRAMMATICAL_TAGS|GENDER_TAGS for r in rows)
    by_key={r['entry_key']:r for r in rows}
    assert by_key['bhat1971belari:p122:s7b:c2:r4:analysis:1']['form']=='battɨgo'
    shared=[r for r in rows if r['source_occurrence_key']=='bhat1971belari:p123:s12:c1:r4']
    assert [r['tags'] for r in shared]==[['pron','2sg','f'],['pron','2pl']]
    plural=by_key['bhat1971belari:p121:s7a:c2:r2:analysis:1']
    assert plural['tags']==['verb','non-past','second-person','pl']
    assert by_key['bhat1971belari:p121:s7a:c1:r5:analysis:1']['tags'][-1]=='fn'
    assert 'conditional' not in by_key['bhat1971belari:p122:s7-nonfinite-c:c1:r2:analysis:1']['tags']
    assert all(r['glyph_reviewed'] for r in rows)
    assert sum('uncertain' in r['tags'] for r in rows)==4
    assert by_key['bhat1971belari:p123:s12:c1:r5:analysis:1']['form']=='ay.i'

def test_overlap_does_not_invent_attestations_or_grammatical_equivalence():
    import importlib.util
    spec=importlib.util.spec_from_file_location('belari_overlap',ROOT/'reconcile_lindgren.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=module.prepare()
    assert rows==[json.loads(x) for x in (ROOT/'lindgren-overlap-review.jsonl').read_text().splitlines()]
    assert len(rows)==108
    by_key={r['derived_entry_key']:r for r in rows}
    for key in ('lindgren:301','lindgren:323','lindgren:363'):
        assert not by_key[key]['primary_candidates']
        assert by_key[key]['status']=='not-located-in-appendix'
    assert by_key['lindgren:270']['status']=='grammatical-disagreement'
    assert by_key['lindgren:302']['status']=='gloss-ambiguity'
    assert by_key['lindgren:288']['status']=='derived-stem'
    assert by_key['lindgren:266']['primary_candidates'][0]['tags']==['pron','2sg','f']
    assert by_key['lindgren:373']['primary_candidates'][0]['tags']==['pron','2pl']

def test_paradigm_relationships_have_explicit_existing_root():
    decisions=json.loads((ROOT/'relationship-decisions.json').read_text())
    analyses={r['entry_key']:r for r in map(json.loads,(ROOT/'analysis-proposals.jsonl').read_text().splitlines())}
    assert analyses[decisions['paradigm_root']]['form']=='bar'
    assert len(decisions['relationships'])==45
    for relation in decisions['relationships']:
        assert relation['entry_key'] in analyses
        assert relation['parent_key']==decisions['paradigm_root']
        assert relation['proposed_relationship']=='variant'

def test_source_emission_preserves_uncertainty_and_paradigm_links(monkeypatch):
    import importlib.util,sys,unicodedata
    from segments import Tokenizer
    monkeypatch.syspath_prepend(str(ROOT))
    spec=importlib.util.spec_from_file_location('belari_import',ROOT/'import_source.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows,audit=module.build()
    assert len(rows)==len(audit)==192
    assert all(len(r)==15 and r[0]=='Belari' and not r[1] and not r[4] and not r[5] for r in rows)
    assert sum(bool(r[11]) for r in rows)==45
    assert sum('uncertain' in r[14].split() for r in rows)==4
    assert sum(bool(r[9]) for r in rows)==14
    tokenizer=Tokenizer(str(ROOT/'bhat-belari.txt'))
    def convert(s):return unicodedata.normalize('NFC',tokenizer(s,column='IPA').replace(' ','').replace('#',' '))
    for r in rows:
        assert '�' not in convert(r[2])
        assert convert(r[2])==convert(unicodedata.normalize('NFD',r[2]))
    assert convert('ay.i')=='ay.i'
    assert convert('hu:ju')=='hūju'
    assert convert('be:yi')=='bēyi'
    assert convert('haṇɨ')=='haṇɨ'

def test_installed_source_uses_shared_citation_with_language_specific_profile():
    import csv,io
    import make_cldf,source_meta
    from pybtex.database import parse_file
    project=ROOT.parents[4]
    source=project/'data/other/forms/20260922-bhat-belari.csv'
    meta=source_meta.load()
    assert meta.transcription('bhat1971koraga',source,'Belari')[0]=='bhat-belari'
    assert meta.transcription('bhat1971koraga',source,'Koraga')[0]=='selected-koraga'
    assert (project/'conversion/bhat-belari.txt').read_text()==(ROOT/'bhat-belari.txt').read_text()
    expected=[r['row'] for r in map(json.loads,(ROOT/'audit.jsonl').read_text().splitlines())]
    assert list(csv.reader(source.open()))==expected
    errors=io.StringIO()
    parsed,stats=make_cldf.parse_file(str(source),errors,name=source.stem)
    assert not errors.getvalue()
    assert len(parsed)==stats['converted']==192
    by_key={r[10]:r for r in expected}
    for row in parsed:
        raw=by_key[row.entry_key]
        assert (row.old_form,row.gloss,row.notes,row.etymology,row.variant_of_key,row.tags)==(raw[2],raw[3],raw[6],raw[9],raw[11],raw[14])
    bibliography=parse_file(str(project/'cldf/sources.bib'))
    assert 'Belari' in bibliography.entries['bhat1971koraga'].fields['included']
