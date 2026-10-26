"""Check case scope and source uncertainty in the uninstalled Naiki proposal."""
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / 'data/other/forms/raw_data/grierson_naiki_1906'


def test_explicit_case_scope_and_negative_controls():
    spec = importlib.util.spec_from_file_location('naiki_relations', BASE / 'specimen_relations.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    decisions = module.decisions()
    assert decisions[('akāsnē', 'heaven-of')]['tags'] == ['noun', 'gen']
    assert decisions[('kāl-lōpul', 'foot-on')]['tags'] == ['noun', 'loc']
    assert ('bānēkil', 'father-to') not in decisions
    assert ('guḍuṅgā-lōpulu', 'neck-on') not in decisions
    assert ('aunē', 'their') not in decisions
    records = [json.loads(line) for line in (BASE / 'specimen-grammatical-proposal.jsonl').read_text().splitlines()]
    indexed = {(r['proposed_form'].casefold(), r['punctuation_cleaned_gloss'].casefold()): r for r in records}
    for pair, decision in decisions.items():
        row = indexed[pair]
        assert row['proposed_gloss'] == decision['lexical_gloss']
        assert row['tags'] == decision['tags']
        assert row['grammatical_support'] == [decision['support']]
        assert row['raw_form'].startswith(row['proposed_form'])
    towards = indexed['bānēkil', 'father-to']
    assert towards['tags'] == ['noun']
    assert towards['proposed_gloss'] == 'father-to'
    agent = indexed['kaktēn', 'doer']
    assert agent['tags'] == ['noun', 'uncertain']
    assert agent['uncertainty']['type'] == 'source-analysis'
    assert 'apparently' in agent['source_caveat']
    assert not indexed['annē', 'me']['metadata_review_pending']
    assert not {'acc','dat','gen'} & set(indexed['annē','me']['tags'])
    assert indexed['annē','me']['uncertainty']['type']=='source-analysis'


def test_source_cautions_survive_contextual_analysis():
    records = [json.loads(line) for line in (BASE / 'specimen-grammatical-proposal.jsonl').read_text().splitlines()]
    indexed = {(r['proposed_form'].casefold(), r['punctuation_cleaned_gloss'].casefold()): r for r in records}
    questioned = indexed['iūn', 'said']
    assert questioned['uncertainty']['type'] == 'source-transcription'
    assert 'uncertain' in questioned['tags']
    assert questioned['grammatical_support'] == ['grierson1906naiki:p571:prose:said-questioned']
    for pair in [('aṇḍur', 'was'), ('antēr', 'was')]:
        row = indexed[pair]
        assert 'uncertain' in row['tags'] and '3pl' not in row['tags']
        assert 'plural' in row['source_caveat']
    for pair in [('kaktēn', 'made'), ('kaktēn', 'madest'), ('kharcha-kaktēn', 'expenditure-made')]:
        row = indexed[pair]
        assert row['uncertainty']['type'] == 'source-analysis'
        assert 'uncertain' in row['tags']
        assert 'apparently nouns of agency' in row['source_caveat']


def test_reviewed_context_is_occurrence_specific_and_complete():
    from tags import GRAMMATICAL_TAGS,GENDER_TAGS
    rows=[json.loads(line) for line in (BASE/'specimen-grammatical-proposal.jsonl').read_text().splitlines()]
    assert len(rows)==537 and not any(r['metadata_review_pending'] for r in rows)
    reviewed=[r for r in rows if r['independent_context_decision']]
    assert len(reviewed)==114
    for r in rows:
        assert set(r['tags']) <= GRAMMATICAL_TAGS|GENDER_TAGS
    pairs={(r['proposed_form'],r['punctuation_cleaned_gloss']):r for r in reviewed}
    assert '1sg' in pairs['āplē','my']['tags']
    assert '3sg' in pairs['āplē','his']['tags']
    assert '3sg' not in pairs['Āplē','Your']['tags']
    assert pairs['ki','that']['tags']==['conj']
    assert pairs['ād','that']['tags']==['pron','demonstrative']
    assert pairs['Aunnē-lōpul','Them-in']['proposed_gloss']=='Them'
    assert 'loc' in pairs['Aunnē-lōpul','Them-in']['tags']
    assert 'loc' not in pairs['guḍuṅgā-lōpulu','neck-on']['tags']
    assert pairs['annun','me-to']['proposed_gloss']=='me'
    assert pairs['miṭī','embracing']['uncertainty']['type']=='contextual-analysis'
