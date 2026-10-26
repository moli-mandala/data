"""Full-expression boundaries and translation scopes must survive integration."""
import csv,json
from pathlib import Path

P=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/grierson_naiki_1906'


def test_complete_utterance_coverage_and_translation_scope():
    units=[json.loads(x) for x in (P/'reviewed-whole-utterances.jsonl').read_text().splitlines()]
    utterances=[u for u in units if u['kind']=='utterance']
    blocks=[u for u in units if u['kind']=='translated-block']
    assert len(utterances)==51 and len(blocks)==7
    tokens=[k for u in utterances for k in u['physical_token_keys']]
    assert len(tokens)==len(set(tokens))==537
    assert sum(len(u['speaker_label_evidence']) for u in utterances)==6
    assert all(not u['free_translation'] for u in utterances)
    assert len(blocks[0]['source_utterance_keys'])==10
    assert 'small children' in blocks[0]['free_translation']
    assert 'father, whence' in blocks[3]['free_translation']
    for u in units:
        assert u['form'] == ' '.join(u['raw_source_tokens'])
        assert u['form'].rstrip('’”').endswith(('.', '?'))
        labels={r['source_unit_key'] for r in u['speaker_label_evidence']}
        assert not labels & set(u['source_token_keys'])
        assert labels | set(u['source_token_keys']) == set(u['physical_token_keys'])
    assert any(';' in u['form'] for u in utterances)
    assert any('?' in u['form'] for u in utterances)
    assert sum(sum(u['form'].count(mark) for mark in '‘’“”') for u in utterances)==31
    assert any('.”’' in u['form'] for u in utterances)
    quoted_question=next(u for u in utterances if 'kōrtēn?' in u['form'])
    assert quoted_question['form'].endswith('?’') and '‘' not in quoted_question['form']
    rows={r[10]:r for r in csv.reader((P/'whole-review-proposal.csv').open())}
    audit={r['source_unit_key']:r for r in map(json.loads,(P/'whole-review-proposal-audit.jsonl').read_text().splitlines())}
    for u in units:
        row=rows[u['source_unit_key']]
        assert row[2:4]==[u['form'],u['free_translation']]
        assert 'sentential' in row[14].split()
        assert not row[4] and not row[5]
    uncertain=[u for u in utterances if any('p572:specimen:line14:word8' in k for k in u['source_token_keys'])][0]
    assert 'uncertain' in rows[uncertain['source_unit_key']][14].split()
    assert audit[uncertain['source_unit_key']]['typed_uncertainties']
