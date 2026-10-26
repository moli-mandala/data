"""Independent review proposals must preserve evidence and reject stale inputs."""
import importlib.util
import json
import shutil
from pathlib import Path
import pytest

PACKAGE = Path(__file__).resolve().parents[1] / 'data/other/forms/raw_data/niranjan_chakma_2010'
spec = importlib.util.spec_from_file_location('chakma_grammar_reconciler', PACKAGE / 'reconcile_grammar_review.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_proposals_preserve_tags_uncertainties_and_source_anomalies():
    rows, status = module.reconcile()
    source = [json.loads(x) for x in (PACKAGE / module.SOURCE).read_text().splitlines()]
    assert [r['source_unit_key'] for r in rows] == [r['source_unit_key'] for r in source]
    assert status['field_recommendations'] == {'candidate_tags':303,'source_note':41,'gloss_bn':3,'form':1}
    assert status['changed_keys'] == 335
    assert sum(len(r['independent_typed_uncertainties']) for r in rows) == 30
    for before, after in zip(source, rows):
        assert set(before['candidate_tags']) <= set(after['reviewed_fields_proposal']['candidate_tags'])
        assert after.get('uncertainties') == before.get('uncertainties')
        assert after['form'] == before['form']
        assert after['accepted_for_installation'] is False
    by = {r['source_unit_key']:r for r in rows}
    odd = by['niranjan2010:p13:grammar:b4:subject:3']
    assert odd['form'] == 'তে'
    assert odd['independent_typed_uncertainties']
    assert odd['reviewed_fields_proposal']['candidate_tags'] == ['pron']
    ellipsis = by['niranjan2010:p13:grammar:b1:example:4.1']
    assert ellipsis['form'] == 'মুই ....'
    assert 'sentential' not in ellipsis['reviewed_fields_proposal']['candidate_tags']
    future = by['niranjan2010:p17:grammar:b2:subject:1']
    assert 'সাধারণ অতীত' in future['source_note'] and 'Future Indefinite' in future['source_note']
    assert not {'fut','pret'} & set(future['reviewed_fields_proposal']['candidate_tags'])
    chicken = by['niranjan2010:p10:grammar:b1:feminine:5']
    assert chicken['form'] == 'কুর্ঘী কুরাহ্'
    assert chicken['reviewed_fields_proposal']['form'] == 'কুহরী কুরাহ্'
    assert chicken['independent_typed_uncertainties']

@pytest.mark.parametrize('mutation',['hash','old_value','duplicate','missing_coverage'])
def test_rejects_stale_or_ambiguous_review(tmp_path, mutation):
    report = json.loads((PACKAGE / module.REPORT).read_text())
    for name in [*report['input_sha256'],module.REPORT]:
        shutil.copyfile(PACKAGE / name, tmp_path / name)
    if mutation == 'hash':
        with (tmp_path / module.SOURCE).open('a') as f:f.write('\n')
    else:
        if mutation == 'old_value':report['corrections'][0]['old'] = ['unreviewed-value']
        elif mutation == 'duplicate':report['corrections'][1] = report['corrections'][0]
        else:report['checks'].pop()
        (tmp_path / module.REPORT).write_text(json.dumps(report,ensure_ascii=False))
    with pytest.raises(AssertionError):module.reconcile(tmp_path)
