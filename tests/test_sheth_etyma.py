import csv
import gzip
import importlib.util
import io
import json
from pathlib import Path
import pytest
import make_cldf

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('sheth_etyma', ROOT/'data/other/forms/raw_data/sheth_etyma.py')
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)

@pytest.mark.parametrize('native,roman', [('अभिध्या','abhidhyā'),('निक्षेपित','nikṣēpita'),('तम्','tam'),('तरीतृ','tarītṛ'),('जितेन्द्रिय','jitēndriya')])
def test_source_transliteration(native,roman):
    assert S.extract(native)['form']==roman

@pytest.mark.parametrize('text,status',[('दे','desya-label'),('दे.','desya-label'),('','absent'),('उत् + कृ','compound-analysis'),('दे. तुलसी','sanskrit-equivalent'),('प्रागल्भिन्, °क','unresolved-expression'),('कृत?','unresolved-expression'),('आ +','compound-analysis')])
def test_labels_and_uncertain_expressions_are_not_ancestors(text,status):
    claim=S.extract(text)
    assert claim['status']==status
    assert claim.get('relation') in (None,'related')
    assert claim.get('direction') in (None,'undetermined')


def test_installed_equivalents_and_comparisons_are_lossless_and_scoped():
    with (S.HERE.parent/S.FILENAME).open() as f: forms=list(csv.reader(f))
    by_key={r[10]:r for r in forms}
    assert len(by_key)==len(forms)==26191
    assert all(r[0]=='Sk' and r[4] and not any(r[i] for i in [1,3,5,8,11,12,13,14]) for r in forms)
    with (S.HERE.parent/'20260914-sheth.csv').open() as f: prakrit={r[10]:r for r in csv.reader(f)}
    with S.COMPARISONS.open() as f: comparisons=list(csv.DictReader(f))
    assert len(comparisons)==27484
    for c in comparisons:
        assert c['Entry_ID'] in prakrit and c['Compared_Entry_ID'] in by_key
        row=prakrit[c['Entry_ID']]
        assert S.extract(row[9])['native']==by_key[c['Compared_Entry_ID']][4]
        assert c['Compared_Entry_ID']==row[10].rsplit(':v',1)[0]+':sanskrit'
        assert c['Relation']=='related' and c['Direction']=='undetermined'
    errors=io.StringIO()
    compiled,stats=make_cldf.parse_file(str(S.HERE.parent/S.FILENAME),errors,file_num='sheth-sanskrit')
    assert not errors.getvalue() and len(compiled)==len(forms)
    assert all(r.form==by_key[r.entry_key][2] for r in compiled)
    with gzip.open(S.PACKAGE/'etymology-audit.jsonl.gz','rt') as f:
        records=[json.loads(line) for line in f]
    assert len(records)==41638
    assert sum(c['status']=='sanskrit-equivalent' for r in records for c in r['claims'])==27484


def test_keyed_comparison_resolution_validates_both_endpoints(tmp_path,monkeypatch):
    for name in ['data/cross-family-comparisons.csv','data/manual-cross-family-comparisons.csv','data/dbia/comparisons.csv']:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(','.join(S.COLUMNS)+'\n')
    (tmp_path/'cldf').mkdir()
    (tmp_path/'cldf/forms.csv').write_text('ID,Entry_Key\np,source:s1:v1\ns,source:s1:sanskrit\n')
    p=tmp_path/'data/other/comparisons/test.csv';p.parent.mkdir(parents=True)
    with p.open('w') as f:
        w=csv.writer(f);w.writerow(S.COLUMNS);w.writerow(['c','source:s1:v1','source:s1:sanskrit','related','undetermined','high','sheth1923[p. 1]','Printed Sanskrit equivalent'])
    monkeypatch.chdir(tmp_path)
    make_cldf.write_cross_family_comparisons(set())
    with (tmp_path/'cldf/comparisons.csv').open() as f:r=next(csv.DictReader(f))
    assert (r['Entry_ID'],r['Compared_Entry_ID'])==('p','s')
    p.write_text(p.read_text().replace('source:s1:sanskrit','missing'))
    with pytest.raises(ValueError,match='unresolved comparison source key'):
        make_cldf.write_cross_family_comparisons(set())
