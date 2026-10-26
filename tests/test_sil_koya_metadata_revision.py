"""Source-grounded scope integration; no canonical installation implied."""
import csv
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/sil_koya_2021/whole-source-recovery'
OLD=BASE/'literal-integration'
NEW=BASE/'literal-integration-metadata-r2'


def rows(folder,kind):
    with (folder/f'proposed-{kind}.csv').open(newline='') as stream:
        return list(csv.reader(stream))


def test_metadata_revision_preserves_every_literal_key_and_graph_field():
    total=0
    for kind,count in [('wordlist',1437),('narrative',210),('comparative',18)]:
        old,new=rows(OLD,kind),rows(NEW,kind)
        assert len(old)==len(new)==count
        assert all(a[i]==b[i] for a,b in zip(old,new) for i in range(15) if i not in [3,14])
        total+=len(new)
    assert total==1665
    assert (OLD/'proposed-entry-key-aliases.json').read_bytes()==(NEW/'proposed-entry-key-aliases.json').read_bytes()


def test_all_narrative_scope_decisions_are_consumed_without_flattening_fragments():
    decisions=json.loads((BASE/'narrative-scope-review-20260927.json').read_text())['decisions']
    by={r[10]:r for r in rows(NEW,'narrative')}
    counts={'sentential':0,'multiword-expression':0}
    for d in decisions:
        row=by[d['source_unit_key']]
        assert (row[2],row[3])==(d['original'],d['source_translation'])
        assert d['scope_tag'] in row[14].split()
        if d['scope_tag']=='multiword-expression':assert 'sentential' not in row[14].split()
        if d['retain_interr']:assert 'interr' in row[14].split()
        counts[d['scope_tag']]+=1
    assert counts=={'sentential':165,'multiword-expression':40}
    for key,row in by.items():
        if key.endswith('-context'):assert row[3]=='' and 'sentential' not in row[14].split()
    stutter=by['devagnanavaram2021:whole:B.7:p30:5']
    assert '[stutter]' in stutter[2] and 'sentential' in stutter[14].split()


def test_complete_wordlist_concept_classes_and_unqualified_pronouns():
    for row in rows(NEW,'wordlist'):
        item=int(row[10].split(':')[2][1:]);tags=set(row[14].split())
        if 151<=item<=164:assert 'num' in tags
        if 165<=item<=170:assert 'interr' in tags
        if 171<=item<=174:assert 'demonstrative' in tags
        if item in [203,204,209]:
            assert row[3]=='you' and {'pron','personal','second-person'}<=tags
            assert not tags & {'2sg','2pl','formal','informal'}
        if item in [207,208]:
            assert row[3]=='we' and {'pron','personal','1pl'}<=tags
            assert not tags & {'inclusive','exclusive'}
        if row[11]:assert 'alternate' in tags
    review=json.loads((NEW/'metadata-repair-review.json').read_text())
    assert len(review['wordlist_concept_coverage'])==210
    assert len(review['narrative_and_comparative_coverage'])==228


def test_comparative_scope_and_literal_profiles_remain_exact():
    for row in rows(NEW,'comparative'):
        if row[3]=='six':assert 'num' in row[14].split()
        if ' ' in row[2]:assert 'multiword-expression' in row[14].split()
        assert row[11:14]==['','','']
    for kind in ['literal','narrative','comparative']:
        name=f'proposed-sil-koya-{kind}.txt'
        assert (OLD/name).read_bytes()==(NEW/name).read_bytes()


def test_frozen_evidence_and_actual_parser_metadata_checks_match():
    old=json.loads((OLD/'review-manifest.json').read_text())
    for name,digest in old['outputs_sha256'].items():
        assert hashlib.sha256((OLD/name).read_bytes()).hexdigest()==digest
    new=json.loads((NEW/'review-manifest.json').read_text())
    for name,digest in new['outputs_sha256'].items():
        assert hashlib.sha256((NEW/name).read_bytes()).hexdigest()==digest
    parsed=json.loads((NEW/'parser-validation.json').read_text())
    assert parsed['total_active_keys']==1665
    assert all(v['conversion_errors']==0 for v in parsed['results'].values())
    meta=json.loads((NEW/'metadata-validation.json').read_text())
    assert len(meta['registered_dialects'])==7 and meta['variant_edges_closed']
    assert meta['variant_edges']==36
