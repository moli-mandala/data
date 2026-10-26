"""Complete paired-source proposals: scope, literal recovery and source metadata."""
import csv,json,unicodedata
from pathlib import Path
from segments import Tokenizer
from tags import GRAMMATICAL_TAGS,GENDER_TAGS

D=Path(__file__).resolve().parents[1]
P=D/'data/other/forms/raw_data/bailey_rampur_1920'
B=P.parent/'bailey_baghi_1920'


def rows(p):return list(csv.reader((p/'whole-proposed.csv').open()))
def units():return [json.loads(x) for x in (P/'whole-paired-expansion-draft-20260927.jsonl').read_text().splitlines()]
def keyed():return {r[10]:r for p in [P,B] for r in rows(p)}


def test_complete_scope_and_exact_installed_key_preservation():
    for p,count,legacy in [(P,647,288),(B,698,296)]:
        data=rows(p)
        assert len(data)==count
        original=list(csv.reader((p/'legacy-before-whole-recovery.csv').open()))
        changes=json.loads((P/'whole-legacy-corrections-20260927-pass2.json').read_text())['changes']
        corrections={c['entry_key']:c for c in changes}
        expected=[]
        for row in original:
            if row[10] in corrections:
                assert corrections[row[10]]['before']==row
                expected.append(corrections[row[10]]['after'])
            else:expected.append(row)
        assert data[:legacy]==expected
        assert [r[10] for r in data[:legacy]]==[r[10] for r in original]
        assert len({r[10] for r in data})==len(data)
        assert len((p/'whole-proposed-audit.jsonl').read_text().splitlines())==640
    source=units()
    assert len(source)==395
    assert {kind:sum(u['source_kind']==kind for u in source) for kind in ['grammar','sentence','commentary','introduction']}=={'grammar':305,'sentence':44,'commentary':19,'introduction':27}
    assert sum(not u['candidates'] for u in source)==23
    assert len([c for u in source for c in u['candidates']])==760
    assert sum(r[0]=='kc' for r in rows(P))==4 and all(r[0]=='ba' for r in rows(B))


def test_explicit_stem_ditto_and_person_specific_expansion():
    data=keyed()
    def f(section,item,field,page=137):
        return [r for k,r in data.items() if k.startswith(f'paired:grammar:{page}:{section}:{item}:') and k.endswith(':'+field+':1')][0]
    assert f('horse',2,'baghi_sg',131)[2]=='gōhṛe ro'
    assert f('girl',5,'rampur_pl',131)[2]=='tshōṭīē'
    assert f('fall-imperfect-feminine',1,'baghi_sg')[2]=='lōṭū te'
    assert f('fall-imperfect-feminine',2,'baghi_sg')[2]=='lōṭā te'
    assert f('fall-imperfect-feminine',3,'rampur_pl')[2]=='lōṭā tī'
    assert f('fall-past-conditional-feminine',1,'baghi_pl')[2]=='lōṭde'
    assert f('go-present',2,'rampur_pl',138)[2]=='năs̲h̲au'
    assert 'indecl' in f('fall-conjunctive',1,'baghi_sg')[14].split()
    assert {'verb','fut','m'}<=set(f('fall-future',1,'baghi_sg')[14].split())
    assert not any('〃' in r[2] or 'etc.' in r[2] for r in data.values())
    reuse=[u for u in units() if u['source_review'].get('section')=='fall-future-reuse'][0]
    assert len(reuse['stated_reuse'])==6
    assert all(x['candidate_key'] in data for x in reuse['stated_reuse'])


def test_occurrence_specific_glyph_corrections_and_uncertainties():
    data=keyed()
    assert data['paired:sentence:141:8:baghi:1'][2]=='Ēhri pĭṭthi gāhri dzīn bănnhau'
    assert data['paired:sentence:142:17:baghi:1'][2]=='Ĭs ătshe pīṭā rŏs̲h̲ie gĭddh bănnhā'
    assert 'āhṭiwāḷe' in data['paired:sentence:142:22:rampuri:1'][2]
    assert data['paired:grammar:135:ordinal-suffix:1:shared:rampur_shared:1'][2]=='-āau'
    assert data['paired:grammar:139:give-present:1:rampur:rampur_sg:1'][2]=='deũ'
    ambiguity=[r for k,r in data.items() if k.startswith('paired:grammar:138:become-future:3:') and ':rampur_pl:' in k]
    assert [r[2] for r in ambiguity]==['ōā','auī']
    assert all('uncertain' in r[14].split() and not r[11] and 'unpunctuated' in r[6] for r in ambiguity)
    assert 'uncertain' in data['paired:sentence:142:19:baghi:1'][14].split()
    body=data['bailey1920rampur:p144:item:19']
    assert body[2]=='jĕā́' and 'uncertain' in body[14].split()
    assert 'stress interpretation is uncertain' in body[6]
    final_audit=[json.loads(x) for x in (P/'whole-proposed-audit.jsonl').read_text().splitlines()]
    recovered=next(a for a in final_audit if a.get('source_cell_key')==body[10])
    assert recovered['status']=='ingested' and recovered['historical_hold']['status']=='hold_typography'


def test_complete_responses_and_label_separation():
    data=keyed()
    s=[r for k,r in data.items() if k.startswith('paired:sentence:')]
    assert len(s)==45
    assert all({'sentential','multiword-expression'}<=set(r[14].split()) for r in s)
    assert data['paired:sentence:141:12:rampuri:1'][3]=='His brother (own) sister from (than) big, i.e. is bigger.'
    assert 'pp. 141–142' in data['paired:sentence:141:12:rampuri:1'][7]
    a=data['paired:sentence:142:18:baghi:1'];b=data['paired:sentence:142:18:baghi:2']
    assert a[2]=='Kūe ănda cīs̲h̲ găṛha' and b[2]=='Kūe ănda pāṇi găṛha'
    assert b[11]==a[10]
    assert data['paired:sentence:141:8:rampuri:1'][3].endswith('tighten.')
    assert data['paired:sentence:141:8:baghi:1'][3].endswith('bind.')
    for r in data.values():
        if r[10].startswith('paired:'):
            assert '(rel.)' not in r[3] and '(correl.)' not in r[3]
    rel=[r for k,r in data.items() if ':place-adverbs:4:' in k]
    assert len(rel)==3 and all('relative' in r[14].split() and r[3]=='where' for r in rel)


def test_closed_variants_and_no_cross_lect_control_emission():
    data=keyed()
    for p in [P,B]:
        own={r[10]:r for r in rows(p)}
        for r in own.values():
            assert not r[11] or (r[11] in own and r[11]!=r[10])
            assert not r[12] and not r[13]
    assert not any(':comparison-jubbal:' in k or ':comparison-kuari:' in k for k in data)
    generic=[r for r in data.values() if r[0]=='kc']
    assert {r[2] for r in generic}=={'gōhṛo','bāhī','bauhṇ','gauhr'}
    assert all(r[3]=='' and 'generic Koci' in r[6] for r in generic)
    assert not any(':go-heading:' in k and r[0]=='ba' for k,r in data.items())


def test_profile_nfc_nfd_and_registered_tags():
    for p in [P,B]:
        converter=Tokenizer(str(p/'whole-proposed-profile.txt'))
        for r in rows(p):
            assert len(r)==15
            assert set(r[14].split())<=GRAMMATICAL_TAGS|GENDER_TAGS
            a=converter(unicodedata.normalize('NFC',r[2]),column='IPA')
            b=converter(unicodedata.normalize('NFD',r[2]),column='IPA')
            assert '\ufffd' not in a+b
            assert unicodedata.normalize('NFC',a)==unicodedata.normalize('NFC',b)


def test_independent_legacy_mark_and_parsed_label_repairs():
    data=keyed()
    assert data['bailey1920baghi:p145:item:28:answer3'][2]=='dhūhĭ'
    changes=json.loads((P/'whole-legacy-corrections-20260927-pass2.json').read_text())['changes']
    assert len(changes)==11
    assert sum(c['changed_columns']==[2] for c in changes)==1
    for c in changes:
        actual=data[c['entry_key']]
        assert actual==c['after'] and actual[10]==c['before'][10]
        if c['changed_columns']!=[2]:
            assert actual[2]==c['before'][2]
            assert not any(label in actual[3] for label in ['relative','interrogative','correlative'])
    assert 'correlative' in data['bailey1920rampur:p145:item:66:answer3'][14].split()
    assert 'correlative' in data['bailey1920baghi:p145:item:66:answer4'][14].split()
