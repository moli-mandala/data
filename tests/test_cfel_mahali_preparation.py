"""Source topology checks; no assertion that unreviewed candidates are installable."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/cfel_mahali_2024'

def records():
    return [json.loads(s) for s in (ROOT/'candidates.jsonl').read_text().splitlines()]

def test_printed_native_ipa_mismatch_is_retained_and_flagged():
    rows={r['entry_key']:r for r in map(json.loads,(ROOT/'analysis-proposals.jsonl').read_text().splitlines())}
    row=rows['cfelmahali2024:p102:entry:4']
    assert row['native']=='মিঃ বার দিন'
    assert row['form']=='ɔlpɔ kɔjekʈi d̪in'
    assert 'uncertain' in row['tags']
    assert 'transcription:source-IPA-native-mismatch' in row['issues']
    assert any('Mahali pronunciation unresolved' in note for note in row['notes'])

def test_review_draft_accounts_for_every_entry_without_inventing_relations():
    import csv
    import importlib.util
    spec=importlib.util.spec_from_file_location('mahali_draft',ROOT/'prepare_draft.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows,audit=module.prepare()
    with (ROOT/'review-draft.csv').open() as stream:
        assert rows==list(csv.reader(stream))
    assert audit==[json.loads(s) for s in (ROOT/'draft-audit.jsonl').read_text().splitlines()]
    assert len(rows)==len(audit)==2451
    for row,record in zip(rows,audit):
        assert len(row)==15 and row[0]=='Mahali'
        assert row[10]==record['entry_key'] and row[7]==record['citation']
        assert row[2] and row[3] and '\x00' not in ''.join(row)
        assert not any(row[i] for i in (1,5,8,9,11,12,13))
        assert bool(row[4])==record['native_status'].startswith('accepted')
        assert record['status']=='included in 2451-row source CSV; compiled build deferred'
    keys={row[10] for row in rows}
    assert 'cfelmahali2024:p27:entry:6' in keys and 'cfelmahali2024:p28:entry:1' in keys

def test_installed_source_matches_review_and_whole_source_audit():
    import csv
    import importlib.util
    spec=importlib.util.spec_from_file_location('mahali_audit',ROOT/'audit_draft.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    subset,whole=module.verify()
    assert len(subset['records'])==whole['sample']==20
    assert whole['population']==2451 and whole['material_errors']==0
    installed=ROOT.parents[1]/'20260925-cfel-mahali.csv'
    with installed.open() as stream:
        rows=list(csv.reader(stream))
    with (ROOT/'review-draft.csv').open() as stream:
        assert rows==list(csv.reader(stream))
    assert len(rows)==2451
    assert (ROOT/'cfel-mahali.txt').read_bytes()==(ROOT.parents[4]/'conversion/cfel-mahali.txt').read_bytes()
    assert not any('\x00' in ''.join(row) for row in rows)

def test_reported_count_and_source_locators_reconcile():
    rows=records()
    assert len(rows)==len({r['entry_key'] for r in rows})==2451
    assert rows[0]['pdf_page']==10 and rows[-1]['pdf_page']==387
    for row in rows:
        assert row['entry_key']==f"cfelmahali2024:p{row['printed_page']}:entry:{row['page_item']}"
        assert row['status']=='candidate only; segmentation and glyph audit pending'
        assert row['raw_ipa'] and row['candidate_english'] and row['raw_grammar']

def test_missing_ipa_delimiter_is_source_exception_not_lost_entry():
    fig=next(r for r in records() if r['entry_key']=='cfelmahali2024:p380:entry:2')
    assert fig['raw_ipa']=='ɖũmbɔr' and fig['candidate_english']=='Fig'
    assert '-ɖũmbɔr/' in fig['raw_lexical_block']
    assert any(x.startswith('source-punctuation:') for x in fig['issues'])

def test_description_absence_and_native_damage_remain_visible():
    rows=records()
    absent=[r for r in rows if not r['has_description']]
    assert len(absent)==10
    assert {r['candidate_english'] for r in absent}==set(map(str,range(1,11)))
    assert sum('\x00' in r['raw_native'] for r in rows)==1176
    for row in rows:
        assert ('\x00' in row['raw_native'])==any(x.startswith('native-font:') for x in row['issues'])
        assert '\x00' not in row['raw_ipa']
    assert any('\n' in r['raw_grammar'] for r in rows)

def test_transcription_inventory_and_review_preserve_unusual_source_marks():
    import importlib.util
    spec=importlib.util.spec_from_file_location('mahali_inventory',ROOT/'inventory_transcription.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.inventory()
    assert result==json.loads((ROOT/'transcription-inventory.json').read_text())
    assert len(result['symbols'])==53 and len(result['review_queue'])==28
    by_key={r['entry_key']:r for r in records()}
    reviews=[json.loads(s) for s in (ROOT/'glyph-review.jsonl').read_text().splitlines()]
    assert len(reviews)==3
    for review in reviews:
        assert review['raw_ipa']==review['reviewed_ipa']==by_key[review['entry_key']]['raw_ipa']
        assert review['issues']==['transcription:unusual-source-diacritic-placement']

def test_positioned_marks_preserve_affricates_without_touching_native_shaping():
    import importlib.util
    spec=importlib.util.spec_from_file_location('mahali_positioned',ROOT/'extract_positioned.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def c(text,x0,x1,top):return dict(text=text,x0=x0,x1=x1,top=top,bottom=top+12,doctop=top,fontname='IPA')
    chars=[c('/',0,3.3,10),c('t',3.3,6.6,10),c('͡',3.13,3.13,6.85),c('ʃ',6.6,10.6,10),c('্',11,11,10)]
    positioned,audit=module.position_marks(chars)
    assert len(audit)==1 and audit[0]['anchor']=='t'
    assert positioned[2]['top']==10 and 3.3<positioned[2]['x0']<6.6
    assert positioned[4]==chars[4] and chars[2]['top']==6.85

def test_domain_boundaries_do_not_assign_an_entire_page_to_next_section():
    data=json.loads((ROOT/'domains.json').read_text())
    assert data['reported_domains']==52 and data['observed_lexical_domains']==50
    assert sum(s['entries'] for s in data['sections'])==2451
    assert [s['toc_number'] for s in data['sections']]==list(range(3,53))
    assigned={r['entry_key']:r['domain'] for r in data['assignments']}
    assert set(assigned)=={r['entry_key'] for r in records()}
    assert assigned['cfelmahali2024:p306:entry:2']=='Measurements'
    assert assigned['cfelmahali2024:p82:entry:4']=='Cardinal Numbers'
    assert assigned['cfelmahali2024:p82:entry:7']=='Causative Verb'

def test_analysis_preserves_phrase_scope_and_meaning():
    import importlib.util
    from tags import GRAMMATICAL_TAGS
    spec=importlib.util.spec_from_file_location('mahali_analysis',ROOT/'prepare_analysis.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=module.prepare()
    assert rows==[json.loads(s) for s in (ROOT/'analysis-proposals.jsonl').read_text().splitlines()]
    assert len(rows)==2451
    by_key={r['entry_key']:r for r in rows}
    phrase=by_key['cfelmahali2024:p100:entry:8']
    assert phrase['tags']==['multiword-expression']
    assert phrase['source_component_labels']==['Cardinal','Noun','Preposition','Noun']
    assert by_key['cfelmahali2024:p70:entry:2']['gloss']=='Ninety-three'
    assert by_key['cfelmahali2024:p306:entry:2']['gloss']=='Grain (unit of weight)'
    assert by_key['cfelmahali2024:p307:entry:6']['gloss']=='Glass (material)'
    assert by_key['cfelmahali2024:p307:entry:6']['source_gloss']=='Glass'
    assert by_key['cfelmahali2024:p153:entry:2']['gloss']=='Belly (flower)'
    assert by_key['cfelmahali2024:p153:entry:2']['source_gloss']=='Belly'
    assert by_key['cfelmahali2024:p283:entry:3']['gloss']=='Belly'
    a,b=(by_key[f'cfelmahali2024:p78:entry:{n}'] for n in (6,7))
    assert a['form']==b['form'] and a['gloss']=='Thirty-four' and b['gloss']=='Thirty-nine'
    assert all('uncertain' in x['tags'] and 'gloss:source-identical-form-for-different-numerals' in x['issues'] for x in (a,b))
    assert 'caus' in by_key['cfelmahali2024:p82:entry:7']['tags']
    assert all(set(r['tags'])<=GRAMMATICAL_TAGS for r in rows)
    assert sum('uncertain' in r['tags'] for r in rows)==22
    assert sum(r['native_status'].startswith('pending') for r in rows)==0
    accepted={r['entry_key']:r['native'] for r in rows if r['native']}
    ledger=[json.loads(s) for s in (ROOT/'native-recovery-review.jsonl').read_text().splitlines()]
    assert len(accepted)==len(ledger)==2451
    assert by_key['cfelmahali2024:p282:entry:5']['form']=='mæd̪ bæŋged̪-dʒapid̪renaʔ hart̪a'
    assert accepted=={r['entry_key']:r['native'] for r in ledger}
    assert accepted['cfelmahali2024:p10:entry:1']=='আঙটি'
    assert accepted['cfelmahali2024:p198:entry:1']=='ঘণ্টা রু'
    assert accepted['cfelmahali2024:p10:entry:2'].endswith('্')
    assert accepted['cfelmahali2024:p13:entry:1']=='চুরি'
    # Online alternates are evidence, not automatically printed variants.
    assert accepted['cfelmahali2024:p13:entry:4']=='জুত্তা'
    assert accepted['cfelmahali2024:p14:entry:1']=='টিকলি'
    assert accepted['cfelmahali2024:p16:entry:3']=='পিন্টুল'
    assert accepted['cfelmahali2024:p187:entry:6']=='উপর দম্ সাঁহেদ্'
    assert all(not r['native'] for r in rows if r['native_status'].startswith('pending'))

def test_source_local_profile_covers_all_forms_without_losing_marks():
    import unicodedata
    from segments import Tokenizer
    tokenizer=Tokenizer(str(ROOT/'cfel-mahali.txt'))
    cv=lambda s:unicodedata.normalize('NFC',tokenizer(s,column='IPA').replace(' ','').replace('#',' '))
    rows=[json.loads(s) for s in (ROOT/'analysis-proposals.jsonl').read_text().splitlines()]
    for row in rows:
        assert '�' not in cv(row['form']),row['entry_key']
        assert cv(row['form'])==cv(unicodedata.normalize('NFD',row['form']))
    assert cv('t͡ʃʰat̪')=='cʰat̪'
    assert cv('d͡ʒapit̪')=='japit̪'
    assert cv('maraŋ m̜aɲ')=='maraŋ m̜añ'
    assert cv('mi?')=='mi?' and cv('miʔ')=='miʔ'
    assert cv('lɔr.lɔpɔr')=='lɔr.lɔpɔr'

def test_publisher_comparison_is_typographic_not_phonological_matching():
    import importlib.util
    spec=importlib.util.spec_from_file_location('mahali_publisher',ROOT/'compare_publisher.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.ipa('/d͡ʒaŋ/')==module.ipa('/ʤaŋ/')
    assert module.ipa('mi?')!=module.ipa('miʔ')
    assert module.ipa('t̪a')!=module.ipa('ta')
    assert module.ipa('maraŋ m̜aɲ')!=module.ipa('maraŋ maɲ')
    assert module.ipa('æ')!=module.ipa('a')
    assert module.query_text('Ninety-\nthree')=='Ninety-three'
    assert module.query_text('Social\nScience')=='Social Science'
    assert module.ipa('mæd̪ bæŋged̪-\ndʒapid̪renaʔ hart̪a')==module.ipa('/mæd̪ bæŋged̪-dʒapid̪renaʔ hart̪a/')

def test_publisher_download_retries_are_bounded_and_skip_permanent_errors(monkeypatch):
    import importlib.util
    import io
    import pytest
    from urllib.error import HTTPError, URLError
    spec=importlib.util.spec_from_file_location('mahali_download',ROOT/'compare_publisher.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sleeps=[];calls=[]
    monkeypatch.setattr(module.time,'sleep',sleeps.append)
    def transient(*args,**kwargs):
        calls.append(1)
        if len(calls)<3:raise URLError('temporary')
        return io.BytesIO(b'{}')
    monkeypatch.setattr(module,'urlopen',transient)
    assert module.read_response('request')==b'{}'
    assert len(calls)==3 and sleeps==[1,2]

    calls.clear();sleeps.clear()
    def permanent(*args,**kwargs):
        calls.append(1);raise HTTPError('url',404,'missing',{},None)
    monkeypatch.setattr(module,'urlopen',permanent)
    with pytest.raises(HTTPError):module.read_response('request')
    assert len(calls)==1 and sleeps==[]
    calls.clear()
    def failing(*args,**kwargs):
        calls.append(1);raise URLError('still failing')
    monkeypatch.setattr(module,'urlopen',failing)
    with pytest.raises(URLError):module.read_response('request')
    assert len(calls)==3 and sleeps==[1,2]

def test_accepted_native_readings_retain_matching_publisher_and_print_evidence():
    import importlib.util
    import unicodedata
    spec=importlib.util.spec_from_file_location('mahali_evidence',ROOT/'compare_publisher.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    proposals={r['entry_key']:r for r in map(json.loads,(ROOT/'publisher-proposals.jsonl').read_text().splitlines())}
    ledger=[json.loads(s) for s in (ROOT/'native-recovery-review.jsonl').read_text().splitlines()]
    for row in ledger:
        if row.get('method')=='direct-print-transcription':
            editions={r['entry_key']:r for r in map(json.loads,(ROOT/'publisher-edition-review.jsonl').read_text().splitlines())}
            edition=editions[row['entry_key']]
            assert row['native']==edition['printed_native']
            assert row['evidence'] and 'visually confirmed' in row['status']
            assert row['pdf_sha256']==json.loads((ROOT/'source-manifest.json').read_text())['pdf_sha256']
            assert row['edition_review']=='publisher-edition-review.jsonl'
            assert edition['status']=='resolved; printed reading accepted'
            if edition.get('difference')=='publisher-ipa-prefix-corruption':
                assert row['native']==edition['publisher_native']
                prefix=edition.get('prefix_mark','̹')
                assert prefix in ('̹','̘')
                assert edition['publisher_ipa']==prefix+'/'+edition['printed_ipa']+'/'
            elif edition.get('difference')=='publisher-ipa-segment-difference':
                assert row['native']==edition['publisher_native']
                assert edition['printed_ipa']!=edition['publisher_ipa'].strip('/')
            else:
                assert row['native']!=edition['publisher_native']
            assert 'api_record' not in row
            continue
        proposal=proposals[row['entry_key']];api=row['api_record']
        assert row['evidence'] and 'visually confirmed' in row['status']
        assert row['api_snapshot_sha256']==proposal['response_sha256']
        assert api['id'] in {c['id'] for c in proposal['api_candidates']}
        assert module.ipa(api['ipa'])==module.ipa(proposal['source_ipa'])
        assert module.domain(api['domain_name'])==module.domain(proposal['source_domain'])
        assert row['native']==unicodedata.normalize('NFC',' '.join(api['word'].split()))


def test_curry_preserves_printed_reading_when_online_edition_differs():
    rows={r['entry_key']:r for r in map(json.loads,(ROOT/'analysis-proposals.jsonl').read_text().splitlines())}
    row=rows['cfelmahali2024:p155:entry:4']
    assert row['native']=='উতু' and row['form']=='ut̪u'
    assert row['gloss']=='Curry'
    assert any('অতঅ /ɔt̪ɔ/' in note for note in row['notes'])


def test_beverage_preserves_printed_n_against_online_palatal_nasal():
    rows={r['entry_key']:r for r in map(json.loads,(ROOT/'analysis-proposals.jsonl').read_text().splitlines())}
    row=rows['cfelmahali2024:p164:entry:2']
    assert row['native']=='নুআঃ' and row['form']=='nuaʔ'
    assert row['gloss']=='Beverage'
    assert any('ঞুআঃ /ɲuaʔ/' in note for note in row['notes'])


def test_taste_noun_keeps_its_printed_form_and_part_of_speech():
    rows={r['entry_key']:r for r in map(json.loads,(ROOT/'analysis-proposals.jsonl').read_text().splitlines())}
    noun=rows['cfelmahali2024:p173:entry:2'];verb=rows['cfelmahali2024:p198:entry:6']
    assert noun['native']=='স্যাবেল' and noun['form']=='sæbel'
    assert 'noun' in noun['tags'] and 'verb' not in noun['tags']
    assert verb['native']=='চাখা' and 'verb' in verb['tags']
    assert noun['form']!=verb['form']


def test_ripe_keeps_printed_ipa_when_publisher_prefix_is_corrupt():
    rows={r['entry_key']:r for r in map(json.loads,(ROOT/'analysis-proposals.jsonl').read_text().splitlines())}
    row=rows['cfelmahali2024:p220:entry:6']
    assert row['native']=='ব্যালেআঃ' and row['form']=='bæleaʔ'
    assert 'adj' in row['tags']
    assert 'publisher-IPA:leading-corrupt-combining-mark' in row['issues']
    edition={r['entry_key']:r for r in map(json.loads,(ROOT/'publisher-edition-review.jsonl').read_text().splitlines())}[row['entry_key']]
    assert edition['publisher_ipa']=='̹/bæleaʔ/' and edition['printed_ipa']=='bæleaʔ'


def test_dark_skinned_keeps_printed_vowel_against_online_edition():
    rows={r['entry_key']:r for r in map(json.loads,(ROOT/'analysis-proposals.jsonl').read_text().splitlines())}
    row=rows['cfelmahali2024:p238:entry:2']
    assert row['native']=='হেঁদে হরম' and row['form']=='hẽd̪e hɔrmɔ'
    assert 'adj' in row['tags']
    assert any('হ্যাঁদে হরম /hæ̃d̪e hɔrmɔ/' in note for note in row['notes'])


def test_house_keeps_printed_retroflex_against_online_edition():
    rows={r['entry_key']:r for r in map(json.loads,(ROOT/'analysis-proposals.jsonl').read_text().splitlines())}
    house=rows['cfelmahali2024:p251:entry:5'];home=rows['cfelmahali2024:p251:entry:4']
    assert house['native']==home['native']=='অরাঃ'
    assert house['form']=='ɔɽaʔ' and home['form']=='ɔraʔ'
    assert any('publisher online edition gives /ɔraʔ/' in note for note in house['notes'])


def test_skeleton_and_right_hand_retain_printed_forms_despite_api_gaps():
    rows={r['entry_key']:r for r in map(json.loads,(ROOT/'analysis-proposals.jsonl').read_text().splitlines())}
    skeleton=rows['cfelmahali2024:p269:entry:5']
    hand=rows['cfelmahali2024:p272:entry:5']
    assert skeleton['native']=='কুঙ্কাল' and skeleton['form']=='kuŋkal'
    assert 'publisher-IPA:leading-corrupt-combining-mark' in skeleton['issues']
    assert hand['native']=='জজম্ তি' and hand['form']=='d͡ʒɔd͡ʒɔm t̪i'
    assert 'publisher:entry-absent-from-exact-query' in hand['issues']
    editions={r['entry_key']:r for r in map(json.loads,(ROOT/'publisher-edition-review.jsonl').read_text().splitlines())}
    assert editions[skeleton['entry_key']]['publisher_ipa']=='̘/kuŋkal/'
    assert editions[hand['entry_key']]['api_result_count']==0

def test_description_restrictions_survive_into_proposed_rows():
    import importlib.util
    spec=importlib.util.spec_from_file_location('mahali_restricted_draft',ROOT/'prepare_draft.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows,_=module.prepare();by={r[10]:r for r in rows}
    def row(page,item):return by[f'cfelmahali2024:p{page}:entry:{item}']
    assert 'already completed event performed by the speaker' in row(115,5)[6]
    assert not {'pret','past','perfect','pfv'} & set(row(115,5)[14].split())
    assert 'cooked pork' in row(172,5)[6]
    assert 'cooked chicken meat' in row(172,1)[6]
    assert 'cooked rat meat' in row(157,5)[6]
    assert 'Source component labels: Determiner + Noun.' in row(96,3)[6]
    assert 'near the speaker' in row(96,3)[6]
    assert 'distant from the speaker' in row(109,7)[6]
    assert 'impv' in row(116,8)[14].split()
    assert 'pl' in row(274,3)[14].split()
    assert 'calcium oxide' in row(308,5)[6]
    assert 'husband’s sister' in row(290,7)[6]
    assert 'husband’s brother' in row(295,3)[6]

def test_sport_head_definition_differences_remain_source_attributed():
    import csv
    by={r[10]:r for r in csv.reader((ROOT/'review-draft.csv').open())}
    def row(page,item):return by[f'cfelmahali2024:p{page}:entry:{item}']
    assert row(341,1)[3]=='Kite'
    assert 'kite-flying as a game' in row(341,1)[6]
    assert 'game played with dice' in row(347,6)[6]
    assert 'both the thrown spear and javelin as a sport' in row(344,4)[6]
    assert 'umpire in cricket' in row(344,1)[6]

def test_cultural_timing_and_source_definition_mismatches_are_attributed():
    import csv
    by={r[10]:r for r in csv.reader((ROOT/'review-draft.csv').open())}
    def row(page,item):return by[f'cfelmahali2024:p{page}:entry:{item}']
    assert 'within ten days after birth' in row(334,5)[6]
    assert 'week-long tree-planting festival in July' in row(139,3)[6]
    assert 'festival in March' in row(139,1)[6]
    assert 'soon after moving into a new home' in row(336,3)[6]
    assert row(356,8)[3]=='Blue moon'
    assert 'source describes the monthly full moon' in row(356,8)[6]
    assert row(259,3)[3]=='Storey' and 'intermediate level' in row(259,3)[6]

def test_full_description_review_preserves_mismatches_without_rewriting_heads():
    import csv
    by={r[10]:r for r in csv.reader((ROOT/'review-draft.csv').open())}
    def row(page,item):return by[f'cfelmahali2024:p{page}:entry:{item}']
    assert row(316,3)[3]=='Thunder' and 'lightning' in row(316,3)[6]
    assert row(215,1)[3]=='Find' and 'searching' in row(215,1)[6]
    assert row(183,5)[3]=='Extremely' and 'very well' in row(183,5)[6]
    assert 'two to five days' in row(97,8)[6]
    assert 'following number' in row(74,1)[6] and row(74,1)[3]=='Fifteen'
    assert 'woollen neck wrap' in row(11,4)[6]
