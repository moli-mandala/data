"""Regression fixtures from source pages; proposal parser is not installation-ready."""
import importlib.util
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "data/other/forms/raw_data/sheth_parse.py"
SPEC = importlib.util.spec_from_file_location("sheth_parse", PATH)
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


def test_quoted_meaning_is_not_removed_as_example():
    raw = "<div><hw><b>उवह</b> <b>uvaha</b></hw><category>अ</category><etymology>दे</etymology>'देखो' अर्थं को बतलातेवाला अव्यय<reference>(षड्)</reference>।</div>"
    result = S.parse_article(raw, 181, 31)
    assert result["rows"][0]["row"][3] == "'देखो' अर्थं को बतलातेवाला अव्यय"
    assert result["rows"][0]["row"][14] == "indecl"
    assert result["printed_references"] == ["(षड्)"]


def test_source_numbered_sense_gender_does_not_leak():
    raw = "<div><hw><b>चलचल</b> <b>calacala</b></hw><category>वि</category><definition>१ चंचल, अस्थिर; 'चलचलयकोडिमोडणकराइं नयणाइं तरुणीणं'<reference>(वज्जा ६०)</reference></definition><definition>२ पुं. घी में तली जाती हुई चीज का पहला तीन घान<reference>(निचू ४)</reference></definition></div>"
    rows = S.parse_article(raw, 320, 22)["rows"]
    assert len(rows) == 2
    assert rows[0]["row"][3] == "चंचल, अस्थिर"
    assert rows[0]["row"][14] == "adj"
    assert rows[1]["row"][14] == "noun m"
    assert rows[0]["row"][10] != rows[1]["row"][10]


def test_unknown_grammar_tag_cannot_discard_definition():
    raw = "<div><hw><b>अ</b><b>a</b></hw><category>१ गाढा लाल ।</category></div>"
    result = S.parse_article(raw, 1, 1)
    assert "गाढा लाल" in result["rows"][0]["row"][3]
    assert result["review"]


def test_printed_headword_alternates_have_stable_local_links():
    raw = "<div><hw><b>अइगच्छ</b><b>aigaccha</b>, <b>अइगम</b><b>aigama</b></hw><category>सक</category>अतिक्रमण करना</div>"
    rows = S.parse_article(raw, 2, 1)["rows"]
    assert len(rows) == 2
    assert rows[1]["row"][11] == rows[0]["row"][10]
    assert rows[1]["row"][14] == "verb tr"


def test_broken_native_roman_pair_is_audited_and_excluded():
    result = S.parse_article("<div><hw><b>अ</b></hw>meaning</div>", 1, 9)
    assert result["status"] == "excluded"
    assert not result["rows"]


def test_embedded_subentry_is_not_marked_clean():
    raw = "<div><hw><b>णड</b><b>ṇaḍa</b></hw><category>पुं</category><definition>१ नट । °खाइया स्त्री दीक्षा-विशेष</definition></div>"
    result = S.parse_article(raw, 379, 47)
    assert "structure:embedded-subentry-or-reference" in result["rows"][0]["review"]
    assert result['rows'][0]['row'][3] == 'नट'
    assert result['unresolved_subentry_regions'][0]['text'] == '°खाइया स्त्री दीक्षा-विशेष'


def test_compound_equivalent_is_not_assigned_to_parent_head():
    raw = "<div><hw><b>णड</b><b>ṇaḍa</b></hw><category>पुं</category><etymology>नट</etymology><definition>१ नट । °खाइया स्त्री<etymology>°खादिता</etymology>दीक्षा-विशेष</definition></div>"
    result = S.parse_article(raw, 379, 47)
    assert result["rows"][0]["row"][9] == "नट"
    assert result["etymology_segments"] == ["नट", "°खादिता"]


def test_crossreference_quotation_is_usage_not_gloss():
    raw = "<div><hw><b>पोम</b><b>pōma</b></hw><var>देखो पउम;</var> 'जहा पोमं जले जायं'<reference>(उत्त २५, २७)</reference>।</div>"
    result = S.parse_article(raw,618,5)
    assert result['rows'][0]['row'][3] == ''
    assert result['rows'][0]['row'][6] == "'जहा पोमं जले जायं'"


def test_nested_meaning_does_not_create_a_blank_duplicate_sense():
    raw = "<div><hw><b>अ</b><b>a</b></hw><category>अ</category><definition><meaning>१ निषेध</meaning><reference>(सुर ७)</reference></definition><definition><meaning>२ विरोध</meaning></definition></div>"
    result = S.parse_article(raw, 1, 4)
    assert [r['row'][3] for r in result['rows']] == ['निषेध', 'विरोध']


def test_bound_head_equivalent_stays_with_bound_head():
    raw = "<div><hw><b>°अ</b><b>°a</b></hw><category>वि</category><etymology>°ज</etymology>उत्पन्न, जात</div>"
    result = S.parse_article(raw, 1, 6)
    assert result['rows'][0]['row'][9] == '°ज'
    assert 'etymology:subentry-scope-pending' not in result['review']


def test_nested_category_applies_only_to_its_sense():
    raw = '<div><hw><b>अ</b><b>a</b></hw><category>वि</category><definition>१ पहला</definition><definition>२ <category>पुं</category>दूसरा</definition></div>'
    rows = S.parse_article(raw, 1, 1)['rows']
    assert [r['row'][14] for r in rows] == ['adj', 'noun m']


def test_nested_lect_does_not_relabel_other_senses():
    raw = '<div><hw><b>अ</b><b>a</b></hw><category>वि</category><definition>१ पहला</definition><definition><meaning>२ <category>(अप) पुं</category>दूसरा</meaning></definition></div>'
    rows = S.parse_article(raw, 1, 1)['rows']
    assert [r['row'][0] for r in rows] == ['Pk', 'Ap']
    assert rows[1]['lect_labels'] == ['अप']
