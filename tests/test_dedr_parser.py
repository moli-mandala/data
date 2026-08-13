from data.dedr.abbrevs import abbrevs
from data.dedr.parser_utils import (
    clean_form_html,
    detach_leading_grammatical_note,
    detach_trailing_forward_grammatical_note,
    grammatical_tag,
    grammatical_tags,
    is_prose_misparse,
    mark_unbolded_gender_forms,
    mark_unbolded_compact_forms,
    marker_only_tags,
    propagate_shared_glosses,
    split_alternatives,
    split_forms,
    split_language_spans,
    split_tagged_forms,
    strip_grammatical_markers,
)


def test_language_spans_survive_malformed_bold_markup():
    html = (
        '<b><i>Ta.</i> caṭṭi</b> to destroy, ruin, <b>kill. '
        '<i>Ka.</i> caṭṭu</b> destruction. <b><i>Tu.</i> caṭṭu</b> end.'
    )
    _, spans = split_language_spans(html, abbrevs)

    assert [label for label, _ in spans] == ['Ta', 'Ka', 'Tu']
    assert 'kill' in spans[0][1]
    assert 'caṭṭu' in spans[1][1]


def test_language_spans_recognize_plain_uncertain_language():
    html = '<b><i>Nk.</i> ōbaṛ-</b> to be got. ? Go. (Mu.) <b>akna</b> room.'
    _, spans = split_language_spans(html, abbrevs)

    assert [label for label, _ in spans] == ['Nk', 'Go']
    assert spans[1][1].lstrip().startswith('(Mu.)')


def test_language_spans_preserve_note_before_malformed_boundary():
    html = (
        '<b><i>Pe.</i> tṛīp-</b> to turn round <b><i>(tr.). Kui</i> '
        'tlīpa</b> to twist.'
    )
    _, spans = split_language_spans(html, abbrevs)

    assert [label for label, _ in spans] == ['Pe', 'Kui']
    assert '<i>(tr.)</i>' in spans[0][1]
    assert grammatical_tags(spans[0][1]) == ['tr']


def test_language_span_accepts_source_after_label():
    html = '<b><i>Ta.</i> kuvaḷai</b> socket. <i><b>Ma.</b> (DCV)</i> <b>kuvaḷa</b> id.'
    _, spans = split_language_spans(html, abbrevs)

    assert [label for label, _ in spans] == ['Ta', 'Ma']
    assert spans[1][1].lstrip().startswith('(DCV)')


def test_form_cleanup_splits_semicolon_and_grammar_label():
    form, gloss = clean_form_html('<i>fem.</i> pulaicci, pulaitti; pulaimai', 'baseness')

    assert split_forms(form) == ['pulaicci', 'pulaitti', 'pulaimai']
    assert gloss == 'baseness'


def test_form_cleanup_moves_italic_botanical_gloss():
    form, gloss = clean_form_html('virigi <i>Cordia sebestena.</i>', '')

    assert form == 'virigi'
    assert gloss == 'Cordia sebestena.'


def test_prose_bolded_by_malformed_html_is_not_a_form():
    assert is_prose_misparse('kill.')
    assert is_prose_misparse('split (<i>tr.</i>)')
    assert not is_prose_misparse('caṭṭi (-pp-, -tt-)')


def test_slash_inside_parenthetical_is_not_an_alternative():
    assert split_alternatives('goṟon (pl. goṟoku/goṟonku)') == [
        'goṟon (pl. goṟoku/goṟonku)'
    ]
    assert split_alternatives('pinda/pinde') == ['pinda', 'pinde']


def test_unbolded_feminine_form_is_marked_as_a_lemma():
    assert mark_unbolded_gender_forms(
        '<b>aṭiyān</b> slave, servant; <i>fem.</i> aṭiyātti.'
    ) == (
        '<b>aṭiyān</b> slave, servant; '
        '<b><i>fem.</i> aṭiyātti</b>.'
    )


def test_already_bold_feminine_run_is_unchanged():
    html = '<b><i>fem.</i> akattōḷ; akaṅkai</b> palm of hand'
    assert mark_unbolded_gender_forms(html) == html


def test_existing_bold_gender_run_is_not_nested_when_span_is_unbalanced():
    html = 'pul</b>; <b><i>fem.</i> pulaicci, pulaitti; pulaimai</b> baseness'
    assert mark_unbolded_gender_forms(html) == html


def test_trailing_grammatical_label_is_handed_to_next_form():
    tags, gloss = detach_trailing_forward_grammatical_note(
        'to live; <i>caus.</i> (SR.)'
    )
    assert tags == ['caus']
    assert gloss == 'to live; (SR.)'
    assert split_tagged_forms('pisusānā', tags) == [('pisusānā', ['caus'])]


def test_forward_gender_label_applies_only_until_semicolon():
    assert split_tagged_forms('tagmaḷů; daṅguni', ['f']) == [
        ('tagmaḷů', ['f']),
        ('daṅguni', []),
    ]


def test_grammar_marked_form_at_language_boundary_is_not_prose():
    raw = '<i>fem.</i> polati.'
    assert clean_form_html(raw, '') == ('polati', '')
    assert not is_prose_misparse(raw)


def test_trailing_pos_marker_applies_to_entire_form_group():
    assert split_tagged_forms('akkakka, akkoḷu <i>n.</i>') == [
        ('akkakka', ['noun']),
        ('akkoḷu', ['noun']),
    ]


def test_multiple_grammar_marked_forms_at_language_boundary_are_not_prose():
    raw = '<i>fem.</i> kiṟiyaḷ; <i>epic. pl.</i> kiṟiyar.'
    assert not is_prose_misparse(raw)
    assert split_tagged_forms(raw) == [
        ('kiṟiyaḷ', ['f']),
        ('kiṟiyar', ['pl']),
    ]


def test_shared_glosses_propagate_across_languages_in_both_directions():
    rows = [
        ['Tam', 'd1', 'a', 'one thousand'],
        ['Mal', 'd1', 'b', ''],
        ['Kota', 'd1', 'c', ''],
    ]
    assert [row[3] for row in propagate_shared_glosses(rows)] == [
        'one thousand', 'one thousand', 'one thousand'
    ]

    rows = [
        ['Kannada', 'd2', 'a', ''],
        ['Telugu', 'd2', 'b', 'palate'],
    ]
    assert [row[3] for row in propagate_shared_glosses(rows)] == ['palate', 'palate']


def test_parenthetical_morphology_is_not_used_as_shared_gloss():
    rows = [
        ['Tam', 'd1', 'āyiram', 'the number 1,000'],
        ['Kota', 'd1', 'cavrm', '(obl. cavrt-)'],
        ['Toda', 'd1', 'sofer', ''],
    ]
    propagate_shared_glosses(rows)
    assert rows[2][3] == 'the number 1,000'
    assert rows[1][2] == 'cavrm (obl. cavrt-)'
    assert rows[1][3] == 'the number 1,000'


def test_explicit_section_fallback_fills_a_definitionless_compact_group():
    rows = [
        ['Mal', 'd1', 'veṇṇa', ''],
        ['Kota', 'd1', 'veṇ', ''],
    ]
    propagate_shared_glosses(rows, 'butter')
    assert [row[3] for row in rows] == ['butter', 'butter']

    displaced = [['Kuwi', 'd2', 'kandrū', '(Mah.) kanˀ eri']]
    propagate_shared_glosses(displaced, 'tears')
    assert displaced == [['Kuwi', 'd2', 'kandrū (Mah.) kanˀ eri', 'tears']]


def test_compact_unbolded_form_list_is_marked_without_treating_prose_as_forms():
    assert mark_unbolded_compact_forms('ilanta, lanta.') == '<b>ilanta, lanta</b>.'
    prose = 'bē salt, piquancy, spirit, flavour.'
    assert mark_unbolded_compact_forms(prose) == prose


def test_scientific_abbreviation_is_not_a_form():
    assert is_prose_misparse('A. tristis')


def test_grammatical_markup_maps_to_schema_tags():
    assert grammatical_tags(
        '<i>fem.</i> kir̤avi (<i>pl.</i> kir̤avikaḷ)',
        'to split (<i>tr.</i>); <i>n.</i> a split',
    ) == ['f', 'pl', 'tr', 'noun']
    assert grammatical_tags('<i>sg. m.</i>', '<i>pl. neut.</i>') == [
        'sg', 'm', 'pl', 'n'
    ]
    assert strip_grammatical_markers(
        'to split (<i>tr.</i>); <i>n.</i> a split'
    ) == 'to split; a split'


def test_grammatical_markers_are_case_sensitive():
    assert grammatical_tag('tr.') == 'tr'
    assert grammatical_tag('Tr.') == ''
    assert grammatical_tag('Voc.') == ''
    assert marker_only_tags('<i>intr., tr.</i>') == ['intr', 'tr']
    assert marker_only_tags('aṛkm (<i>obl.</i> aṛkt-)') == []


def test_leading_transitivity_marker_belongs_to_previous_lemma():
    tags, form = detach_leading_grammatical_note(
        '(<i>tr.</i>); aṭukkaḷai'
    )
    assert tags == ['tr']
    assert form == 'aṭukkaḷai'
    tags, form = detach_leading_grammatical_note(
        '<i>(intr.)</i>; pakup- (pakut-)'
    )
    assert tags == ['intr']
    assert form == 'pakup- (pakut-)'


def test_tags_are_associated_with_individual_forms():
    assert split_tagged_forms(
        'aṭanna, <i>fem.</i> aṭī, <i>hon.</i> aṭō'
    ) == [
        ('aṭanna', []),
        ('aṭī', ['f']),
        ('aṭō', ['honorific']),
    ]
    assert split_tagged_forms(
        '<i>fem.</i> pulaicci, pulaitti; pulaimai'
    ) == [
        ('pulaicci', ['f']),
        ('pulaitti', ['f']),
        ('pulaimai', []),
    ]


def test_part_of_speech_marker_carries_across_form_list():
    assert split_tagged_forms(
        '<i>adj.</i> anta, aṉai, āṉa; akkaṭa'
    ) == [
        ('anta', ['adj']),
        ('aṉai', ['adj']),
        ('āṉa', ['adj']),
        ('akkaṭa', ['adj']),
    ]
