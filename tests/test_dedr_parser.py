import csv
from pathlib import Path

from data.dedr.abbrevs import abbrevs, latin_genera
from data.dedr.parser_utils import (
    _LEMMA,
    clean_form_html,
    detach_inline_latin_glosses,
    detach_leading_grammatical_note,
    detach_trailing_forward_grammatical_note,
    grammatical_tag,
    grammatical_tags,
    is_prose_misparse,
    mark_plain_language_labels,
    mark_unbolded_gender_forms,
    mark_unbolded_compact_forms,
    marker_only_tags,
    propagate_shared_glosses,
    repair_botanical_markup,
    split_alternatives,
    split_forms,
    split_language_spans,
    split_tagged_forms,
    strip_grammatical_markers,
)


ROOT = Path(__file__).parents[1]


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


# Botanical authorities DEDR copies from Kittel's Latin names; a bare one is never a form.
BOTANICAL_AUTHORITIES = {
    "Arn", "Benth", "Blume", "Burm", "DC", "Dalz", "Gaertn", "Heyne", "Hook", "Juss", "Lam",
    "Lin", "Linn", "Lour", "Roxb", "Retz", "Spreng", "Thw", "Wall", "Wight", "Willd",
}


def test_botanical_authority_is_not_parsed_as_a_form():
    with (ROOT / "data/dedr/dedr_new.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    assert not [row for row in rows if row[2].rstrip(".") in BOTANICAL_AUTHORITIES]
    # 2399: DSAL drops "savunte a kind of cucumber, Cucumis utilissimus" before "Roxb."
    kannada = [row for row in rows if row[1] == "d2399" and row[0] == "Kannada"]
    assert [row[2] for row in kannada] == ["savute", "saute", "savate", "savati", "savunte"]
    assert {row[3] for row in kannada} == {"a kind of cucumber, <i>Cucumis utilissimus</i> Roxb"}
    # 2628: the bracketed correction stays in the gloss and the next form keeps its dialect
    gondi = {row[2]: row for row in rows if row[1] == "d2628"}
    assert gondi["sarēkā"][3].endswith("Roxb. [better, <i>B. latifolia</i> Roxb.]")
    assert gondi["saṛēka"][6] == "Ch."


def test_compiled_graph_has_no_botanical_authority_forms():
    forms_path = ROOT / "cldf/forms.csv"
    if not forms_path.exists():
        return
    with forms_path.open(encoding="utf-8", newline="") as stream:
        bad = [
            (row["ID"], row["Language_ID"], row["Form"])
            for row in csv.DictReader(stream)
            if row["Source"].split(";")[0] == "dedr"
            and row["Original"].rstrip(".") in BOTANICAL_AUTHORITIES
        ]
    assert bad == []


def _labels(html):
    """Language labels the parser reads from one section, after its markup repairs."""
    html = repair_botanical_markup(html, abbrevs, latin_genera)
    _, spans = split_language_spans(mark_plain_language_labels(html, abbrevs), abbrevs)
    return [label for label, _ in spans], spans


def _lemmas(span_html):
    """(form markup, gloss) pairs as parse.py's lemma scan reads them."""
    span_html = detach_inline_latin_glosses(span_html, abbrevs, latin_genera)
    return [(m.group(2).strip(), m.group(4).strip()) for m in _LEMMA.finditer(span_html)
            if m.group(2).strip()]


def test_language_label_set_inside_the_bold_run_starts_a_span():
    # 74: "<b>Ka. aḍalu</b>" filed Kannada aḍalu under Tamil as "Ka. aḍalu"
    labels, spans = _labels('<b><i>Ta.</i> aṭuppu</b> fear. <b>Ka. aḍalu</b> to be shaken.')
    assert labels == ['Ta', 'Ka']
    assert spans[1][1].startswith(' aḍalu</b>')
    # 1562: the label alone in bold, before its source
    labels, spans = _labels('<b><i>Ka.</i> siri</b> to smile. <b>Kuwi</b> (F.) <b>khṛikhali</b> to neigh.')
    assert labels == ['Ka', 'Kuwi']
    # 695: labels without a period
    assert _labels('<b><i>Go.</i> uṛp-</b> to scrape. <b>Konḍa ūṛs- (-t-)</b> to sweep.')[0] == [
        'Go', 'Konḍa']


def test_cross_reference_in_bold_is_not_a_language_span():
    # 1225 "Or with 1221 Ta. kappu", 856 "Cf. 776 Ka. Te. eggu", 3753 "(cf. s.v. 3753 Ta. nel)"
    assert _labels('<b><i>Malt.</i> kape</b> to touch. Or with 1221 <b>Ta. kappu</b> (Pfeiffer).')[0] == [
        'Malt']
    assert _labels('<b><i>Te.</i> ellidamu</b> contemptible. Cf. 776 Ka. Te. <b>eggu</b>.')[0] == ['Te']
    assert _labels('<b><i>Ko.</i> nel</b> paddy (cf. s.v. 3753 Ta. <b>nel</b>).')[0] == ['Ko']


def test_plain_language_label_before_a_bold_form_starts_a_span():
    # 55: Brahui hījing was filed under Kuwi; 153: Kuwi apka under Pengo
    assert _labels('<b><i>Kuwi</i> ajinai</b> to fear. Br. <b>hījing</b> to be scared.')[0] == [
        'Kuwi', 'Br']
    labels, spans = _labels('<b><i>Pe.</i> apka</b> only. Kuwi (Isr.) <b>apka</b> id.')
    assert labels == ['Pe', 'Kuwi']
    assert spans[1][1].lstrip(' </i>').startswith('(Isr.) <b>apka')
    # the authority of "C. gigantea R. Br." is not Brahui; "R. Br. Ma." still starts Malayalam
    assert _labels('<b><i>Ta.</i> erukku</b> <i><b>C. gigantea</b></i> R. Br. <b><i>Tu.</i> ekka</b>')[0] == [
        'Ta', 'Tu']
    assert _labels('<b><i>Ka.</i> x</b> <i>S. nobilis</i> R. Br. <b>Ma. kāvaḷam</b> id.')[0] == [
        'Ka', 'Ma']


def test_label_source_parenthesis_may_close_after_the_italics():
    # 215 <i><b>Bel. (LSB</b></i> 2.2) argi; 1199 <b><i>Kurub. (LSB</i> 1.11) kadvāya</b>
    labels, spans = _labels('<b><i>Tu.</i> ari</b> rice. <i><b>Bel. (LSB</b></i> 2.2) <b>argi</b> rice.')
    assert labels == ['Tu', 'Bel']
    assert spans[1][1].startswith('(LSB 2.2)  <b>argi</b>')
    labels, spans = _labels('<b><i>Ta.</i> katuppu</b> cheek. <b><i>Kurub. (LSB</i> 1.11) kadvāya</b> cheek.')
    assert labels == ['Ta', 'Kurub']
    assert spans[1][1].startswith('(LSB 1.11)  kadvāya</b>')
    assert 'Kurub' not in spans[0][1]


def test_label_punctuation_and_spelling_slips_still_name_the_language():
    for html, label in [('<b><i>Ko..</i> nunk-</b> to cut', 'Ko'), ('<b><i>Malt;</i> boṉg̣e</b> to run', 'Malt'),
                        ('<b><i>Koḍ:</i> baṭṭi</b> basket', 'Koḍ'), ('<b><i>Ta:</i> toy</b> to breathe', 'Ta'),
                        ('<b><i>Manḍa</i> hab-</b> to decay', 'Manḍa'), ('<b><i>Mand.</i> aŋlā-</b> id.', 'Mand')]:
        assert _labels(html)[0] == [label]
    assert {abbrevs[label] for label in ('Mand', 'Manḍa')} == {'Manda'}
    assert {abbrevs[label] for label in ('Koṇḍa', 'Konḏa')} == {'Konda'}
    assert (abbrevs['Ālku'], abbrevs['Ṭe'], abbrevs['Koḏ'], abbrevs['Koḷ']) == (
        'AluKurumba', 'Telugu', 'Kodagu', 'Kolami')


def test_coordinated_labels_share_one_form_list():
    labels, spans = _labels('<b><i>Ir.</i> uli</b> waterfall. <b><i>PālKu., ĀlKu.</i> oli</b> id.')
    assert labels == ['Ir', 'PālKu', 'ĀlKu']
    assert spans[1][1] == spans[2][1]


def test_latin_name_before_a_doubtful_or_sourced_label():
    # 1081 "C. fistula. ? Koḍ."; 2474 "A. Farnesiana. Te. (VPK)"; 1845 "O. punctatus" is not Oriya
    assert repair_botanical_markup('<b><i>C. fistula. ? Koḍ.</i> kë·kï</b>', abbrevs) == (
        '<i>C. fistula</i>. ? <b><i>Koḍ.</i> kë·kï</b>')
    labels, spans = _labels('<b><i>Ka.</i> jāli</b> tree, <b><i>A. Farnesiana. Te. (VPK)</i> jāli</b> id.')
    assert labels == ['Ka', 'Te']
    assert '<i>A. Farnesiana</i>. ' in spans[0][1]
    assert spans[1][1].startswith('(VPK) ')
    assert _labels('<b><i>Ta.</i> kuṟavai</b> murrel, '
                   '<b><i>Ophiocephalus striatus; O. punctatus. Te.</i> koṟṟa</b> fish')[0] == ['Ta', 'Te']


def test_short_genus_names_are_glosses_not_forms():
    assert repair_botanical_markup('galangal, <b><i>Alpinia. Ma.</i> aratta</b>', abbrevs, latin_genera) == (
        'galangal, <i>Alpinia</i>. <b><i>Ma.</i> aratta</b>')
    assert repair_botanical_markup('cassia, <i><b>Cassia</b></i> fistula', abbrevs, latin_genera) == (
        'cassia, <i>Cassia</i> fistula')
    # Voc. (the Gondi vocabulary), titles and labels keep their markup
    for html in ('(<i><b>Voc.</b></i> 281)', '<b><i>Mand.</i> aŋlā-</b>', '<i><b>Gramm.</b></i>'):
        assert repair_botanical_markup(html, abbrevs, latin_genera) == html


def test_citation_xref_is_not_a_form_but_cdial_stays_a_label():
    html = 'Kuiper, <i><b><xref="iij">IIJ</xref="iij"></b></i> 2.239)'
    assert repair_botanical_markup(html, abbrevs) == 'Kuiper, <i><xref="iij">IIJ</xref="iij"></i> 2.239)'
    cdial = 'Turner, <i><b><xref="cdial">CDIAL</xref="cdial"></b></i>, no. 1347'
    assert repair_botanical_markup(cdial, abbrevs) == cdial


def test_italic_latin_gloss_inside_the_bold_run_ends_the_lemma():
    # 210: "alari Nerium odorum" and "Plumeria acuminata" were forms
    assert _lemmas(' araḷi, alari <i>Nerium odorum; Plumeria acuminata</i></b>. DED 173.') == [
        ('araḷi, alari', '<i>Nerium odorum; Plumeria acuminata</i>. DED 173.')]
    # 560: a name between two forms of one run
    assert _lemmas(' uka-maram <i>(D. speciosa), Careya arborea</i>; malay-uka '
                   '<i>Aleurites moluccana</i></b> Rh.') == [
        ('uka-maram', '<i>(D. speciosa), Careya arborea</i>;'),
        ('malay-uka <i>Aleurites moluccana</i>', 'Rh.'),
    ]
    # not a source title (588), a grammar label, or text that is not a following form (62)
    for span in (' (Keikádi dialect; Hislop, <i>Papers relating to the Aboriginal Tribes of the Central '
                 'Provinces</i>, Part II, p. 19) <b>udku</b> hot.',
                 '<b>akkakka, akkoḷu <i>n.</i></b> a sister',
                 ' aṭakkā-maṇiyan <i>Sphaeranthus indicus</i> (maṇiyan</b> a kind of paddy).'):
        assert detach_inline_latin_glosses(span, abbrevs, latin_genera) == span


CAPITALISED_DEDR_FORMS = {
    # Telugu personal names cited as reflexes of eṟṟa 'red' (865), not misparses
    ('d865', 'Telugu', 'Eṟṟana'), ('d865', 'Telugu', 'Eṟama'),
    ('d865', 'Telugu', 'Errayya'), ('d865', 'Telugu', 'Erramma'),
}


def test_capitalised_dedr_forms_are_only_documented_proper_names():
    with (ROOT / "data/dedr/dedr_new.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert {(row[1], row[0], row[2]) for row in rows if row[2][:1].isupper()} == CAPITALISED_DEDR_FORMS
    entry = lambda number: [row for row in rows if row[1] == number]
    # gloss words set in bold stay in the gloss (2037 country names, 4198 caste name)
    assert {row[2]: row[3] for row in entry('d2037')}['koṅkaṉ'] == 'inhabitant of the Koṅku country'
    assert 'title of Veḷḷāḷa caste' in {row[2]: row[3] for row in entry('d4198')}['piḷḷai']
    # misfiled forms are under their own languages (74, 2531, 3512 lost its Tamil rows)
    assert {(row[0], row[2]) for row in entry('d74')} >= {('Kannada', 'aḍalu'), ('Telugu', 'aḍalu')}
    assert ('salur', 'cinup ( cinpul)') in {(row[0], row[2]) for row in entry('d2531')}
    assert [row[2] for row in entry('d3512') if row[0] == 'Tam'] == ['toy (-v-, -nt-)', 'toyvu']
    # the LSB source closes after the label's italics and still maps to its bibliography key
    ebbi = next(row for row in entry('d226') if row[2] == 'ebbi')
    assert (ebbi[0], ebbi[6], ebbi[7]) == ('BettaKurumba', 'LSB 1.11', 'dedr;LSB-BettaKurumba')
    # text dropped by DSAL and restored from the 2013 SQL dump
    assert ('Mal', 'cuṇṭa') in {(row[0], row[2]) for row in entry('d2665')}
    assert ('Kota', 'am gakadg') in {(row[0], row[2]) for row in entry('d1077')}


def test_compiled_graph_has_no_capitalised_dedr_misparses():
    forms_path = ROOT / "cldf/forms.csv"
    if not forms_path.exists():
        return
    allowed = {form for _, _, form in CAPITALISED_DEDR_FORMS}
    with forms_path.open(encoding="utf-8", newline="") as stream:
        bad = [
            (row["ID"], row["Language_ID"], row["Original"])
            for row in csv.DictReader(stream)
            if row["Source"].split(";")[0] == "dedr"
            and row["Original"][:1].isupper() and row["Original"] not in allowed
        ]
    assert bad == []
