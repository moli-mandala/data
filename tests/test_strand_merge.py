from make_cldf import Row, merge_redundant_strand_rows
from data.other.forms.raw_data.strand import strand_pos_tags, strand_row as build_strand_row


def strand_row(input_file, language, parameter, form, gloss, ipa=""):
    row = Row(
        [language, parameter, form, gloss, "", ipa, "", "strand"],
        id=f"{input_file}:{language}:{parameter}:{form}",
    )
    row.input_file = input_file
    return row


def test_strand3_etymology_wins_for_an_exact_form_duplicate():
    old = strand_row("20220913-strand.csv", "Kam", "7621", "puk", "ripening", "puk")
    new = strand_row("20221003-strand3.csv", "Kam", "n88", "puk", "ripening")

    rows, merged = merge_redundant_strand_rows([old, new])

    assert merged == 1
    assert rows == [new]
    assert new.param == "n88"
    assert new.ipa == "puk"


def test_strand_homophones_merge_only_when_the_gloss_selects_one_etymon():
    old = strand_row("20220913-strand.csv", "nis", "11235", "maṭa", "agnate")
    lineage = strand_row(
        "20221003-strand3.csv", "nis", "n498", "maṭa", "share; division; agnatic lineage"
    )
    agnate = strand_row("20221003-strand3.csv", "nis", "n499", "maṭa", "agnate")

    rows, merged = merge_redundant_strand_rows([old, lineage, agnate])

    assert merged == 1
    assert rows == [lineage, agnate]
    assert agnate.param == "n499"


def test_ambiguous_strand_homophones_are_not_automatically_merged():
    old = strand_row("20220913-strand.csv", "Kam", "8656", "pr̆e", "hit; touch")
    give = strand_row("20221003-strand3.csv", "Kam", "n1443", "pr̆e", "give")
    give_away = strand_row("20221003-strand3.csv", "Kam", "n1529", "pr̆e", "gives away")

    rows, merged = merge_redundant_strand_rows([old, give, give_away])

    assert merged == 0
    assert rows == [old, give, give_away]


def test_strand_grammatical_codes_become_canonical_tags():
    assert strand_pos_tags("VT") == "verb tr"
    assert strand_pos_tags("(via Pashto) VI") == "verb intr"
    assert strand_pos_tags("NF") == "noun f"
    assert strand_pos_tags("NQt") == "num"
    assert strand_pos_tags("AjQt") == "adj num"
    assert strand_pos_tags("Pn?An") == "pron interr"


def test_strand_locations_become_dialect_tags():
    row = build_strand_row("Kam", "n1", "x", "test", "", "", "strand", "N", "Kmkt.km")

    assert row[14].split() == ["noun", "dialect:Kamviri"]
