from form_grammar import annotation_tags, extract_gloss_tags


def parsed(gloss: str, filename: str = "20260813-dewas-rai.csv", source: str = "x"):
    return extract_gloss_tags(
        gloss,
        input_file=filename,
        source_key=source,
        full_input_path=f"data/other/forms/{filename}",
    )


def test_survey_person_number_tense_and_polarity_labels():
    assert parsed("go (3S-PT)") == ("go", ("3sg", "pret"))
    assert parsed("go (2S-neg)") == ("go", ("2sg", "neg"))
    assert parsed("bring down (3sg past)") == ("bring down", ("3sg", "pret"))
    assert parsed("eat (past, 3rd, inf.)") == ("eat (past, 3rd, inf.)", ())


def test_pos_gender_valency_and_register_labels():
    assert parsed("spit (noun)", "20230306-wadiyara.csv") == ("spit", ("noun",))
    assert parsed("seat (tr.)", "20230705-pashai.csv") == ("seat", ("verb", "tr"))
    assert parsed("he (3rd sg, masculine)", "20230526-kannauji.csv") == (
        "he", ("3sg", "m")
    )
    assert parsed("you (2nd sg, formal)", "20230526-kannauji.csv") == (
        "you", ("2sg", "formal")
    )
    assert parsed("eat (honorific)", "20260813-mustang-loke.csv") == (
        "eat", ("honorific",)
    )


def test_inclusive_exclusive_dual_and_near_future():
    assert parsed("we (1st pl, inclusive)", "20230521-rajasthani.csv") == (
        "we", ("1pl", "inclusive")
    )
    assert parsed("we (dual/exclusive)", "20260813-yamphu.csv") == (
        "we", ("du", "exclusive")
    )
    assert parsed("tomorrow (near future)", "20260813-hajong.csv") == (
        "tomorrow", ("fut", "near-future")
    )
    assert parsed("you plural", "20260813-eastern-magar.csv") == ("you", ("pl",))
    assert parsed("I beat (Past Tense)", "20260813-grierson-lsi.csv") == (
        "I beat", ("pret",)
    )


def test_source_table_verb_labels_omitted_by_legacy_snapshots():
    assert parsed("drink", "20230524-sindhic.csv", "maimani") == ("drink", ("verb",))
    assert parsed("burn", "20220913-zadjali.csv", "zadjali") == ("burn", ("verb",))
    assert parsed("to drink", "20220913-zadjali.csv", "zadjali") == (
        "to drink", ("verb",)
    )
    assert parsed("water", "20230524-sindhic.csv", "maimani") == ("water", ())


def test_false_positives_remain_lexical_prose():
    assert parsed("bark (of a tree)", "20230524-sindhic.csv", "maimani") == (
        "bark (of a tree)", ()
    )
    assert parsed("area (used with forest, road, etc.)", "20260813-wolf-kota.csv") == (
        "area (used with forest, road, etc.)", ()
    )
    assert annotation_tags("cf. Skt. v. 103") == ()
    assert extract_gloss_tags(
        "drink (V)", input_file="unreviewed.csv", source_key="x"
    ) == ("drink (V)", ())


def test_munda_filename_is_disambiguated_from_other_forms_csv_files():
    assert extract_gloss_tags(
        "weave (v)",
        input_file="forms.csv",
        source_key="rau",
        full_input_path="data/munda/forms.csv",
    ) == ("weave", ("verb",))
    assert extract_gloss_tags(
        "weave (v)",
        input_file="forms.csv",
        source_key="unrelated",
        full_input_path="data/dbia/forms.csv",
    ) == ("weave (v)", ())


def test_spelled_out_person_number_labels():
    assert parsed("I (1st singular)", "20230521-rajasthani.csv") == (
        "I", ("1sg",)
    )
