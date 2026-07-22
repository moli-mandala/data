from tamil_morphology import append_note, extract_tamil_verb_morphology


def test_extracts_clear_strong_class():
    result = extract_tamil_verb_morphology("ala (-pp-, -nt-)")

    assert result.citation_form == "ala"
    assert result.note == "(-pp-, -nt-)"
    assert result.tags == ("verb", "Tamil-class-7", "strong")
    assert result.review_reason == ""


def test_accepts_spacing_and_hyphen_variants():
    result = extract_tamil_verb_morphology("aṅkā (-pp- -tt-)")

    assert result.tags == ("verb", "Tamil-class-6", "strong")
    assert result.review_reason == ""


def test_keeps_multiple_clear_classes():
    result = extract_tamil_verb_morphology("akai (-v-, -nt-; -pp-, -tt-)")

    assert result.tags == (
        "verb",
        "Tamil-class-2",
        "weak",
        "Tamil-class-6",
        "strong",
    )


def test_classifies_expanded_class_3_paradigm():
    result = extract_tamil_verb_morphology("vāṅku (vāṅki-)")

    assert result.tags == ("verb", "Tamil-class-3", "weak")
    assert result.review_reason == ""


def test_classifies_expanded_class_4_paradigm():
    result = extract_tamil_verb_morphology("aṭu (aṭuv-, aṭṭ-)")

    assert result.tags == ("verb", "Tamil-class-4", "weak")
    assert result.review_reason == ""


def test_classifies_expanded_class_5_paradigm():
    result = extract_tamil_verb_morphology("kēḷ (kēṭp-, kēṭṭ-)")

    assert result.tags == ("verb", "Tamil-class-5", "middle")
    assert result.review_reason == ""


def test_flags_irregular_full_stem_paradigm_for_review():
    result = extract_tamil_verb_morphology("vā (varuv-, vant-)")

    assert result.tags == ("verb",)
    assert result.review_reason == "unclassified full-stem or irregular paradigm"


def test_ignores_non_morphological_parenthetical():
    assert extract_tamil_verb_morphology("akam (poetry)") is None


def test_appends_raw_morphology_to_existing_notes():
    assert append_note("dialectal", "(-v-, -nt-)") == "dialectal; (-v-, -nt-)"
