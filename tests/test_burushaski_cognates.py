from burushaski_cognates import (
    SourceForm,
    apply_catalog,
    attach_yoshioka_evidence,
    comparison_form,
    discover_berger_sets,
    discover_hkat_sets,
    load_catalog,
)


def form(language, word, gloss, key, parent="", source=""):
    return SourceForm(language, word, gloss, key, parent, "noun", source)


def test_comparison_normalizes_source_transcription_without_erasing_segments():
    assert comparison_form("tɕʰuːmo") == comparison_form("čhúmo")
    assert comparison_form("ariːŋ") == "arin"


def test_berger_requires_an_explicit_cross_dialect_link():
    base = form("Bur", "áḍa", "remaining", "berger-entry-1")
    yasin = form("Werch", "áḍe", "remaining", "berger-entry-1-y", base.key)
    unrelated = form("Werch", "bá", "remaining", "berger-entry-2")
    sets = discover_berger_sets([base, yasin, unrelated])
    assert len(sets) == 1
    assert sets[0]["Proto_Form"] == ""
    assert sets[0]["Evidence_Keys"] == "berger-entry-1|berger-entry-1-y"


def test_hkat_same_meaning_is_not_enough_for_cognacy():
    hunza = form(
        "HKAT-bsk_h", "wiːras", "die", "h-die", source="src[concept 17]"
    )
    nagar = form(
        "HKAT-bsk_n", "ajram", "die", "n-die", source="src[concept 17]"
    )
    assert discover_hkat_sets([hunza, nagar]) == []


def test_hkat_accepts_same_concept_with_phonological_evidence():
    hunza = form(
        "HKAT-bsk_h", "ariːn", "hand", "h-hand", source="src[concept 10]"
    )
    nagar = form(
        "HKAT-bsk_n", "ariːŋ", "hand", "n-hand", source="src[concept 10]"
    )
    sets = discover_hkat_sets([hunza, nagar])
    assert len(sets) == 1
    assert sets[0]["Proto_Form"] == ""
    assert sets[0]["Evidence_Keys"] == "h-hand|n-hand"


def test_eastern_burushaski_requires_both_form_and_gloss_and_excludes_loans():
    hunza = form("HKAT-bsk_h", "tɕʰumo", "fish (noun)", "h-fish", source="src[concept 2]")
    nagar = form("HKAT-bsk_n", "tɕʰumo", "fish (noun)", "n-fish", source="src[concept 2]")
    sets = discover_hkat_sets([hunza, nagar])
    eastern = form("Bur", "čhúmo", "fish", "y-fish")
    synonym = form("Bur", "čhúmo", "animal", "y-wrong-gloss")
    loan = SourceForm("Bur", "čhúmo", "fish", "y-loan", "", "noun loanword", "")
    attach_yoshioka_evidence(sets, [hunza, nagar], [eastern, synonym, loan])
    assert sets[0]["Evidence_Keys"] == "h-fish|n-fish|y-fish"


def test_catalog_makes_proto_entry_and_reflex_edges():
    left = ["l", "Bur", "áḍa", "remaining", "", "", "", "", "", "", "berger", "", "", "local", "", "", ""]
    right = ["r", "Werch", "áḍe", "remaining", "", "", "", "", "", "", "berger", "l", "", "variant", "", "l", ""]
    catalog = [{
        "Set_ID": "ada",
        "Proto_Form": "",
        "Gloss": "remaining",
        "Evidence_Keys": "left|right",
        "Method": "test",
        "Status": "accepted",
        "Notes": "form intentionally blank",
    }]
    proto, keys = apply_catalog([left, right], {"left": "l", "right": "r"}, catalog)
    assert proto[0][0:4] == ["pbsk-ada", "PBr", "", "remaining"]
    assert proto[0][6] == ""
    assert (left[11], left[13]) == ("pbsk-ada", "reflex")
    assert (right[11], right[13], right[15]) == ("pbsk-ada", "reflex", "")
    assert keys == [("pbsk-ada", "proto-burushaski:ada")]


def test_catalog_rejects_proto_burushaski_reconstruction():
    left = ["l", "Bur", "áḍa", "remaining", "", "", "", "", "", "", "berger", "", "", "local", "", "", ""]
    right = ["r", "Werch", "áḍe", "remaining", "", "", "", "", "", "", "berger", "l", "", "variant", "", "l", ""]
    catalog = [{
        "Set_ID": "ada", "Proto_Form": "*áḍa", "Gloss": "remaining",
        "Evidence_Keys": "left|right", "Method": "test", "Status": "accepted", "Notes": "",
    }]

    try:
        apply_catalog([left, right], {"left": "l", "right": "r"}, catalog)
    except ValueError as error:
        assert "Proto_Form must be blank" in str(error)
    else:
        raise AssertionError("a Proto-Burushaski reconstruction was accepted")


def test_checked_in_proto_burushaski_catalog_has_no_reconstructed_forms():
    catalog = load_catalog()
    assert len(catalog) == 706
    assert all(row["Proto_Form"] == "" for row in catalog)
