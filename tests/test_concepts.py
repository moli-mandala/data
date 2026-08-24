from concepts import _legacy_senses, _split_top_level, map_glosses, sense_candidates


def test_sense_candidates_preserve_context_and_pos():
    assert ("kill (v.)", "verb") in sense_candidates("kill (v.)")
    assert ("bank (river edge)", "") in sense_candidates("bank (river edge)")


def test_sense_candidates_split_enumerations_but_not_slashes():
    assert sense_candidates("hair, body hair") == [
        ("hair, body hair", ""),
        ("hair", ""),
        ("body hair", ""),
    ]
    assert sense_candidates("arm/hand") == [("arm/hand", "")]


def test_split_top_level_preserves_punctuation_inside_brackets():
    assert _split_top_level("to pay (fine, debt); to repay") == [
        "to pay (fine, debt)",
        "to repay",
    ]


def test_legacy_candidates_remain_available_for_compatibility():
    assert _legacy_senses("kill (v.); murder") == ["kill", "murder"]


def test_map_glosses_keeps_multiple_enumerated_concepts():
    concepts = {cid for cid, _label, _pos in map_glosses(["hair, body hair"])["hair, body hair"]}
    assert {"1040", "189"} <= concepts


def test_map_glosses_keeps_combined_slash_concept():
    concepts = {cid for cid, _label, _pos in map_glosses(["arm/hand"])["arm/hand"]}
    assert "2121" in concepts
