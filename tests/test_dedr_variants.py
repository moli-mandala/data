from dedr_variants import expand_attached_sound_variants


def test_expands_attached_optional_sound():
    assert expand_attached_sound_variants("mur̤(u)ku") == ["mur̤ku", "mur̤uku"]


def test_expands_word_final_optional_sound():
    assert expand_attached_sound_variants("aŋgoṭe(y)") == ["aŋgoṭe", "aŋgoṭey"]


def test_expands_multiple_optional_sounds():
    assert expand_attached_sound_variants("a(y)i(n)") == [
        "ai",
        "ain",
        "ayi",
        "ayin",
    ]


def test_leaves_space_separated_morphology_untouched():
    form = "muŋŋ- (muŋŋi-)"
    assert expand_attached_sound_variants(form) == [form]


def test_leaves_parenthetical_source_labels_untouched():
    form = "222. DED(S) 121"
    assert expand_attached_sound_variants(form) == [form]


def test_leaves_leading_dialect_labels_untouched():
    form = "(F.) aṇḍatasi"
    assert expand_attached_sound_variants(form) == [form]
