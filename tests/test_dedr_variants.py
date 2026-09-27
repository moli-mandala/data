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


def test_raised_length_dot_becomes_a_macron():
    from dedr_variants import normalize_dedr_marks
    from segments.tokenizer import Tokenizer
    import unicodedata

    # Burrow–Emeneau's Kota/Toda/Kodagu/Kolami length dot (printed as U+0387 or U+00B7)
    assert normalize_dedr_marks("a·k") == "āk"
    assert normalize_dedr_marks("a·(k) ka·ṛ") == "ā(k) kāṛ"
    assert normalize_dedr_marks("pu ·") == "pū"          # the website's stray space
    assert normalize_dedr_marks("lā·ru") == "lāru"       # already long: no double macron
    assert normalize_dedr_marks("ã·") == "ā̃"
    # Toda/Kota special vowels keep their own letter and take the house long form
    house = Tokenizer("conversion/dedr.txt")
    convert = lambda s: unicodedata.normalize("NFC", house(normalize_dedr_marks(s), column="IPA").replace(" ", ""))
    assert convert("ï·štyu·") == "ɨ̄śtyū"
    assert convert("ë·ḷ-") == "ə̄ḷ-"
    assert convert("nö·ṟ") == "nø̄ṟ"
    assert convert("nä·ṯṯu") == "nǣṯṯu"
