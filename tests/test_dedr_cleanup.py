from data.dedr.cleanup import footer_note, is_footer_misparse


def test_detects_bold_footer_reference_misparse():
    assert is_footer_misparse("DED(S)")
    assert is_footer_misparse("pp. 251-2. DED(S) 282")
    assert is_footer_misparse("DEN DBIA SI")
    assert is_footer_misparse("pïf Indigofera pulchella. DEDS 687")


def test_detects_footer_with_spacing_variants():
    assert is_footer_misparse("DED (S, N) 1193")
    assert is_footer_misparse("s.v. DED(S. N) 4438")


def test_preserves_normal_forms_and_optional_sounds():
    assert not is_footer_misparse("mur̤(u)ku")
    assert not is_footer_misparse("dedu")
    assert not is_footer_misparse("dādi den-me")


def test_footer_form_and_gloss_are_preserved_as_one_note():
    assert footer_note("<i>DEDS</i>", "687.") == "<i>DEDS</i> 687."
    assert footer_note(". DED 84.", "\t<div>\xa0</div>") == "DED 84."
