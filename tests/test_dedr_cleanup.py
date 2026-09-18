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


def test_resolve_idem_expands_to_preceding_gloss():
    from data.dedr.parser_utils import resolve_idem

    def row(gloss):
        return ["Ta", "d1", "x", gloss] + [""] * 11

    rows = [row("elder sister (<i>Voc.</i> 1640)"), row("id. (<i>Voc.</i> 46)"),
            row("id., afterwards"), row("id"), row("cockroach. DEDS 7"), row("id. DED 4")]
    last = resolve_idem(rows)

    assert [r[3] for r in rows] == [
        "elder sister (<i>Voc.</i> 1640)", "elder sister (<i>Voc.</i> 46)",
        "elder sister, afterwards", "elder sister, afterwards", "cockroach. DEDS 7", "cockroach DED 4",
    ]
    assert last == "cockroach DED 4"


def test_resolve_idem_uses_previous_section_gloss():
    from data.dedr.parser_utils import resolve_idem

    rows = [["Ka", "d2", "y", "id."] + [""] * 11]
    resolve_idem(rows, previous="to dig")

    assert rows[0][3] == "to dig"


def test_resolve_idem_expands_ditto():
    from data.dedr.parser_utils import resolve_idem

    rows = [["H", "1", "x", "loin-cloth, waistband"] + [""] * 11, ["G", "1", "y", "small do."] + [""] * 11,
            ["G", "1", "z", "small do."] + [""] * 11]
    resolve_idem(rows)

    assert rows[1][3] == "small loin-cloth"
    assert rows[2][3] == "small loin-cloth"
