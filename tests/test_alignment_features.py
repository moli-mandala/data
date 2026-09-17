"""Regressions for distinctions used in prosodic correspondence analysis."""
from align import Seg, describe, score, segment_identity


def test_accent_is_not_a_segmental_vowel_change():
    for a, b in [('á', 'a'), ('ā́', 'ā'), ('ā̂', 'ā̌'), ('ŕ̩', 'r̩')]:
        assert describe(Seg(a), Seg(b)) == 'kept'
        assert score(Seg(a), Seg(b)) == score(Seg(b), Seg(b))
    assert segment_identity(Seg('ś')) == 'ś'
    assert describe(Seg('ś'), Seg('s')) != 'kept'


def test_nuclei_and_quantity():
    assert Seg('r̩').kind == 'V'
    assert Seg('ŕ̩').kind == 'V'
    assert Seg('r').kind == 'C'
    assert Seg('a̯').kind != 'V'
    assert Seg('ai').long and Seg('au').long
    assert describe(Seg('ā'), Seg('a')) == 'shortening'
    assert describe(Seg('ã'), Seg('a')) == 'denasalization'
