"""Source-specific release profile decisions; no source installation or DB writes."""
from pathlib import Path
from segments import Tokenizer
import profile_policy

ROOT = Path(__file__).resolve().parents[1]


def converted(profile, value):
    return Tokenizer(str(ROOT / 'conversion' / (profile + '.txt')))(value, column='IPA').replace(' ', '').replace('#', ' ')


def test_cfel_ipa_to_house_keeps_munda_vowel_quality():
    for name in ['cfel-koda-api','cfel-koda-print']:
        assert name in profile_policy.SCHWA
        assert converted(name, 'əɔæ') == 'əɔæ'
        assert converted(name, 'jwɟʒʱʊ') == 'yvjźʰu'
        assert converted(name, 'কোড়া') == 'কোড়া'
    assert converted('cfel-koda-api','ǝɪ') == 'əi'


def test_historical_nasal_house_notation():
    assert converted('grierson-koda-birbhum-1906','ṃ') == 'ṁ'
    assert converted('grierson-sansi-1922','ṅ') == 'ŋ'


def test_hahn_song_colon_is_punctuation_not_length():
    assert ':' not in profile_policy.length_marks('hahn-asur-1900')
    assert 'ː' in profile_policy.length_marks('hahn-asur-1900')
    assert converted('hahn-asur-1900','roŋōlenā: roŋōlena:') == 'roŋōlenā roŋōlena'
