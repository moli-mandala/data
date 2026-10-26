"""Exact source categories, never triggers for natural-language prose."""
from pathlib import Path
from tags import GRAMMATICAL_TAGS, extract_tags


def test_permissive_and_injunctive_canonical_registration():
    assert {'permissive','injunctive','concessive','polite','opt','hortative','impv'} <= GRAMMATICAL_TAGS
    assert extract_tags('suffix permissive',attestations=False)==('suffix permissive','')
    assert extract_tags('verb injunctive',attestations=False)==('verb injunctive','')
    assert extract_tags('suffix concessive',attestations=False)==('suffix concessive','')
    assert extract_tags('verb impv polite',attestations=False)==('verb impv polite','')
    assert extract_tags('verb opt hortative impv',attestations=False)==('verb opt hortative impv','')


def test_no_natural_prose_alias_or_unrelated_mood_extraction():
    for note in ('The permissive construction permits an action.',
                 'An injunctive is a suggestion rather than a command.',
                 'This is a concessive clause.', 'A polite suggestion is printed.',
                 'permissive use of the old imperative', 'injunctive-looking'):
        assert extract_tags(note,attestations=False)==('',note)


def test_frontend_canonical_category_and_label_parity():
    root=Path(__file__).resolve().parents[6]
    text=(root/'jambu-static/src/lib/tags.ts').read_text()
    grammar=text.split('export const GRAMMATICAL_TAGS = [',1)[1].split('];',1)[0]
    assert "'permissive'" in grammar and "'injunctive'" in grammar
    assert "'concessive'" in grammar and "'polite'" in grammar
    assert "permissive: 'permissive'" in text and "injunctive: 'injunctive (suggestion)'" in text
    assert "concessive: 'concessive'" in text and "polite: 'polite'" in text
