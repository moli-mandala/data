"""Bounded proposal gates; no installation or whole-thesis completion claim."""
import csv
import io
import json
from pathlib import Path

P = Path(__file__).resolve().parents[1] / 'data/other/forms/raw_data/dhakal_darai_2011'

def test_existing_rows_and_keys_unchanged_and_exclusions_recovered():
    with (P.parents[1] / '20260921-dhakal-darai.csv').open() as stream:
        old = list(csv.reader(stream))
    with (P / 'clause-recovery-proposal.csv').open() as stream:
        new = list(csv.reader(stream))
    audit = [json.loads(line) for line in (P / 'clause-recovery-proposal-audit.jsonl').read_text().splitlines()]
    assert len(new) == 501 and new[:485] == old
    recovered = json.loads((P / 'clause-recovery-reviewed.json').read_text())
    assert {r[10] for r in new[485:]} == {r['key'] for r in recovered}
    assert len(audit) == 502
    assert sum(r['output'] is None for r in audit) == 1
    assert {r['key'] for r in audit if r['output']} == {r[10] for r in new}


def test_actual_parser_profile_and_grammar_scope():
    import make_cldf
    from tags import GRAMMATICAL_TAGS, GENDER_TAGS
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(P / 'clause-recovery-proposal.csv'), errors,
                                        name='20260921-dhakal-darai')
    assert len(parsed) == stats['converted'] == 501
    assert not errors.getvalue()
    added = parsed[485:]
    assert all('sentential' in r.tags.split() and r.notes and r.ipa for r in added)
    for row in added:
        assert {t for t in row.tags.split() if not t.startswith('dialect:')} <= GRAMMATICAL_TAGS | GENDER_TAGS
    by_key = {r.entry_key:r for r in added}
    # Possessor 1sg is not main-predicate first person.
    assert '1sg' not in by_key['dhakal2011:p73:g693'].tags.split()
    assert '1sg' not in by_key['dhakal2011:p75:g123'].tags.split()
    # The embedded causative is not asserted of the main predicate.
    assert 'caus' not in by_key['dhakal2011:p75:g755'].tags.split()
    assert 'DaF.CND.129' in by_key['dhakal2011:p75:g948'].source
