"""Prepare the complete recovered additions and transcription drafts; never install."""
import csv
import io
import json
from pathlib import Path

from import_koya import SITES, SOURCE_KEY, dialect_tag
from prepare_narrative_analysis import analyze

HERE = Path(__file__).resolve().parent / 'whole-source-recovery'
LECTS = {info[0]: code for code, info in SITES.items() if info[1] == 'target'}
LECTS.update({name.removesuffix(' Koya'): code for name, code in list(LECTS.items())})


def row(form, gloss, notes, key, lect, locator, extra_tags=''):
    return ['Gondi', '', form, gloss, '', '', notes,
            f'{SOURCE_KEY}[{locator}]', '', '', key, '', '', '',
            ' '.join(filter(None, [dialect_tag(LECTS[lect]), extra_tags]))]


def prepare():
    narrative = []
    audit = []
    units = analyze()
    scope_report = json.loads((HERE / 'narrative-scope-review-20260927.json').read_text())
    scope = {d['source_unit_key']: d for d in scope_report['decisions']}
    assert len(scope) == len(units) == 205
    assert set(scope) == {u['source_unit_key'] for u in units}
    for unit in units:
        decision = scope[unit['source_unit_key']]
        assert (decision['original'], decision['source_translation']) == (unit['original_linguistic_text'], unit['gloss'])
        extra_tags = decision['scope_tag'] + (' interr' if unit['kind'] == 'test-question' else '')
        page = unit['target_lines'][0]['printed_page']
        narrative.append(row(unit['original_linguistic_text'], unit['gloss'], unit['notes'],
                             unit['source_unit_key'], unit['lect'],
                             f"Appendix {unit['section']}, p. {page}, text unit {unit['source_unit_key'].rsplit(':', 1)[1]}",
                             extra_tags))
        audit.append(dict(unit, proposed_entry_keys=[unit['source_unit_key']], scope_review=decision))
    for unit in map(json.loads, (HERE / 'supplemental-lexical-first-reading.jsonl').read_text().splitlines()):
        if unit['form'] == 'Vija':
            audit.append(dict(unit, proposed_entry_keys=[], disposition='documentary festival proper name; no lexical definition supplied'))
            continue
        notes = '' if unit['gloss'] else f"Source translation: {unit['source_quote']} The source supplies no separate lexical definition."
        narrative.append(row(unit['form'], unit['gloss'], notes, unit['source_unit_key'], unit['lect'],
                             f"Appendix B, p. {unit['printed_page']}, named term in text"))
        audit.append(dict(unit, proposed_entry_keys=[unit['source_unit_key']], disposition='explicit term shared by target text and translation'))
    comparative = []
    for unit in map(json.loads, (HERE / 'comparative-first-reading.jsonl').read_text().splitlines()):
        comparative.append(row(unit['source_form'], unit['gloss'],
                               'Source compares dialect sound correspondences; no etymological relationship is asserted.',
                               unit['source_unit_key'], unit['lect'], f"section 3.1.4, p. {unit['printed_page']}, {unit['gloss']}",
                               'num' if unit['gloss'] == 'six' else 'multiword-expression' if ' ' in unit['source_form'] else ''))
        audit.append(dict(unit, proposed_entry_keys=[unit['source_unit_key']]))
    assert len(narrative) == 210 and len(comparative) == 18
    assert len({r[10] for r in narrative + comparative}) == 228
    return narrative, comparative, audit


def profile(rows, comparative=False):
    symbols = sorted(set(''.join(r[2] for r in rows)) - {' '})
    lines = ['Grapheme\tIPA']
    if not comparative:
        lines.append('[stutter]\tNULL')
    lines += [f'{symbol}\t{chr(0x0263) if comparative and symbol == "G" else symbol}' for symbol in symbols]
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    narrative, comparative, audit = prepare()
    for name, rows in [('narrative', narrative), ('comparative', comparative)]:
        with (HERE / f'proposed-{name}.csv').open('w', newline='') as handle:
            csv.writer(handle).writerows(rows)
        (HERE / f'proposed-sil-koya-{name}.txt').write_text(profile(rows, name == 'comparative'))
    (HERE / 'proposed-additions-audit.jsonl').write_text(''.join(json.dumps(a, ensure_ascii=False) + '\n' for a in audit))
    report = {
        'status': 'complete recovered-addition draft; not installed or independently accepted',
        'narrative_rows': len(narrative), 'comparative_rows': len(comparative),
        'audit_units': len(audit), 'documentary_festival_names': 1,
        'existing_wordlist_rows': 1438, 'proposed_total_source_rows': 1666,
        'transcription': 'Narrative graphemes preserved, including capitals and unexplained punctuation; editorial stutter retained in Original and omitted from display. Comparative G maps to voiced velar fricative per explicit source statement. No invented Phonemic.',
        'installation_design': 'Retain old wordlist CSV and add two source CSVs with the same bibliography key and file-default transcription profiles. Move old source-global profile to its file defaults. Disable alternate splitting for the shared key.',
        'pending': ['Public residual Notes must survive existing audit-only source policy', 'Corpus profile validation', 'Registered parser validation', 'Focused tests', 'Independent output audit', 'Metadata and guarded installation'],
    }
    (HERE / 'proposed-additions-summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))
