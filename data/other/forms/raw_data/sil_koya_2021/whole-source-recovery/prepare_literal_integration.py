"""Assemble review-only whole-source outputs, preserving the failed first freeze."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys
import unicodedata as ud

HERE = Path(__file__).resolve().parent
OUT = HERE / 'literal-integration'
sys.path.insert(0, str(HERE.parent))
import import_koya as legacy


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def residual_note(record):
    if not record['status'].startswith('uncertain_'):
        return ''
    subject = record['status'].removeprefix('uncertain_').replace('_', ' ')
    note = f'Source handwriting/notation uncertainty: {subject}.'
    narrow = record.get('narrow_adjudication') or {}
    alternatives = narrow.get('alternatives', [])
    if alternatives:
        note += ' Alternative literal reading(s): ' + ' / '.join(alternatives) + '.'
    if record.get('question_shaped_sign_review'):
        note += ' The question-shaped sign is retained; phonetic notation versus collector doubt is unresolved.'
    if 'ᵒ' in record['literal_source_proposal']:
        note += ' Raised round glyph represented provisionally by ᵒ; vowel or release interpretation is not established.'
    return note


def main():
    OUT.mkdir(exist_ok=True)
    records = [json.loads(line) for line in (HERE / 'notation-keyed-proposal-20260927.jsonl').read_text().splitlines()]
    by_key = {r['source_unit_key']: r for r in records}
    old_forms, old_audit, _ = legacy.build()
    old_by_key = {r[10]: r for r in old_forms}
    prompts = legacy.read_prompts()
    rows, audit, aliases = [], [], []
    source_keys = set()
    for old in old_audit:
        code, item = old['Site_Code'], int(old['Item'])
        key = f'wordlist:{code}:{item}'
        unit = {'source_unit_key': key, 'legacy_audit': old, 'entry_keys': [], 'status': old['Status']}
        if key not in by_key:
            unit['status'] = 'comparison_control' if old['Role'] != 'target' else 'source_omitted_slot'
            audit.append(unit)
            continue
        record = by_key[key]
        source_keys.add(key)
        unit['notation_review'] = record
        literal = record['literal_source_proposal']
        variants = legacy.split_variants(literal)
        old_keys = old['Entry_Keys'].split('|') if old['Entry_Keys'] else []
        unit['previous_entry_keys'] = old_keys
        unit['status'] = 'proposed' if literal else ('source_absent_slot' if key == 'wordlist:MAL:200' else 'source_blank')
        new_keys = [f'{record["entry_key_prefix"]}:v{i}' for i in range(1, len(variants) + 1)]
        for i, (form, entry_key) in enumerate(zip(variants, new_keys)):
            note = residual_note(record)
            if key == 'wordlist:POD:191':
                note = (note + ' Source identifies the parenthesized component kisu as “fire”.').strip()
            if key == 'wordlist:POD:186':
                note = (note + ' Canceled middle fragment is excluded from the visible replacement; full corrected-ink evidence is retained in the source audit.').strip()
            tags = legacy.dialect_tag(code)
            if record['status'].startswith('uncertain_'):
                tags += ' uncertain'
            row = ['Gondi', '', form, prompts[item], '', '', note, old['Source'], '', '', entry_key,
                   new_keys[0] if i else '', '', '', tags]
            rows.append(row)
        unit['entry_keys'] = new_keys
        if len(old_keys) > len(new_keys):
            assert key == 'wordlist:UTN:198' and len(old_keys) == 2 and len(new_keys) == 1
            aliases.append({'alias_entry_key': old_keys[1], 'canonical_entry_key': new_keys[0],
                            'source_unit_key': key, 'reason': 'Old erroneous slash split; original has one continuous answer va:val.',
                            'status': 'proposed_alias_pending_independent_review'})
        audit.append(unit)
    assert source_keys == set(by_key)
    assert len(audit) == 1890
    keyset = {r[10] for r in rows}
    assert len(keyset) == len(rows)
    alias_keys = {a['alias_entry_key'] for a in aliases}
    assert set(old_by_key) == keyset | alias_keys
    assert all(a['canonical_entry_key'] in keyset for a in aliases)
    assert all(not r[11] or r[11] in keyset for r in rows)
    with (OUT / 'proposed-wordlist.csv').open('w', newline='') as handle:
        csv.writer(handle).writerows(rows)
    (OUT / 'proposed-wordlist-audit.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in audit))
    (OUT / 'proposed-entry-key-aliases.json').write_text(json.dumps(aliases, ensure_ascii=False, indent=2) + '\n')
    # Explicit preservation, not inferred IPA; canonical Phonemic remains blank.
    characters = set(''.join(r[2] for r in rows))
    characters |= set(ud.normalize('NFD', ''.join(characters)))
    lines = ['Grapheme\tIPA']
    lines += [f'{c}\t{c}' for c in sorted(characters) if c != ' ']
    (OUT / 'proposed-sil-koya-literal.txt').write_text('\n'.join(lines) + '\n')
    (OUT / 'proposed-wordlist.yaml').write_text('''file: 20260828-sil-koya.csv
defaults:
  identity:
    dedupe_by_entry_key: true
  transcription:
    profile: sil-koya-literal
    preserve_hyphens: true
sources:
  devagnanavaram-et-al2021koya:
    identity:
      dedupe_by_entry_key: true
    forms:
      split_alternates: false
    reference:
      editor: OpenAI Codex
      ocr: true
      etymology_provenance: none
''')
    copied = []
    for kind in ['narrative', 'comparative']:
        for name in [f'proposed-{kind}.csv', f'proposed-{kind}.yaml', f'proposed-sil-koya-{kind}.txt']:
            shutil.copyfile(HERE / name, OUT / name)
            assert (HERE / name).read_bytes() == (OUT / name).read_bytes()
            copied.append(name)
    additions = [json.loads(line) for line in (HERE / 'proposed-additions-audit.jsonl').read_text().splitlines()]
    with (OUT / 'proposed-whole-audit.jsonl').open('w') as handle:
        for unit in audit + additions:
            handle.write(json.dumps(unit, ensure_ascii=False) + '\n')
    counts = {kind: sum(1 for _ in csv.reader((OUT / f'proposed-{kind}.csv').open())) for kind in ['wordlist', 'narrative', 'comparative']}
    bibliography = json.loads((HERE / 'bibliography-update-proposal.json').read_text())
    bibliography['fields']['included'] = bibliography['fields']['included'].replace('1,438', '1,437').replace('1,666', '1,665')
    bibliography['fields']['note'] = bibliography['fields']['note'].replace('Wordlists retain their separate phonetic profile.', 'Wordlists preserve literal single-dot versus colon length, raised vowels and consonant marks in a separate source-notation profile; Phonemic remains blank. Unresolved handwritten glyphs and question-shaped signs retain explicit interpretation uncertainty.').replace('Existing wordlist uncertainty remains in audit and tags', 'Wordlist uncertainty remains in Notes, audit and tags')
    bibliography['fields']['provenance'] += '. Complete wordlist rereading and source-key reconciliation evidence: data/other/forms/raw_data/sil_koya_2021/whole-source-recovery/literal-integration; one historical erroneous alternative key is proposed as an alias of the same source attestation'
    (OUT / 'bibliography-update-proposal.json').write_text(json.dumps(bibliography, ensure_ascii=False, indent=2) + '\n')
    report = {
        'status': 'review_proposal_not_frozen_or_installed', 'rows': counts, 'total_rows': sum(counts.values()),
        'whole_audit_units': len(audit) + len(additions), 'wordlist_statuses': dict(Counter(u['status'] for u in audit)),
        'legacy_wordlist_keys': len(old_by_key), 'preserved_active_keys': len(keyset), 'proposed_key_aliases': aliases,
        'copied_historical_addition_files_unchanged': copied,
        'normalization': 'Wordlist uses explicit literal preservation profile; Phonemic blank. No interpretation of uncertain glyphs as IPA.',
        'historical_failures_preserved': True, 'canonical_modified': False,
        'sha256': {p.name: digest(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'proposal-summary.json'},
        'input_sha256': {name: digest(HERE / name) for name in ['notation-keyed-proposal-20260927.jsonl', 'notation-narrow-adjudications-20260927.json', 'notation-iota-encoding-review-20260927.json', 'proposed-additions-audit.jsonl', 'whole-output-freeze-20260927.json', 'independent-whole-output-audit-20260927.json']},
    }
    (OUT / 'proposal-summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in {'sha256', 'input_sha256'}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
