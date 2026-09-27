"""Stage Roy's complete Appendix I, without building the database or installing."""
import csv
import argparse
import shutil
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = 'roy1925birhors'

def nfc(s):
    return unicodedata.normalize('NFC', s)

def generate():
    inventory = [json.loads(s) for s in (ROOT / 'full-reviewed-inventory-staged.jsonl').read_text().splitlines()]
    # Vocabulary keys precede supplementary prose so legacy identities survive reuse.
    inventory.sort(key=lambda r: (r['source_section'] != 'vocabulary', r['printed_page'], r['entry_key']))
    rows, audit, identities = [], [], {}
    for item in inventory:
        assert item['pdf_page'] == item['printed_page'] + 88
        record = dict(item, entry_keys=[], reuse_entry_keys=[])
        if item['status'] == 'specific_damaged_print_hold':
            record['disposition'] = 'specific_damaged_print_hold'
            audit.append(record)
            continue
        units = [item] + [dict(item, **sub, source_comparison='', subentries=[]) for sub in item.get('subentries', [])]
        for unit in units:
            parent = ''
            for index, form in enumerate(re.split(r'[;,]', unit['printed_form'])):
                key = unit['entry_key'] + (f':answer{index + 1}' if index else '')
                form, gloss = nfc(form.strip()), nfc(unit['gloss'].strip())
                assert form and gloss
                # Only exact source spellings with equal lexical glosses are reused.
                identity = (form, gloss.casefold())
                locator = f"p. {unit['printed_page']}"
                if unit['source_section'] == 'vocabulary':
                    locator += f", {unit['column']} column, item {unit['column_item']}"
                else:
                    locator += ', introductory prose'
                citation = f'{SOURCE}[{locator}]'
                notes = []
                if unit.get('source_note'):
                    notes.append(unit['source_note'])
                comparison = unit.get('source_comparison', '')
                if comparison:
                    notes.append('Roy comparison: ' + comparison + ' (M. = Mundari, H. = Hindi, B. = Bengali; comparison alone does not assert borrowing or a cognate link).')
                if unit.get('typed_uncertainty') and not unit.get('source_note'):
                    notes.append('Unexplained literal source mark retained without phonemic interpretation.')
                if unit['entry_key'] == 'roy1925birhor:p584:L05' and index == 1:
                    notes.append(unit['transcription_uncertainty'])
                if unit['entry_key'] in {'roy1925birhor:p573:L18', 'roy1925birhor:p573:L19'}:
                    notes.append(unit.get('transcription_uncertainty', 'Literal source spacing retained.'))
                tags = list(unit.get('tags', []))
                if gloss.lower().startswith('to ') and gloss.lower() != 'to that place' and 'verb' not in tags:
                    tags.append('verb')
                if unit.get('typed_uncertainty') or (unit['entry_key'] == 'roy1925birhor:p584:L05' and index == 1):
                    tags.append('uncertain')
                if identity in identities:
                    old = identities[identity]
                    if citation not in old[7].split('; '):
                        old[7] += '; ' + citation
                    for note in notes:
                        if note not in old[6]:
                            old[6] += (' ' if old[6] else '') + note
                    old[14] = ' '.join(dict.fromkeys(old[14].split() + tags))
                    record['reuse_entry_keys'].append(old[10])
                    parent = parent or old[10]
                    continue
                row = ['Birhor', '', form, gloss, '', '', ' '.join(notes), citation, '', '', key, parent, '', '', ' '.join(dict.fromkeys(tags))]
                rows.append(row)
                identities[identity] = row
                record['entry_keys'].append(key)
                parent = parent or key
        record['disposition'] = 'accepted' if record['entry_keys'] else 'reused_attestation'
        audit.append(record)
    assert len(inventory) == len(audit) == 989
    assert len({r[10] for r in rows}) == len(rows)
    legacy = list(csv.reader((ROOT / 'legacy-pilot.csv').open()))
    assert {r[10] for r in legacy} <= {r[10] for r in rows}
    return rows, audit

def write(install=False):
    rows, audit = generate()
    with (ROOT / 'full-preview.csv').open('w', newline='') as handle:
        csv.writer(handle).writerows(rows)
    (ROOT / 'full-preview-audit.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in audit))
    # Source-literal display profile; no invented IPA values for ch/chh/apostrophes.
    chars = sorted(set(''.join(r[2] for r in rows)) - {' '})
    mapping = {c: c.lower() for c in chars}
    mapping.update({'w':'v', 'ṅ':'ŋ', 'ng':'ŋ'})
    (ROOT / 'full-profile-staged.txt').write_text('Grapheme\tIPA\n' + ''.join(f'{c}\t{v}\n' for c,v in sorted(mapping.items())))
    if install:
        shutil.copyfile(ROOT / 'full-preview.csv', ROOT.parents[1] / '20260925-roy-birhor-p567-p568.csv')
        shutil.copyfile(ROOT / 'full-profile-staged.txt', ROOT.parents[4] / 'conversion/roy-birhor.txt')
    print(f'{len(rows)} forms; {len(audit)} accounted head/prose units; installed={install}; no DB build')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    write(parser.parse_args().install)
