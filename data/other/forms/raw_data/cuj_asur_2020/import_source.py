"""Build audited CUJ Asur lexical inputs from the pinned dictionary snapshot."""
import argparse
import collections
import csv
import json
import re
import unicodedata as ud
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEM = '20260921-cuj-asur'
SOURCE = 'cuj2020asur'
POS = {'n': 'noun', 'v': 'verb', 'adj': 'adj', 'adv': 'adv',
       'sfx': 'suffix', 'pfx': 'prefix', 'pro': 'pron', 'pro-form': 'pron',
       'dem': 'demonstrative', 'quant': 'quantifier', 'interj': 'interj',
       'prt': 'part', 'part': 'part', 'post': 'postp', 'prep': 'prep',
       'det': 'determiner', 'num': 'num', 'aux': 'auxiliary'}
CODES = {
    'PST': ('past-tense marker', ['pret']), 'NEG': ('negation', ['neg']),
    'LOC': ('locative marker', ['loc']), 'PRS': ('present-tense marker', ['pres']),
    'REFL': ('reflexive', ['refl']), 'POSS': ('possessive marker', ['poss']),
    'COP': ('copula', ['copula']), 'ACC': ('accusative marker', ['acc']),
    'INS': ('instrumental marker', ['instr']), 'INCL': ('inclusive', ['inclusive']),
    '1SG': ('first person singular', ['1sg']), '2SG': ('second person singular', ['2sg']),
    '3SG': ('third person singular', ['3sg']), '3S': ('third person singular', ['3sg']),
    '2PL': ('second person plural', ['2pl']), '3PL': ('third person plural', ['3pl']),
    '=3PL': ('third person plural', ['3pl']),
    '3DL': ('third person dual', ['third-person', 'du']),
    '3S.GEN': ('third person singular genitive', ['3sg', 'gen']),
    '1SG.Inan': ('first person singular, inanimate', ['1sg', 'inanimate']),
    '3SG.Ani': ('third person singular, animate', ['3sg', 'animate']),
    '1DL.Inc': ('first person dual, inclusive', ['first-person', 'du', 'inclusive']),
    '1DL.Exc': ('first person dual, exclusive', ['first-person', 'du', 'exclusive']),
    '1PL.Exc': ('first person plural, exclusive', ['1pl', 'exclusive']),
}
# Visually reviewed Hindi-only embedded definitions, not inferred cognate glosses.
# The printed Devanagari is recorded because its PDF font mapping is defective.
INLINE_DEFINITIONS = {
    'cujasur2020:p44:c1:y197.9': {
        'ढेम्बा डन्डुर': ('काला रागी', 'black finger millet'),
        'पुड़ी डांडुर': ('बाजरा', 'pearl millet'),
    },
    'cujasur2020:p65:c2:y473.1': {
        'पोतना हास': ('दिवाल रंगने वाली मिट्टी', 'soil used to paint walls'),
    },
}


def nfc(text):
    return ud.normalize('NFC', text)


def build(candidates, resolution):
    resolved = {r['entry_key']: r for r in resolution['records']}
    rows, audit, by_entry = [], [], collections.defaultdict(list)
    directives, inverse = collections.defaultdict(list), collections.defaultdict(list)
    for record in candidates:
        key = record['entry_key']
        item = dict(resolved[key], emitted_keys=[], decisions=[])
        audit.append(item)
        if '(cid:' in record['native'] or any('\ue000' <= c <= '\uf8ff' for c in record['native']):
            item.update(decision='excluded-corrupt-head', installed_rows=0)
            continue
        for sense in record['senses']:
            child = key + ':sense:' + (sense['printed_sense'] or '0')
            tags, notes, reasons = [], [], []
            for label in sense['pos']:
                if label in POS:
                    tags.append(POS[label])
                else:
                    notes.append('Source grammatical label: ' + label + '.')
                    reasons.append('grammar: unexpanded source POS label ' + label)
            gloss = sense['english_gloss']
            if gloss in CODES:
                gloss, code_tags = CODES[gloss]
                tags.extend(code_tags)
            elif re.fullmatch(r'[A-Z0-9=.¹]+', gloss):
                notes.append('Source grammatical code: ' + gloss + '.')
                reasons.append('grammar: source code not unambiguously expanded: ' + gloss)
                gloss = ''
            if 'unresolved-native-glyph' in record['review_flags']:
                reasons.append('transcription: printed dotted-circle sequence retained')
            if any(c in record['ipa'] for c in 'ᵍᵏ'):
                reasons.append('transcription: unresolved superscript-stop convention retained')
            if sense['scientific_names']:
                notes.append('Source scientific identification: ' +
                             '; '.join(sense['scientific_names']) + '.')
                if not gloss:
                    gloss = '; '.join(sense['scientific_names'])
            etymology = ''
            for annotation in record['annotations']:
                if annotation in {'(sad)', '(hin, sad)'}:
                    etymology += ('Source language label: ' + annotation +
                                  ' (sad = Sadri; hin = Hindi). No donor entry is specified. ')
            for ref in record['references']:
                if ref['kind'].startswith('lexical-') or ref['kind'] == 'complex-form-of':
                    notes.append('Source entry ' + ref['label'] + ': ' +
                                 ', '.join(t['native'] + (' (sense ' + t['printed_sense'] + ')' if t['printed_sense'] else '')
                                           for t in ref['targets']) + '.')
            if reasons:
                tags.append('uncertain')
            citation = f"{SOURCE}[p. {record['printed_page']}, col. {record['column']}"
            if sense['printed_sense']:
                citation += ', sense ' + sense['printed_sense']
            citation += ']'
            row = ['Asuri', '', record['ipa'] or record['native'], gloss,
                   record['native'], record['ipa'], ' '.join(notes), citation, '',
                   etymology.strip(), child, '', '', '', ' '.join(dict.fromkeys(tags))]
            row = [nfc(v) for v in row]
            rows.append(row)
            by_entry[key].append(row)
            item['emitted_keys'].append(child)
            item['decisions'].append({'row_key': child, 'review_reasons': reasons})
        item.update(decision='proposed', installed_rows=0, proposed_rows=len(item['emitted_keys']))

    # Both forward cross-references and entry-level lists explicitly assert
    # variants. Nested references describe another referenced word, not this head.
    for key, item in resolved.items():
        for ref in item['references']:
            if ref['scope'] != 'entry':
                continue
            for target in ref['targets']:
                if target['status'] != 'unique':
                    continue
                other = target['candidate_keys'][0]
                if ref['kind'] == 'variant-of':
                    directives[key].append((other, target['printed_sense']))
                elif ref['kind'] == 'variant-list':
                    inverse[other].append((key, ''))

    # An explicit "B var. of C" gives the direction more precisely than A's
    # list of related variants B and C. Keep that list in the source audit.
    for child, targets in inverse.items():
        if child not in directives:
            directives[child] = targets

    proposals, pending = {}, []
    for key, item in resolved.items():
        for ref in item['references']:
            for target in ref['targets']:
                if ref['scope'] == 'entry' and ref['kind'] == 'variant-of' and target['status'] != 'unique':
                    pending.append({'entry_key': key, 'reason': 'unmatched printed variant target', 'target': target})
    for child, targets in directives.items():
        targets = list(dict.fromkeys(targets))
        if len(targets) != 1 or len(by_entry[child]) != 1:
            pending.append({'entry_key': child, 'reason': 'multiple targets or child senses', 'targets': targets})
            continue
        parent, sense = targets[0]
        parents = by_entry[parent]
        if sense:
            parents = [r for r in parents if r[10].endswith(':sense:' + sense)]
        if len(parents) != 1:
            pending.append({'entry_key': child, 'reason': 'unresolved target sense', 'targets': targets})
            continue
        proposals[by_entry[child][0][10]] = parents[0][10]
    # Reject every cycle member before writing any rank-1 relationship.
    cyclic = set()
    for start in proposals:
        trail, current = [], start
        while current in proposals:
            if current in trail:
                cyclic.update(trail[trail.index(current):])
                break
            trail.append(current)
            current = proposals[current]
    keyed = {r[10]: r for r in rows}
    for child, parent in proposals.items():
        if child in cyclic:
            pending.append({'row_key': child, 'reason': 'source-reference cycle', 'target': parent})
        else:
            keyed[child][11] = parent
    # Embedded subentries also have their own alphabetic cross-reference entry.
    # Reconcile onto that stable record and retain both printed locators.
    native_index = collections.defaultdict(list)
    for record in candidates:
        native_index[record['native']].append(record['entry_key'])
    for record in candidates:
        for subentry in record['inline_compounds']:
            matches = native_index[subentry['native']]
            if len(matches) != 1 or len(by_entry[matches[0]]) != 1:
                pending.append({'entry_key': record['entry_key'], 'reason': 'unresolved embedded subentry', 'subentry': subentry})
                continue
            row = by_entry[matches[0]][0]
            translation = INLINE_DEFINITIONS.get(record['entry_key'], {}).get(subentry['native'])
            if translation:
                assert not row[3], 'Do not overwrite an independently printed definition'
                row[3] = translation[1]
                row[6] += (' ' if row[6] else '') + 'Hindi definition: ' + translation[0] + '.'
            row[14] = ' '.join(dict.fromkeys(row[14].split() + [POS[p] for p in subentry['pos'] if p in POS]))
            row[7] += f";{SOURCE}[p. {record['printed_page']}, col. {record['column']}, under {record['native']}]"
            for item in audit:
                if item['entry_key'] == matches[0]:
                    item['decisions'].append({'embedded_definition_entry': record['entry_key'],
                                              'source_subentry': subentry,
                                              'reviewed_Hindi_and_translation': translation})
    for key, item in resolved.items():
        for ref in item['references']:
            if ref['kind'] != 'compound-of' or ref['scope'] != 'entry' or len(by_entry[key]) != 1:
                continue
            parents = []
            for target in ref['targets']:
                matches = by_entry[target['candidate_keys'][0]] if target['status'] == 'unique' else []
                if len(matches) != 1:
                    break
                parents.append(matches[0][10])
            if parents and len(parents) == len(ref['targets']):
                by_entry[key][0][13] = '|'.join(parents)
    for item in pending:
        entry = item.get('entry_key', '')
        affected = by_entry[entry] if entry else [keyed[item['row_key']]]
        for row in affected:
            row[14] = ' '.join(dict.fromkeys(row[14].split() + ['uncertain']))
        for record in audit:
            if record['entry_key'] == entry:
                record['decisions'].append({'review_reason': 'variant: ' + item['reason']})
    for record in audit:
        record['emitted_rows'] = by_entry[record['entry_key']]
    assert len(keyed) == len(rows)
    assert all(len(r) == 15 and r[2] and (not r[11] or r[11] in keyed) for r in rows)
    return rows, {'status': 'proposal-not-installed', 'records': audit,
                  'pending_relationships': pending,
                  'limitations': ['full compiled integration and browser checks remain open']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    if args.output_dir and args.install:
        parser.error('Choose proposal --output-dir or canonical --install, not both')
    candidates = list(map(json.loads, (HERE / 'candidates.jsonl').open()))
    rows, report = build(candidates, json.loads((HERE / 'audit.json').read_text()))
    if not args.output_dir and not args.install:
        parser.error('Pass --output-dir or --install')
    output = args.output_dir or HERE.parent.parent
    output.mkdir(parents=True, exist_ok=True)
    with (output / (STEM + '.csv')).open('w', newline='') as out:
        csv.writer(out).writerows(rows)
    if args.install:
        report['status'] = 'source-inputs-installed-integration-pending'
        for record in report['records']:
            if record['decision'] == 'proposed':
                record['decision'] = 'ingested'
                record['installed_rows'] = record['proposed_rows']
    (HERE if args.install else output).joinpath(STEM + '-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'records': len(candidates), 'proposed_rows': len(rows),
                      'variant_links': sum(bool(r[11]) for r in rows),
                      'pending_relationships': len(report['pending_relationships'])}))
