"""Validate source-aware display conversion without installing or changing Original."""
import argparse
import json
from pathlib import Path
import unicodedata
from segments.tokenizer import Tokenizer

HERE = Path(__file__).resolve().parent
PROFILE = HERE.parents[4] / 'conversion/bhattacharya-ollari.txt'


def convert(form, overrides=()):
    original = unicodedata.normalize('NFC', form)
    phonetic_input = original
    for rule in overrides:
        if rule['scope'] != 'entry' or (rule['source'], rule['value']) not in {('j', 'z'), ('j', 'dz'), ('c', 'ts')}:
            raise ValueError(f'Unreviewed pronunciation rule: {rule}')
        if rule['source'] not in phonetic_input:
            raise ValueError(f'Pronunciation rule has no source symbol: {original}')
        phonetic_input = phonetic_input.replace(rule['source'], rule['value'])
    display = unicodedata.normalize('NFC', TOKENIZER(phonetic_input, column='IPA').replace(' ', '').replace('#', ' '))
    if '\ufffd' in display:
        raise ValueError(f'Unmapped form: {original}')
    return {'original': original, 'pronunciation_input': phonetic_input, 'display': display}


TOKENIZER = Tokenizer(str(PROFILE))


def build(output):
    units = [json.loads(line) for line in (HERE / 'draft/lexical-units.jsonl').read_text().splitlines()]
    rows = []
    for unit in units:
        result = convert(unit['form'], unit['pronunciation_overrides'])
        rows.append(dict(entry_key=unit['entry_key'], **result,
                         pronunciation_overrides=unit['pronunciation_overrides'],
                         uncertainties=unit['uncertainties'], status='draft-not-installed'))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'converted-units.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows))
    report = {'draft_units': len(rows), 'unmapped_forms': 0,
              'changed_display_forms': sum(r['display'] != r['original'] for r in rows),
              'entry_pronunciation_units': sum(bool(r['pronunciation_overrides']) for r in rows),
              'input_characters': sorted(set(''.join(r['original'] for r in rows))),
              'status':'source-local conversion validated; pipeline routing pending',
              'evidence':'Printed pp.9–11 and entry-local pronunciation annotations',
              'decisions': ['Preserve vowel quality, vowel length, retroflex contrasts, spaces and verb hyphens.',
                            'Map ṅ to house ŋ; map nasal vowel plus raised dot to nasalized macron vowel.',
                            'Apply j=z, j=dz and c=ts only under reviewed entry-level scope.',
                            'House affricates ts/dz are ʦ/ʣ; plain c/j remain c/j.',
                            'Uncertain f in ulfṭe is preserved and remains typed uncertain.',
                            'No global sandhi, nasal assimilation or comparative correspondence rules.'],
              'limitations':['CSV/profile routing must preserve original source form while applying per-entry notes.',
                             'Coverage is over the current draft; final expanded rows must be checked again.']}
    (output/'conversion-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.output),ensure_ascii=False,indent=2))
