"""Prepare a rich CSV preview; installation remains disabled pending source review."""
import argparse
import csv
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
from tags import GRAMMATICAL_TAGS, GENDER_TAGS

SOURCE = 'bhattacharya1957ollari'
POS = {'sb.':['noun'], 'vb.':['verb'], 'adj.':['adj'], 'adv.':['adv'],
       'pron.':['pron'], 'num.':['num'], 'postpos.':['postp'],
       'conj.':['conj'], 'indecl.':['indecl'], 'infl. adj.':['adj'],
       'vb. cs.':['verb','caus'], 'vb.cs.':['verb','caus'],
       'vb. intr.':['verb','intr'], 'vb. tr.':['verb','tr'],
       'pron. indecl.':['pron','indecl']}


def grammar(unit):
    tags = []
    for part in filter(None, (s.strip() for s in unit['source_pos'].split(';'))):
        if part not in POS:
            raise ValueError(f'Unreviewed grammatical label: {part}')
        tags.extend(POS[part])
    gloss = unit['gloss']
    match = re.search(r'\s*\((m\.|m\. n\.|f\. n\.|n\.)\)$', gloss)
    if match:
        tags.append({'m.':'m','m. n.':'mn','f. n.':'fn','n.':'n'}[match[1]])
        gloss = gloss[:match.start()]
    if unit['unit_kind'] == 'inflection':
        label = unit['morphology_label']
        if label == 'pl.': tags.append('pl')
        elif label == 'obl. stem': tags.extend(['obl','stem'])
        else: raise ValueError(f'Unreviewed inflection label: {label}')
    if unit['uncertainties']: tags.append('uncertain')
    tags = list(dict.fromkeys(tags))
    assert set(tags) <= GRAMMATICAL_TAGS | GENDER_TAGS
    return gloss, tags


def build(output):
    # Regenerate from authoritative review files, not a stale draft snapshot.
    import expand_lexical_units
    import convert_draft
    records = [r for f in sorted(HERE.glob('reviewed-p*.json')) for r in json.loads(f.read_text())]
    units, audit = expand_lexical_units.expand(records)
    source_records = {r['entry_key']: r for r in records}
    rows, row_audit = [], []
    keys = {u['entry_key'] for u in units}
    for unit in units:
        record = source_records[unit['physical_entry_key']]
        gloss, tags = grammar(unit)
        converted = convert_draft.convert(unit['form'], unit['pronunciation_overrides'])
        variant = unit['physical_entry_key'] if unit['unit_kind'] == 'printed-alternate' else ''
        parents = unit['physical_entry_key'] if unit['unit_kind'] == 'inflection' else record['explicit_derivation_parent'] if unit['unit_kind'] == 'headword' else ''
        for target in [variant, parents]:
            if target and target not in keys: raise ValueError(f'Missing source-local parent: {target}')
        citation = f"{SOURCE}[p. {unit['printed_page']}, col. {unit['column']}]"
        # Only author-specified pronunciation creates a separate phonemic layer.
        phonemic = converted['pronunciation_input'] if unit['pronunciation_overrides'] else ''
        # Prose and unresolved comparisons stay in the complete record audit until
        # their semantic roles and citations are reviewed for sidecar publication.
        row = ['OllariGadaba','',unit['form'],gloss,'',phonemic,'',citation,'','',unit['entry_key'],variant,'',parents,' '.join(tags)]
        rows.append(row)
        row_audit.append(dict(entry_key=unit['entry_key'], physical_entry_key=unit['physical_entry_key'],
                             proposed_row=row, expected_display=converted['display'],
                             relation_basis=unit['unit_kind'], status='preview-not-installed',
                             pending=['comparative-prose-and-graph-review','fresh-source-to-output-audit']))
    output.mkdir(parents=True,exist_ok=True)
    with (output/'20260921-bhattacharya-ollari.csv').open('w',newline='') as stream:
        csv.writer(stream).writerows(rows)
    (output/'20260921-bhattacharya-ollari.yaml').write_text('''file: 20260921-bhattacharya-ollari.csv
defaults:
  transcription:
    profile: bhattacharya-ollari
    input: phonemic
    preserve_hyphens: true
sources:
  bhattacharya1957ollari:
    identity:
      dedupe_by_entry_key: true
    forms:
      split_alternates: false
    reference:
      editor: Aryaman Arora; OpenAI Codex
      ocr: true
      etymology_provenance: source
''')
    for name, values in [('rich-row-audit.jsonl',row_audit),('physical-record-audit.jsonl',audit)]:
        (output/name).write_text(''.join(json.dumps(v,ensure_ascii=False,sort_keys=True)+'\n' for v in values))
    summary=dict(status='preview-not-installed',physical_records=len(records),rows=len(rows),
                 variants=sum(bool(r[11]) for r in rows),derivations=sum(bool(r[13]) for r in rows),
                 phonemic_rows=sum(bool(r[5]) for r in rows),uncertain_rows=sum('uncertain' in r[14].split() for r in rows),
                 pending_suffix_scopes=sum(len(a['pending_morphology']) for a in audit),
                 missing_gloss=sum(not r[3] for r in rows),
                 gates_remaining=['comparative review and reference resolution','suffix scope and uncertainty disposition',
                                  'fresh seeded source-to-output audit','bibliography and final settings registration',
                                  'installation and complete build/suite'])
    (output/'rich-preview-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.output),ensure_ascii=False,indent=2))
