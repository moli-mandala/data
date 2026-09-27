"""Account for the complete pinned Gorum MLA snapshot.

This writes only the source CSV and local audit. It never builds Jambu's database.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'source-wayback.txt'
CSV = HERE.parents[1] / '20260925-donegan-stampe-gorum.csv'
AUDIT = HERE / 'audit.jsonl'
SOURCE_SHA256 = '17c19bd2158d5c7ec3bd3c6afd9c6c653b1a7f33b6246a36f4e6b831a2ab418b'
SOURCE_KEY = 'DSGO'
POS = {
    'N': 'noun', 'NB': 'noun', 'NK': 'noun',
    'V': 'verb', 'V(u)': 'verb', 'V(nu)': 'verb', 'V(u/nu)': 'verb',
    'ADJ': 'adj', 'ADV': 'adv', 'NUM': 'num', 'PP': 'postp',
    'PRON': 'pron', 'CONJ': 'conj', 'INTERJ': 'interj',
}


def raw_records(data: bytes) -> list[dict]:
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('Gorum archive snapshot SHA-256 changed')
    parts = data.decode('ascii').split('\f')
    if len(parts) != 3 or 'Munda Lexical Archive' not in parts[0] or 'Next ID:3855' not in parts[2]:
        raise ValueError('Gorum archive boundaries changed')
    body = parts[1]
    matches = list(re.finditer(r'#(\d+)\.', body))
    if len(matches) != 5824:
        raise ValueError(f'Expected 5824 numbered chunks, found {len(matches)}')
    records = []
    previous = 0
    for ordinal, match in enumerate(matches, 1):
        records.append({'ordinal': ordinal, 'source_id': match.group(1),
                        'raw': body[previous:match.end()].strip()})
        previous = match.end()
    if body[previous:].strip():
        raise ValueError('Unexpected unnumbered source text')
    return records


def legacy_parse(record: dict, id_counts: collections.Counter) -> tuple[list[str] | None, dict]:
    raw = record['raw']
    source_id = record['source_id']
    key = f'gorum-mla2004:{source_id}'
    audit = {'entry_key': key, 'ordinal': record['ordinal'], 'source_id': source_id, 'raw': raw}
    header = raw.split('{', 1)[0]
    heads = re.findall(r'<([^<>]+)>', header)
    gloss_match = re.search(r"``(.*?)''", raw, re.S)
    labels = re.findall(r'\{([^}]+)\}', raw[:gloss_match.start()] if gloss_match else header)
    audit.update({'headwords': heads, 'source_labels': labels})

    def skip(reason: str):
        audit.update({'status': 'excluded', 'reason': reason})
        return None, audit

    if id_counts[source_id] != 1:
        return skip('source numeric ID repeats; ambiguous stable identity')
    if len(heads) != 1 or not re.search(r'\(Z\)\s*$', header):
        return skip('not one unqualified Z-marked headword')
    form = heads[0]
    if not re.fullmatch(r'[a-z-]+', form):
        return skip('source transcription or segmentation needs review')
    if not gloss_match:
        return skip('no bounded lexical gloss')
    source_gloss = gloss_match.group(1)
    if not source_gloss.strip('? ') or any(s in source_gloss for s in ('?', '<', '>', '^G')):
        return skip('uncertain, placeholder, or embedded-form gloss')
    if len(labels) != 1 or labels[0] not in POS:
        return skip('unmapped or mixed grammatical label')
    gloss = re.sub(r'\s+', ' ', source_gloss.replace('^', '').replace('_', ' ')).strip()
    if not gloss or '\ufffd' in gloss or len(gloss) > 100:
        return skip('empty, corrupt, or extended gloss')
    tags = [POS[labels[0]]]
    if '??' in raw:
        tags.append('uncertain')
        audit['review_reason'] = 'source editorial uncertainty; scope may concern analysis, not form'
    citation = f'{SOURCE_KEY}[entry {source_id}]'
    row = ['go', '', form, gloss, '', '', '', citation, '', '', key, '', '', '', ' '.join(tags)]
    audit.update({'status': 'ingested', 'form': form, 'gloss': gloss,
                  'tags': tags, 'citation': citation, 'source_witness': 'Z marker; referent not inferred'})
    return row, audit



def physical_records(data: bytes) -> list[dict]:
    """Account for every nonblank physical line, merging explicit continuation only."""
    raw_records(data)  # pin snapshot and numeric census
    records = []
    for line, raw in enumerate(data.decode('ascii').split('\f')[1].splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        if raw.startswith('|') and records and re.search(r'#\d+\.', raw):
            records[-1]['raw'] += '\n' + raw
            records[-1]['line_end'] = line
            records[-1]['source_id'] = re.search(r'#(\d+)\.', raw).group(1)
            continue
        number = re.search(r'#(\d+)\.', raw)
        # The numeral nine has its number but lacks only the hash delimiter.
        if raw.startswith('<mulgi>') and raw.endswith('22960.'):
            number = re.search(r'(22960)\.$', raw)
        records.append({'line': line, 'line_end': line, 'source_id': number.group(1) if number else None,
                        'raw': raw})
    assert len(records) == 6562
    return records


def clean(text):
    return re.sub(r'\s+', ' ', text.replace('^', '').replace('_', ' ').replace('%', '').replace('<', '').replace('>', '')).strip()


def grammar(label):
    tags = []
    if re.search(r'(?:^|[ ,;])V(?:\b|ay)', label): tags.append('verb')
    mapping = {'N': 'noun', 'NB': 'noun', 'NK': 'noun', 'ADJ': 'adj', 'ADV': 'adv',
               'NUM': 'num', 'PP': 'postp', 'PRON': 'pron', 'CONJ': 'conj', 'INTERJ': 'interj',
               'DEM': 'demonstrative', 'INTERR': 'interr', 'INDECL': 'indecl', 'IND': 'indecl',
               'PX': 'prefix', 'SX': 'suffix', 'AUX': 'auxiliary', 'NEG': 'neg', 'PL': 'pl',
               'PAST': 'pret'}
    for token in re.findall(r'[A-Z]+', label):
        if token in mapping and mapping[token] not in tags: tags.append(mapping[token])
    if label == 'VP': tags += ['verb', 'multiword-expression']
    if label == 'VX': tags.append('verb')
    if label == 'NP': tags += ['noun', 'proper-noun']
    if re.search(r'\bNK\b', label): tags.append('kinship')
    if 'EMPHATIC' in label: tags += ['emph', 'part']
    if 'clitic' in label.lower(): tags.append('part')
    return tags


def gloss_for_witness(text, witness):
    """Split only top-level gloss commas/semicolons; suffix labels bind locally."""
    if '``' in text:
        # Three malformed entries restart the opening quote at a semicolon;
        # each visibly quoted synonym group has its own terminal witness label.
        groups = re.split(r';\s*``', text)
        retained = []
        for group in groups:
            mark = re.search(r'\(([ZA]+)\)\s*$', group)
            if mark and not set(witness).intersection(mark.group(1)): continue
            retained.append(clean(re.sub(r'\([ZA]+\)', '', group).replace('``', '')))
        return '; '.join(retained)
    pieces, start, depth = [], 0, 0
    for i, ch in enumerate(text):
        if ch == '(': depth += 1
        elif ch == ')': depth = max(depth - 1, 0)
        elif ch in ',;' and not depth:
            pieces.append((text[start:i], ch)); start = i + 1
    pieces.append((text[start:], ''))
    selected = []
    for piece, separator in pieces:
        mark = re.search(r'\(([ZA]+)\)\s*$', piece)
        if mark:
            if not set(witness).intersection(mark.group(1)):
                continue
            piece = piece[:mark.start()]
        selected.append((clean(piece), separator))
    return ''.join(part + (sep + ' ' if i < len(selected)-1 else '')
                   for i, (part, sep) in enumerate(selected) if part).strip()


def personal_grammar(gloss, label, source_id):
    """Extract explicit person/number/case, retaining source contradictions."""
    tags, flags = [], []
    if not re.search(r'PRON|INTERR|PX|SX', label): return gloss, tags, flags
    pattern = r'(1st|2nd|3rd|first|second|third) person (singular|plural)(?: (subject|object|possessive))?'
    match = re.search(pattern, gloss)
    if match:
        person = {'1st': '1', 'first': '1', '2nd': '2', 'second': '2', '3rd': '3', 'third': '3'}[match[1]]
        number = {'singular': 'sg', 'plural': 'pl'}[match[2]]
        if source_id in ('24240', '19830'):
            flags.append('source_person_number_contradiction')
        else:
            tags.append(person + number)
            candidate = gloss[:match.start()] + gloss[match.end():]
            candidate = re.sub(r'\(\s*\)', '', candidate).strip(' (,;')
            if candidate: gloss = candidate
        if match[3]: tags.append({'subject': 'subj', 'object': 'obj', 'possessive': 'poss'}[match[3]])
    if source_id == '10660': tags.append('3pl')
    if source_id == '9350': tags += ['acc', 'obj']
    for marker, tag in [('objective', 'obj'), ('subject', 'subj'), ('object', 'obj')]:
        if '(' + marker + ')' in gloss:
            tags.append(tag); gloss = clean(gloss.replace('(' + marker + ')', ''))
    return clean(gloss), tags, flags


def parse_unit(record, key):
    raw = record['raw']
    repairs = []
    if record['source_id'] in ('3650', '34620', '34622'):
        raw = re.sub(r"(?<!')'(?=\. )", "''", raw, count=1)
        repairs.append('Recovered one missing closing gloss delimiter; literal source retained in raw')
    if record['source_id'] == '24670':
        raw = raw.replace('<ol)', '<ol>', 1)
        repairs.append('Recovered malformed closing headword delimiter <ol); source letters unchanged')
    audit = dict(record, entry_key=key, emitted_keys=[], held_senses=[], delimiter_repairs=repairs)
    quote = re.search(r"``", raw)
    cutoff = min([m.start() for m in re.finditer(r'\{|``|\?\?|\bCf\.', raw)] or [len(raw)])
    header = raw[:cutoff]
    heads = []
    for match in re.finditer(r'(?<!<)<([^<>]+)>(?!>)(?:\(([^)]*)\))?', header):
        heads.append({'form': match.group(1), 'witness': match.group(2) or '', 'start': match.start(), 'end': match.end()})
    # Unlabelled coordinated head forms share a following explicitly printed label.
    for i, head in enumerate(heads):
        if not head['witness']:
            for later in heads[i+1:]:
                if later['witness']:
                    head['witness'] = later['witness']; break
    audit['headwords'] = heads
    audit['header_analysis'] = re.findall(r'<<([^<>]+)>>', header)
    audit['bracket_transcriptions'] = re.findall(r'(?<!\w)\[([^]]+)\]', re.sub(r'<[^<>]+>', '', header))
    if not heads:
        audit.update(status='held' if quote else 'excluded_context', reason='Malformed/headword-less lexical record' if quote else 'Nonlexical separator or empty numeric record')
        return [], audit
    if not quote:
        audit.update(status='excluded_context', reason='Unglossed grouping/root heading or editorial speculation; no bounded lexical definition')
        return [], audit
    senses = []
    previous = cutoff
    for sense_num, match in enumerate(re.finditer(r"``(.*?)''", raw), 1):
        between = raw[previous:match.start()]
        if re.search(r'\bCf\.|\s\||\*Des\.', between):
            break  # quoted editorial commentary is not a further lexical sense
        labels = re.findall(r'\{([^}]*)\}', between)
        senses.append({'number': sense_num, 'label': labels[-1] if labels else '', 'gloss': match.group(1)})
        previous = match.end()
    audit['senses'] = senses
    known_witnesses = set(''.join(h['witness'] for h in heads))
    audit['unassigned_witness_senses'] = [dict(sense=se['number'], witness=w, raw_gloss=se['gloss'], reason='Gloss witness absent from headword declarations; no unsupported form alignment inferred') for se in senses for w in sorted(set(''.join(re.findall(r'\(([ZA]+)\)', se['gloss']))) - known_witnesses)]
    commentary = re.sub(r'\s*#\d+\.\s*$', '', raw[previous:]).strip(' .')
    audit['source_commentary'] = commentary
    rows = []
    for sense in senses:
        first_key = None
        first_witness = None
        scoped_heads = []
        for hi, head in enumerate(heads, 1):
            labels = list(head['witness']) if head['witness'] in ('ZA', 'AZ') and re.search(r'\([ZA]+\)', sense['gloss']) else [head['witness']]
            for wi, witness in enumerate(labels):
                scoped_heads.append((hi, head['form'], witness, '' if wi == 0 else ':witness:' + witness))
        for hi, form, witness, witness_suffix in scoped_heads:
            if witness not in ('Z', 'A', 'ZA', 'AZ', ':', ':Z', ':A', '', 'Bh.', 'S'):
                audit['held_senses'].append(dict(sense=sense['number'], head=hi, reason='Unresolved witness/language attribution', witness=witness))
                continue
            source_gloss = sense['gloss']
            nested_grammar = []
            # One source construction scopes valency inside the explanatory parentheses.
            for marker, scope in re.findall(r'(tr\.|intr\.|caus\.)\(([ZA]+)\)', source_gloss):
                if set(witness).intersection(scope): nested_grammar.append({'tr.': 'tr', 'intr.': 'intr', 'caus.': 'caus'}[marker])
            if nested_grammar:
                source_gloss = re.sub(r'\((?:tr\.|intr\.|caus\.)\([ZA]+\)(?:,\s*(?:tr\.|intr\.|caus\.)\([ZA]+\))*\)', '', source_gloss)
            gloss = gloss_for_witness(source_gloss, witness)
            if not gloss.strip(' ?.,;'):
                audit['held_senses'].append(dict(sense=sense['number'], head=hi, reason='Empty/placeholder or nonapplicable witness gloss'))
                continue
            rowkey = (key if sense['number'] == 1 and hi == 1 else f'{key}:sense:{sense["number"]}:head:{hi}') + witness_suffix
            tags = grammar(sense['label']) + nested_grammar
            for marker, tag in [('tr.', 'tr'), ('intr.', 'intr'), ('caus.', 'caus')]:
                if '(' + marker + ')' in gloss:
                    tags.append(tag); gloss = clean(gloss.replace('(' + marker + ')', ''))
                if ', ' + marker + ')' in gloss:
                    tags.append(tag); gloss = gloss.replace(', ' + marker + ')', ')')
            donor_claim = re.search(r'\*\??(?:Loan|Engl?|Des|Or|IA|Skt|Hi|Pers|Tel?|Dr(?:av(?:idian)?)?|Kon[Dd]a)\b', commentary, re.I)
            if donor_claim: tags.append('loanword')
            gloss, personal_tags, uncertainties = personal_grammar(gloss, sense['label'], record['source_id'])
            tags.extend(personal_tags)
            if '??' in raw: uncertainties.append('source_editorial')
            if repairs: uncertainties.append('source_markup_repair')
            if '?' in gloss or '?' in sense['label']: uncertainties.append('gloss_or_grammar')
            if donor_claim and '?' in commentary[donor_claim.start():donor_claim.end()+2]: uncertainties.append('borrowing')
            if sense['label'] in ('VX', 'PARC', 'NCF', 'NCF?'): uncertainties.append('source_grammar_class')
            if re.search(r'[^a-z -]', form): uncertainties.append('transcription_preserved')
            if witness in ('', ':'): uncertainties.append('witness_unspecified')
            if witness in ('Bh.', 'S'): uncertainties.append('witness_unresolved')
            if not tags: uncertainties.append('grammar_unmapped')
            if uncertainties: tags.append('uncertain')
            locator = f'entry {record["source_id"]}' if record['source_id'] else f'unnumbered line {record["line"]}'
            citation = f'{SOURCE_KEY}[{locator}, witness {witness or "unspecified"}, sense {sense["number"]}]'
            # Keep first-generation keys/citations for all 559 previously installed records.
            if rowkey in LEGACY:
                citation = LEGACY[rowkey][7]
            analysis = commentary
            if audit['header_analysis']:
                analysis = 'Source header analysis: ' + '; '.join(audit['header_analysis']) + '. ' + analysis
            variant = ''
            if first_key and witness == first_witness and '??' not in raw and ',,' in header and '/' not in header and '\\' not in header:
                variant = first_key
            phonemic = audit['bracket_transcriptions'][0] if len(heads) == 1 and len(audit['bracket_transcriptions']) == 1 and len(form.split()) == len(audit['bracket_transcriptions'][0].split()) and len(audit['bracket_transcriptions'][0]) * 2 >= len(re.sub(r'[-=.~?\[\]]', '', form)) else ''
            note = ('Source header/component bracket transcription(s), full-form alignment not established: ' + '; '.join('[' + x + ']' for x in audit['bracket_transcriptions'])) if audit['bracket_transcriptions'] and not phonemic else ''
            row = ['go', '', form, gloss, '', phonemic, note, citation, '', analysis, rowkey, variant, '', '', ' '.join(dict.fromkeys(tags))]
            rows.append(row)
            audit['emitted_keys'].append(rowkey)
            audit.setdefault('outputs', []).append({'entry_key': rowkey, 'form': form, 'gloss': gloss, 'witness': witness,
                 'source_label': sense['label'], 'tags': tags, 'phonemic': phonemic, 'notes': note, 'uncertainty_types': uncertainties})
            if first_key is None: first_key, first_witness = rowkey, witness
    audit.update(status='ingested' if rows else 'held', reason='All structurally scoped target forms/senses retained' if rows else 'No safely attributed non-placeholder lexical output')
    return rows, audit


# Pin previous installed identities independently of expanded parsing policy.
LEGACY = {}


def prepare() -> tuple[list[list[str]], list[dict]]:
    data = SOURCE.read_bytes()
    old_records = raw_records(data)
    counts = collections.Counter(r['source_id'] for r in old_records)
    LEGACY.clear()
    for record in old_records:
        row, _ = legacy_parse(record, counts)
        if row: LEGACY[row[10]] = row
    assert len(LEGACY) == 559
    duplicate = {x['source_id']: x for x in json.loads((HERE/'duplicate-reconciliation.json').read_text())['entries']}
    rows, audit, seen = [], [], collections.Counter()
    first_by_id = {}
    for record in physical_records(data):
        sid = record['source_id']
        seen[sid] += 1
        key = f'gorum-mla2004:{sid}' if sid else f'gorum-mla2004:unnumbered:{record["line"]}'
        decision = duplicate.get(sid)
        if sid and seen[sid] > 1:
            key += f':occurrence:{seen[sid]}'
        emitted, item = parse_unit(record, key)
        if sid == '6680':
            remark = record['raw'].split('??(Z) has ', 1)[1]
            extra, extra_audit = parse_unit(dict(record, raw=remark), key + ':source-remark')
            for row in extra:
                if 'uncertain' not in row[14].split(): row[14] += ' uncertain'
            emitted.extend(extra)
            item['source_remark'] = extra_audit
            item['emitted_keys'].extend(extra_audit['emitted_keys'])
        if decision:
            item['duplicate_decision'] = decision['decision']
            item['duplicate_reason'] = decision['reason']
            if decision['decision'] in ('hold_id_collision', 'hold_headword_conflict'):
                item['identity_uncertainty'] = 'Duplicate numeric ID does not establish same lexical identity; occurrence keys preserve distinct records'
                for row in emitted:
                    if 'uncertain' not in row[14].split(): row[14] += ' uncertain'
        if sid and seen[sid] > 1 and decision and decision['reuse_safe']:
            originals = first_by_id[sid]
            item['status'] = 'same_source_reuse'
            item['reuse_keys'] = [r[10] for r in originals]
            item['emitted_keys'] = []
            # All distinct source analysis survives on the retained lexical record.
            for row in originals:
                texts = list(dict.fromkeys([row[9]] + [r[9] for r in emitted if r[2] == row[2]]))
                row[9] = '\n'.join(t for t in texts if t)
            emitted = []
        elif sid:
            first_by_id.setdefault(sid, emitted)
        rows.extend(emitted); audit.append(item)
    assert set(LEGACY) <= {r[10] for r in rows}
    assert len({r[10] for r in rows}) == len(rows)
    return rows, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true', help='write canonical source CSV only after review')
    args = parser.parse_args()
    rows, audit = prepare()
    target = CSV if args.install else HERE/'staged.csv'
    with target.open('w', newline='', encoding='utf-8') as stream:
        csv.writer(stream).writerows(rows)
    with (AUDIT if args.install else HERE/'staged-audit.jsonl').open('w', encoding='utf-8') as stream:
        for item in audit: stream.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(json.dumps({'physical_units': len(audit), 'selected_rows': len(rows), 'statuses': collections.Counter(a['status'] for a in audit)}, sort_keys=True))


if __name__ == '__main__':
    main()
