#!/usr/bin/env python3
"""Install structurally resolved Sheth DDSA records; quarantine unresolved articles.

The full 952-page digital snapshot is audited. This does not claim complete
coverage of the printed edition or resolve the draft's embedded subentries.
No historical links are inferred from Sanskrit equivalents or See references.
"""
from __future__ import annotations
import argparse
import collections
import csv
import gzip
import hashlib
import importlib.util
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PACKAGE = HERE / 'sheth_2026'
FILENAME = '20260914-sheth.csv'
SOURCE = 'sheth1923'
SPEC = importlib.util.spec_from_file_location('sheth_parser', HERE / 'sheth_parse.py')
parser = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parser)
SOURCE_SPEC = importlib.util.spec_from_file_location('sheth_source_catalog', ROOT / 'sheth_sources.py')
source_catalog = importlib.util.module_from_spec(SOURCE_SPEC)
SOURCE_SPEC.loader.exec_module(source_catalog)
DIALECTS = {
    'शौ': 'dialect:Pk:s:Shauraseni',
    'मा': 'dialect:Pk:mg:Magadhi',
    'पै': 'dialect:Pk:pais:Paishachi',
}
# Source romanization is preserved. Keep the bound-form marker and combining
# candrabindu, but reject digit, question-mark and other damaged headwords.
HEAD = re.compile(r"[a-zāīūēōṛṟḷṅñṇṭḍśṣḥṃẏ\u0310°'\-]+\Z")


def prepare(markup, page, ordinal):
    record = parser.parse_article(markup, page, ordinal)
    reasons = list(record['review'])
    for proposed in record['rows']:
        row = proposed['row']
        reasons.extend(proposed['review'])
        if not HEAD.fullmatch(row[2]):
            reasons.append('transcription:damaged-or-unrecognized-headword')
        # More than one source sentence often introduces an untagged finite
        # form/paradigm. Preserve the complete article for later segmentation.
        if '।' in row[3] or '.' in row[3]:
            reasons.append('structure:unsegmented-sentence-or-abbreviation')
        if any(c in row[3] for c in '{}=[]°'):
            reasons.append('structure:unresolved-layout-or-reference')
        if re.search(r'[०-९0-9()]', row[3]):
            reasons.append('reference:untagged-locator-or-parenthetical-review')
        if re.search(r"['‘’]", row[3]) and not re.search(r"['’]\s*अर्थ", row[3]):
            reasons.append('scope:unsegmented-quotation')
        if not row[3] and not proposed['crossreferences']:
            reasons.append('structure:no-definition-or-crossreference')
    reasons = list(dict.fromkeys(reasons))
    record['selection_reasons'] = reasons
    record['candidate_rows'] = record.pop('rows')
    record['rows'] = []
    record['source_tag_audit'] = []
    if reasons or record['status'] == 'excluded':
        record['status'] = 'audit-only'
        return record
    for proposed in record['candidate_rows']:
        row = proposed['row'].copy()
        tags = row[14].split()
        tags.extend(DIALECTS[label] for label in proposed['lect_labels'] if label in DIALECTS)
        source_tags, claims = source_catalog.reference_tags(proposed['references'])
        tags.extend(source_tags)
        record['source_tag_audit'].append({'entry_key': row[10], 'claims': claims})
        row[14] = ' '.join(dict.fromkeys(tags))
        if proposed['crossreferences']:
            row[6] = ' '.join(filter(None, [row[6], 'Source cross-reference: ' + '; '.join(proposed['crossreferences'])]))
        record['rows'].append(row)
    record['status'] = 'installed-input'
    record['reference_resolution'] = 'Primary DDSA locator resolved; verified work abbreviations become scoped source tags. Exact locators and unresolved codes remain in source_tag_audit; edition-specific auxiliary bibliography remains pending.'
    return record


def source_records(cache):
    """Read an immutable article snapshot, or the original page cache."""
    if cache.is_file():
        with gzip.open(cache, 'rt', encoding='utf-8') as stream:
            for line in stream:
                item = json.loads(line)
                yield item['raw_markup'], item['page'], item['ordinal']
        return
    manifest = json.loads((cache / 'manifest.json').read_text())
    assert manifest['complete'] and manifest['headwords'] == 41638
    assert len(manifest['pages']) == 952
    for item in manifest['pages']:
        raw = (cache / f"{item['page']:04d}.html").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item['sha256']
        soup = parser.BeautifulSoup(raw, 'html.parser')
        heads = soup.find_all('hw')
        assert len(heads) == item['headwords']
        for ordinal, head in enumerate(heads, 1):
            yield str(head.parent), item['page'], ordinal


def run(cache, output, seed):
    output.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    exclusions = collections.Counter()
    languages = collections.Counter()
    symbols = collections.Counter()
    source_tag_counts = collections.Counter()
    source_claim_status = collections.Counter()
    keys = set()
    sample = []
    rng = random.Random(seed)
    eligible = 0
    with (output / FILENAME).open('w', newline='', encoding='utf-8') as forms, gzip.open(output / 'audit.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1) as audit:
        writer = csv.writer(forms)
        for markup, page, ordinal in source_records(cache):
            record = prepare(markup, page, ordinal)
            counts['articles'] += 1
            counts[record['status']] += 1
            counts['candidate_rows'] += len(record['candidate_rows'])
            exclusions.update(record['selection_reasons'])
            if record['rows']:
                counts['unscoped_source_references'] += len(record['unscoped_references'])
            for detail in record['source_tag_audit']:
                source_claim_status.update(c['status'] for c in detail['claims'])
            for row in record['rows']:
                assert len(row) == 15 and row[10] not in keys
                keys.add(row[10])
                writer.writerow(row)
                counts['installed_rows'] += 1
                counts['variants'] += bool(row[11])
                languages[row[0]] += 1
                symbols.update(row[2])
                source_tags = [t for t in row[14].split() if t.startswith('Sheth:')]
                source_tag_counts.update(source_tags)
                counts['source_tagged_rows'] += bool(source_tags)
            audit.write(json.dumps(record, ensure_ascii=False) + '\n')
            if record['rows']:
                eligible += 1
                if len(sample) < 20:
                    sample.append(record)
                else:
                    index = rng.randrange(eligible)
                    if index < 20:
                        sample[index] = record
    assert counts['articles'] == 41638
    report = {'source': SOURCE, 'scope': 'Structurally resolved articles from the complete DDSA digital snapshot; unresolved articles/subentries audit-only', 'counts': dict(counts), 'languages': dict(languages), 'exclusion_reasons': dict(exclusions), 'symbols': dict(symbols), 'seed': seed, 'full_ingestion_complete': False, 'deferred': ['unresolved article segmentation and damaged heads', 'printed-edition/supplement reconciliation', 'auxiliary reference catalogue resolution', 'full CLDF build and full suite']}
    report['source_tag_counts'] = dict(source_tag_counts)
    report['source_reference_claim_status'] = dict(source_claim_status)
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (output / 'sample.json').write_text(json.dumps(sample, ensure_ascii=False, indent=2) + '\n')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', type=Path, default=PACKAGE / 'audit.jsonl.gz')
    ap.add_argument('--output', type=Path, default=ROOT / 'tmp/sheth-integration-20260914/proposal')
    ap.add_argument('--seed', type=int, default=20260914)
    ap.add_argument('--install', action='store_true')
    args = ap.parse_args()
    report = run(args.cache, args.output, args.seed)
    if args.install:
        import shutil
        # Installation uses only the fully written, audited proposal.
        shutil.copyfile(args.output / FILENAME, HERE.parent / FILENAME)
        PACKAGE.mkdir(exist_ok=True)
        for name in ['audit.jsonl.gz', 'report.json', 'sample.json']:
            shutil.copyfile(args.output / name, PACKAGE / name)
    print(json.dumps({k: v for k, v in report.items() if k not in ['exclusion_reasons', 'symbols']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
