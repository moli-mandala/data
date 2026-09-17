"""Extract source-local Sanskrit equivalents without asserting historical ancestry.

Sheth's title calls these Sanskrit equivalents; bracketed दे means देश्य-शब्द
(frontmatter PDF p. 2). Compounds, alternatives and uncertain readings stay audited.
"""
from __future__ import annotations
import argparse
import collections
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import shutil
import unicodedata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PACKAGE = HERE / 'sheth_2026'
FILENAME = '20260914-sheth-sanskrit.csv'
COMPARISONS = ROOT / 'data/other/comparisons/20260914-sheth-sanskrit.csv'
COLUMNS = ['ID', 'Entry_ID', 'Compared_Entry_ID', 'Relation', 'Direction', 'Confidence', 'Source', 'Evidence']
spec = importlib.util.spec_from_file_location('sheth_devanagari', ROOT / 'data/other/params/raw_data/wiktionary_piir.py')
transcription = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transcription)
WORD = re.compile(r'[अ-औक-हळािीुूृॄॢॣेैोौंँः्ॠॡऽ]+\Z')


def extract(text):
    text = unicodedata.normalize('NFC', text.strip())
    if not text:
        return {'status': 'absent'}
    desya = re.match(r'^दे(?=$|[.\s])\.?\s*', text)
    if desya:
        text = text[desya.end():].strip()
        if not text:
            return {'status': 'desya-label'}
    if '+' in text:
        return {'status': 'compound-analysis', 'components_raw': text.split('+')}
    if not WORD.fullmatch(text):
        return {'status': 'unresolved-expression', 'expression': text}
    roman = transcription.devanagari_to_iast(text)
    if not roman or not re.fullmatch(r"[a-zāīūṛṝḷḹṅñṇṭḍśṣḥṃ\u0310']+", roman):
        return {'status': 'unresolved-transcription', 'expression': text}
    # Same explicit long-e/o display convention as the Sheth headword profile.
    roman = roman.replace('e', 'ē').replace('o', 'ō')
    return {'status': 'sanskrit-equivalent', 'native': text, 'form': roman,
            'desya': bool(desya), 'relation': 'related', 'direction': 'undetermined'}


def run(output, seed=20260915):
    output.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    sample, eligible = [], 0
    rng = random.Random(seed)
    with gzip.open(PACKAGE / 'audit.jsonl.gz', 'rt') as source, \
         gzip.open(output / 'etymology-audit.jsonl.gz', 'wt', compresslevel=1) as audit, \
         (output / FILENAME).open('w', newline='') as forms, \
         (output / 'comparisons.csv').open('w', newline='') as comparisons:
        writer = csv.writer(forms)
        compare = csv.DictWriter(comparisons, fieldnames=COLUMNS)
        compare.writeheader()
        for line in source:
            article = json.loads(line)
            record = {'entry_key': article['entry_key'], 'raw_markup': article['raw_markup'],
                      'claims': [], 'excluded_article': not bool(article['rows'])}
            counts['articles'] += 1
            if not article['rows']:
                counts['excluded_articles'] += 1
            for row in article['rows']:
                claim = dict(extract(row[9]), entry_key=row[10], raw=row[9])
                record['claims'].append(claim)
                counts[claim['status']] += 1
                if claim['status'] != 'sanskrit-equivalent':
                    continue
                # One counterpart per sense, shared by its explicit alternate heads.
                counterpart = row[10].rsplit(':v', 1)[0] + ':sanskrit'
                claim['counterpart_key'] = counterpart
                if not row[11]:
                    # The Hindi definition belongs to the Prakrit head, not automatically
                    # to its Sanskrit correspondent. Keep it as explicit context in prose.
                    prose = 'Sanskrit equivalent printed by Sheth for ' + row[2] + '. Prakrit definition: ' + (row[3] or '(cross-reference only)')
                    if claim['desya']:
                        prose += '. Sheth labels the Prakrit word deśya.'
                    out = ['Sk', '', claim['form'], '', claim['native'], '', '', row[7], '', prose, counterpart, '', '', '', '']
                    writer.writerow(out)
                    claim['row'] = out
                    counts['sanskrit_rows'] += 1
                comparison = dict(zip(COLUMNS, [row[10]+':sanskrit-comparison', row[10], counterpart,
                    'related', 'undetermined', 'high', row[7],
                    'Sheth prints Sanskrit equivalent [' + row[9] + '] for ' + row[2] + '; this is a lexical correspondence, not an asserted historical derivation.']))
                compare.writerow(comparison)
                claim['comparison'] = comparison
                counts['comparisons'] += 1
            audit.write(json.dumps(record, ensure_ascii=False)+'\n')
            if any(c['status']=='sanskrit-equivalent' for c in record['claims']):
                eligible += 1
                if len(sample)<20: sample.append(record)
                else:
                    i=rng.randrange(eligible)
                    if i<20: sample[i]=record
    report = {'counts':dict(counts), 'seed':seed, 'historical_ancestry_links':0,
              'policy':'Exact source-local Sanskrit equivalents; no fuzzy CDIAL matching or automatic ancestry',
              'input_sha256':hashlib.sha256((PACKAGE/'audit.jsonl.gz').read_bytes()).hexdigest()}
    (output/'etymology-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (output/'etymology-sample.json').write_text(json.dumps(sample,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))
    return report


def install(output):
    shutil.copyfile(output/FILENAME, HERE.parent/FILENAME)
    COMPARISONS.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(output/'comparisons.csv',COMPARISONS)
    for name in ['etymology-audit.jsonl.gz','etymology-report.json','etymology-sample.json']:
        shutil.copyfile(output/name, PACKAGE/name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=20260915);p.add_argument('--install',action='store_true');a=p.parse_args()
    run(a.output,a.seed)
    if a.install: install(a.output)
