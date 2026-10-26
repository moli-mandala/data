"""Small, source-local validation and immutable-review handoff; no installation."""
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pybtex
import pybtex.database
from segments import Tokenizer

P = Path(__file__).resolve().parent
WORKSPACE = P.parents[5]
SCRATCH = WORKSPACE / 'tmp/fawcett-saora-review'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(name):
    return [json.loads(line) for line in (P / name).read_text().splitlines() if line]


def main():
    units = jsonl('proposal-audit.jsonl')
    pages = jsonl('article-page-census-first-reading-20260927.jsonl')
    rows = list(csv.reader((P / 'proposal.csv').open()))
    html = (SCRATCH / 'google-volume1b.html').read_text(errors='replace')
    order = dict(re.findall(r'\{"pid":"([^"]+)"[^{}]*?"order":(\d+)[^{}]*?\}', html))
    assert all(int(order[f'PA{p}']) == p + 49 for p in range(206, 274))
    by_page = {p: [u for u in units if u['printed_page'] == p] for p in range(206, 273)}
    reviewed = []
    for page in pages:
        p = page['printed_page']
        local = by_page[p]
        reviewed.append(dict(
            printed_page=p, physical_order=int(order[f'PA{p}']),
            source_url=page['source_url'],
            scope=page['scope'],
            status='Whole page visually read; lexical inventory reconciled with first-reading census.',
            historical_census='article-page-census-first-reading-20260927.jsonl',
            source_unit_keys=[u['source_unit_key'] for u in local],
            target_units=sum(bool(u['entry_keys']) for u in local),
            audit_only_units=sum(not u['entry_keys'] for u in local),
            emitted_forms=[form for u in local if u['entry_keys'] for form in u['forms']],
            disposition=('Keyed lexical or documentary evidence above; other text is contextual narrative.'
                         if local else 'No new glossed target form; repeated names, English narrative, and comparison context accounted in historical census.'),
        ))
    (P / 'article-page-census-reviewed-20260927.jsonl').write_text(
        ''.join(json.dumps(p, ensure_ascii=False, sort_keys=True) + '\n' for p in reviewed))

    convert = Tokenizer(str(P / 'proposal-profile.txt'))
    for row in rows:
        nfc = convert(unicodedata.normalize('NFC', row[2]), column='IPA')
        nfd = convert(unicodedata.normalize('NFD', row[2]), column='IPA')
        assert nfc == nfd and '\ufffd' not in nfc, row[10]
    bib = pybtex.database.parse_file(str(P / 'reference-proposal.bib'))
    assert set(bib.entries) == {'fawcett1888saora'}
    entry = bib.entries['fawcett1888saora']
    rendered = pybtex.PybtexEngine().format_from_string(entry.to_string('bibtex'), 'plain', output_backend='markdown')
    for row in rows:
        assert re.fullmatch(r'fawcett1888saora\[[^;\[\]]+\]', row[7]), row[10]
    report = dict(
        status='passed source-local validation; independent final output review and installation pending',
        rows=len(rows), units=len(units),
        sections=dict(Counter(u['section'] for u in units)),
        nfc_nfd_equivalent_rows=len(rows), profile_unknown_graphemes=0,
        formatted_reference=rendered.strip(),
        citation_ids=['fawcett1888saora'], malformed_citations=0,
        physical_boundary=dict(first_page=206, last_page=272, first_order=255, last_order=321,
            next_page=273, next_order=322, unnumbered_internal_gaps=0,
            interpretation='All 67 printed pages occupy consecutive scan positions. Final object catalogue and postscript end on p. 272; p. 273 starts Society proceedings. No internal unnumbered appendix is omitted.'),
        source_context=[
            'The title and first paragraph scope this article to the Eastern Ghats Saoras; other similarly named populations are documentary controls.',
            'The article quotes earlier gazetteers, administrative manuals and census reports for context. These are attributed through Fawcett, not represented as independently consulted primary sources.',
            'The arrow item on p. 208 footnote 5 is Fawcett’s explicit correction of the manual vocabulary, not a new import of that earlier vocabulary.',
            'Kinship informants and the Missionary-alphabet/pronunciation qualification are preserved. No native-script facsimile or additional native-script witness is printed.',
            'Proper names and explicitly glossed terms outside tables are included with source-specific interpretation and attribution qualifications. Incidental repeated mentions are not silently treated as new senses.',
            'Foreign botanical, religious, historical and musical context is accounted in the page census; no English translation is converted into an invented Saora utterance.',
        ],
        boundary_evidence_hashes={n:digest(SCRATCH/n) for n in ['google-volume1b.html','boundary273.html','boundary273.jpg']},
    )
    (P / 'whole-article-validation-20260927.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    names = list(json.loads((P/'proposal-summary.json').read_text())['hashes'])
    names += ['prepare_reviewed.py','prepare_proposal.py','validate_freeze.py',
              'article-original-acquisition-20260927.json','article-page-census-reviewed-20260927.jsonl',
              'whole-article-validation-20260927.json','proposal-summary.json',
              'import_source.py','secondary-reference-scope-20260927.json',
              'root-whole-output-audit-20260927-pass1.json',
              'profile-at-first-independent-audit.txt','post-review-continuity-20260927.json',
              'installation-plan-20260927.json']
    manifest = dict(status='frozen for independent whole-output review; no canonical installation',
        row_count=len(rows), unit_count=len(units),
        audit_schema='One JSON object per source_unit_key, with printed_page, section, forms, gloss, scope, notes, tags and entry_keys. Empty entry_keys are explicit controls/documentary evidence. CSV has the standard 15 source columns; Entry_Key is column 11.',
        hashes={n:digest(P/n) for n in names},
        focused_tests=dict(path='tests/test_fawcett_saora_1888.py', sha256=digest(P.parents[4]/'tests/test_fawcett_saora_1888.py')),
        originals='tmp/fawcett-saora-review/article-pages/p{printed_page}.jpg; selected higher-resolution originals in detail-pages/. Acquisition manifest records original URLs/hashes.',
        deferred=['independent final output audit','canonical source/profile/YAML/bibliography installation','post-install focused checks','full CLDF/database build','browser QA'])
    (P / 'frozen-manifest-20260927.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows), units=len(units), nfc_nfd='pass', reference_format='pass', boundary='67 consecutive original pages', manifest='frozen-manifest-20260927.json')))


if __name__ == '__main__':
    main()
