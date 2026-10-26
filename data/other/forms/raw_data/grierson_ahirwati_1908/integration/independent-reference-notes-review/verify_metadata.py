"""Bounded real SourceMeta/formatter and strict per-row citation check."""
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent.parent
DATA = PACKAGE.parents[4]
sys.path.insert(0, str(DATA))
import source_meta
from pybtex import PybtexEngine
from pybtex.database import parse_file

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    assembly = PACKAGE / 'integration/review-assembly-r2'
    csv_path = assembly / 'combined-review-candidate.csv'
    yaml_path = assembly / 'combined-review-candidate.yaml'
    bib_path = PACKAGE / 'metadata-profile-proposal/references-proposed.bib'
    pinned = {str(p): sha(p) for p in [csv_path, yaml_path, bib_path, assembly / 'status.json',
        PACKAGE / 'scope/source-and-errata-review.json', PACKAGE / 'scope/major-addenda-applicability-review-20260927.json']}
    rows = list(csv.reader(csv_path.open(newline='')))
    assert len(rows) == 653 and all(len(r) == 15 for r in rows)
    meta = source_meta.SourceMeta([*source_meta.yaml_files(), yaml_path])
    reference = meta.source('grierson1908ahirwati')['reference']
    assert reference == {'editor': 'OpenAI Codex', 'ocr': False, 'etymology_provenance': 'source'}
    assert meta.file_flag(csv_path, 'identity', 'dedupe_by_entry_key') is True
    routes = []
    for language in sorted({r[0] for r in rows}):
        profile, convert = meta.transcription('grierson1908ahirwati', csv_path, language)
        rule = meta.transcription_rule('grierson1908ahirwati', csv_path, language)
        assert profile == 'grierson-ahirwati-1908-literal-preliminary' and convert is True
        assert rule['input'] == 'form' and rule['preserve_hyphens'] is True
        routes.append({'language': language, 'profile': profile, 'convert': convert})
    staged = parse_file(str(bib_path))
    current = parse_file(str(DATA / 'cldf/sources.bib'))
    assert set(staged.entries) == {'grierson1908ahirwati'}
    assert not set(staged.entries) & set(current.entries)
    bib_ids = set(staged.entries) | set(current.entries)
    formatted = PybtexEngine().format_from_string(staged.to_string('bibtex'), 'plain', output_backend='markdown')
    assert all(t in formatted for t in ['1908', 'Grierson', 'Calcutta', 'Rajasthani', 'Gujarati'])
    citations, issues = [], []
    count = Counter()
    allowed_pages = {49, 50, 51, 149, 151, *range(233, 240)}
    for row in rows:
        assert row[7], row[10]
        for token in row[7].split(';'):
            token = token.strip()
            match = re.fullmatch(r'([A-Za-z0-9_.:-]+)\[([^\[\];]+)\]', token)
            if not match:
                issues.append({'key': row[10], 'citation': token, 'issue': 'Malformed or absent bracket locator'})
                continue
            key, locator = match.groups()
            if key not in bib_ids:
                issues.append({'key': row[10], 'citation': token, 'issue': 'Unknown bibliography key'})
            page_locator = re.sub(r'\bspecimen\s+\d+\b', '', locator, flags=re.I)
            numbers = [int(n) for n in re.findall(r'\d+', page_locator)]
            if not numbers or not set(numbers) <= allowed_pages:
                issues.append({'key': row[10], 'citation': token, 'issue': 'Outside selected original page scope'})
            count[key] += 1
            citations.append({'entry_key': row[10], 'citation_key': key, 'locator': locator})
    assert not issues, issues
    assert set(count) == {'grierson1908ahirwati'}
    (HERE / 'all-citation-decisions.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in citations))
    with (HERE / 'all653-notes-etymology-review.tsv').open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['key', 'form', 'gloss', 'notes', 'source', 'etymology', 'tags'])
        for r in rows:
            writer.writerow([r[10], r[2], r[3], r[6], r[7], r[9], r[14]])
    notes = Counter(r[6] for r in rows if r[6])
    screening = []
    for r in rows:
        if re.search(r'\b(?:do not|must|pipeline|parser|CSV|Entry_Key|emission|reconstruct|raw.head|raw.gloss|metadata|fixture|protected|uncertain tag)\b', r[6], re.I):
            screening.append({'key': r[10], 'notes': r[6]})
    entry = staged.entries['grierson1908ahirwati']
    report = {'status': 'PASS real APIs and strict citation resolution; complete Notes semantic review pending',
        'input_sha256': pinned, 'rows': len(rows), 'citation_tokens': len(citations), 'citation_key_counts': dict(count),
        'citation_issues': issues, 'reference_flags': reference, 'transcription_routes': routes,
        'bibliography_formatted': formatted, 'bibliography_fields': dict(entry.fields),
        'bibliography_authors': [str(p) for p in entry.persons['author']],
        'notes_nonblank_rows': sum(notes.values()), 'unique_notes': len(notes),
        'repeated_notes': [{'notes': note, 'count': n} for note, n in notes.most_common() if n > 1],
        'procedural_screen_candidates': screening,
        'pending': ['Actual importer command intentionally absent until guarded installer exists.',
                    'Scope planning provenance files are historical; current accepted assembly/audit should supplement final canonical provenance.'],
        'OCR_rationale': 'Manual original-page transcription; existing OCR derivatives used as locator aids only, not accepted installed readings.',
        'canonical_changes': False}
    for p, h in pinned.items(): assert sha(p) == h, ('Input drift', p)
    (HERE / 'metadata-and-citation-review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'rows': 653, 'citation_tokens': len(citations), 'citation_keys': dict(count), 'notes_nonblank': sum(notes.values()), 'unique_notes': len(notes), 'screen_candidates': len(screening)}))

if __name__ == '__main__': main()
