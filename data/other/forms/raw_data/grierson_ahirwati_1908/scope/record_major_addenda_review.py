"""Record bounded original-page applicability review; never overwrite prior evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
HERE = Path(__file__).resolve().parent
ASSETS = ROOT / 'tmp/jambu-next-source'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    pages = {
        2: ('unnumbered title', 'Volume I, Part I, Introductory; Grierson; Calcutta Government of India Central Publication Branch, 1927.'),
        9: ('iii', 'Summary contents: Supplement I Addenda et Corrigenda Majora begins 201; Supplement II issued loose; language-name index 425.'),
        22: ('xvi', 'Detailed contents repeats Supplement I 201, separate loose Supplement II and index 425.'),
        232: ('201', 'Complete Supplement I section list, transcribed below; no Volume IX section.'),
        234: ('203', 'First substantive section is Volume III Part II Simi or Sema, confirming section-list start.'),
        457: ('426', 'Alphabetical primary index includes Ahirwal and Ahirwati or Hirwati, alongside distinct Ahiri aliases.'),
    }
    images = []
    for physical, (printed, finding) in pages.items():
        path = ASSETS / f'ahirwati-sup1-{physical:03d}.jpg'
        images.append({'physical_image': physical, 'printed_locator': printed,
                       'path': str(path.relative_to(ROOT)), 'sha256': digest(path),
                       'bytes': path.stat().st_size,
                       'url': f'https://dsal.uchicago.edu/books/lsi/images/lsi-v1-1-{physical:03d}.jpg',
                       'method': 'direct visual original-page review; no OCR', 'finding': finding})
    sections = [
        ('III.2', 'Simi or Sema', 203), ('III.2', 'Chang or Mojung', 210),
        ('III.2', 'Tangkhul', 216), ('V.1', 'Bengali', 221),
        ('V.2', 'Oriya', 224), ('VI', 'Awadhi', 231), ('VI', 'Chhattisgarhi', 235),
        ('VII', 'Marathi', 245), ('VIII.2', 'Kafir Languages', 247),
        ('VIII.2', 'Tirahi', 265), ('VIII.2', 'Gilgiti Shina', 328),
        ('X', 'Ormuri and Parachi', 385),
    ]
    report = {
        'status': 'applicability_closed_by_primary_complete_section_list_and_language_index',
        'date': '2026-09-27', 'reviewer': 'OpenAI Codex / norton_finish',
        'work': {'author': 'George Abraham Grierson', 'title': 'Linguistic Survey of India. Volume I, Part I. Introductory',
                 'year': 1927, 'publisher': 'Government of India Central Publication Branch', 'place': 'Calcutta',
                 'supplement': 'Supplement I. Addenda Majora', 'printed_start': 201},
        'original_pages': images,
        'complete_major_addenda_sections': [{'volume_part': v, 'label': l, 'start': p} for v, l, p in sections],
        'index_records': [
            {'label': 'Ahirwal', 'volume_part': 'IX.2', 'pages': [49], 'classification': 'Another name for Ahirwati (759)'},
            {'label': 'Ahirwati or Hirwati', 'classified_number': 759, 'survey_speakers': 448945,
             'volume_part': 'IX.2', 'pages': [3, 43, 49, 233],
             'classification': 'A sub-dialect of North-Eastern or (753), Rajasthani (712). Spoken in the South-East Punjab.'},
            {'label': 'Ahiri', 'volume_part': 'IX.1', 'pages': [53, 240, 258, 263, 305], 'disposition': 'Distinct standard Malvi alias; not Ahirwati.'},
            {'label': 'Ahiri or Ayari', 'volume_part': 'IX.3', 'pages': [5, 63], 'disposition': 'Distinct Bhili dialect in Cutch; not Ahirwati.'},
        ],
        'applicable_changes': [],
        'conclusion': 'No Ahirwati/Ahirwal/Hirwati or IX.2 Addenda Majora section is present in the complete original section list. The primary language-name index points to the original IX.2 scope, not to Supplement I. No extra correction or target specimen is identified by this applicability check.',
        'limits': ['This closes correction-section applicability, not a claim that all 188 supplement pages contain no incidental comparative word.',
                   'No global physical-to-printed offset inferred: each inspected footer/title locator is recorded individually.',
                   'Earlier empty Nordic search is unavailable coverage (Volume I absent), not supporting negative evidence.',
                   'Earlier verified-scope-manifest remains immutable and historically records this gate as open; this addendum supplies closure.'],
        'prior_scope_manifest': {'path': 'verified-scope-manifest.json', 'sha256': digest(HERE / 'verified-scope-manifest.json')},
        'script_sha256': digest(Path(__file__)),
    }
    path = HERE / 'major-addenda-applicability-review-20260927.json'
    data = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode()
    if path.exists() and path.read_bytes() != data:
        raise SystemExit('Refusing to overwrite historical review')
    path.write_bytes(data)
    print(json.dumps({'path': str(path.relative_to(ROOT)), 'sha256': digest(path), 'original_pages': len(images), 'sections': len(sections), 'applicable_changes': 0}))

if __name__ == '__main__':
    main()
