"""Record 14 independently inspected thematic corrections; leave CSV untouched."""
import json
from pathlib import Path
from record_brokskat_first_reading import sha
from thematic_validate_whole_review import validate

ROOT = Path(__file__).resolve().parent
ACCEPTED = {
    'তেক্নাফ': 'তেক্‌নাফ', 'ঝর্ঝরি': 'ঝর্‌ঝরি', 'তাম্ঝাং': 'তাম্‌ঝাং',
    'নাক্স গাচ': 'নাক্‌স গাচ', 'ভিল্ভিলি কানা': 'ভিল্‌ভিলি কানা',
    'বাক্সু': 'বাক্‌সু', 'দৌক্দোক্যা রাঙা': 'দৌক্‌দোক্যা রাঙা',
    'ইস্কুল': 'ইস্‌কুল', 'আবির্তি': 'আবির্‌তি', 'জিল্গাত': 'জিল্‌গাত',
    'ভাযতত্ত্ব': 'ভাষতত্ত্ব', 'আল্গা জিল': 'আল্‌গা জিল',
    'য়েরো': 'য়েরা', 'নাক্স': 'নাক্‌স',
}


def main():
    validation = validate()
    report_path = ROOT / 'thematic-whole-glyph-review-proposal.json'
    report = json.loads(report_path.read_text())
    assert len(report['corrections']) == len(ACCEPTED) == 14
    assert {c['before']: c['after'] for c in report['corrections']} == ACCEPTED
    decisions = []
    for i, c in enumerate(report['corrections'], 1):
        decisions.append({'source_entry_key': c['source_entry_key'],
            'old': c['before'], 'accepted_reading': c['after'],
            'class': c['class'],
            'original_image': f'tmp/pdfs/chakma-discovery/thematic-findings/corrections-{i:02d}.jpg',
            'root_basis': ('Original shows full consonant plus detached lower hasanta followed by a separate consonant; preserve disjoining with U+200C.'
                           if c['class'] == 'visible-hasanta-disjoining' else
                           'Original divided ষ body is distinct from য; no extra vowel added.' if c['before'] == 'ভাযতত্ত্ব' else
                           'Final রা has no preposed e-sign; রো is unsupported.'),
            'other_existing_uncertainties_preserved': True})
    target = ROOT / 'thematic-root-glyph-adjudication-20260927.json'
    target.write_text(json.dumps({
        'date': '2026-09-27', 'reviewer': 'root',
        'status': '14 correction proposals accepted at transcription stage; whole-source installation pending',
        'source_pdf_sha256': report['source_pdf_sha256'],
        'canonical_sha256_at_review': report['canonical_sha256'],
        'report_sha256': sha(report_path),
        'method': 'Root viewed all14 fresh6x original row strips including source Bengali/English controls. Crops use stable key y from the original4x OCR frame; no text reconstructed from proposal strings.',
        'decisions': decisions,
        'remaining_candidates': report['uncertain_candidates'],
        'withheld_cells': validation['withheld_source_cells'],
        'policy': ['No h-class correction accepted from lower diagonal alone.',
                   'Existing transcription/gloss uncertainties are not cleared by a disjoining correction.',
                   'Eye-ball unsplit group recovery requires explicit whole-source integration; damaged Warm/Worm remains audited.'],
        'accepted_for_installation': False, 'canonical_rows_changed': 0,
    }, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'accepted_corrections': len(decisions), 'remaining_candidates': len(report['uncertain_candidates']),
                      'output': target.name, 'sha256': sha(target), 'canonical_rows_changed': 0}))


if __name__ == '__main__':
    main()
