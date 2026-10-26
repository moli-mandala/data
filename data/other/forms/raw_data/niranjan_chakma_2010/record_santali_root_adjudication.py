"""Record root's original-image decisions, checking the exact reviewed proposal."""
import json
from pathlib import Path
from record_brokskat_first_reading import sha

ROOT = Path(__file__).resolve().parent
EXPECTED = '5f88a09fbe7c8a53d85082a704811325028ba36cde66939e011a739ec3adedfa'
# Explicit page/row/column decisions made after viewing all eight original
# half-page images. This is not a rule accepting arbitrary future proposals.
DECISIONS = [
    (96, 5, 'santali', 'আচ্‌কা'), (96, 5, 'chakma', 'আত্‌কা'),
    (96, 6, 'santali', 'আঙ্‌রা'),
    (96, 9, 'santali', 'উখ্‌রান'), (96, 9, 'chakma', 'উগ্‌রান'),
    (96, 11, 'santali', 'ওক্‌তে'), (96, 11, 'chakma', 'অক্‌ত'),
    (96, 18, 'santali', 'কুট্‌কুট'), (96, 18, 'chakma', 'কুদ্‌কুদি'),
    (96, 19, 'santali', 'কুন্‌ডেল'), (96, 19, 'chakma', 'কন্‌ডোল'),
    (97, 8, 'santali', 'গিরী'),
    (97, 9, 'santali', 'গিন্‌তি'), (97, 9, 'chakma', 'গন্‌তি'),
    (97, 12, 'santali', 'গোর্‌জি'), (97, 12, 'chakma', 'গোর্‌জি'),
    (97, 17, 'santali', 'জ, জো'), (97, 18, 'santali', 'জাম্‌বি'),
    (97, 23, 'chakma', 'ডাম্‌রা, ডামারা'),
    (98, 11, 'santali', 'দাহ্‌গি'), (98, 11, 'chakma', 'দাহ্‌ঘি'),
    (98, 14, 'santali', 'দিগ্‌ধা'),
    (98, 17, 'santali', 'নগ্‌দা'), (98, 17, 'chakma', 'নগ্‌দা'),
    (98, 19, 'santali', 'পার্‌সি'),
    (98, 20, 'santali', 'ফার্‌ফারিয়া'), (98, 20, 'chakma', 'ফোর্‌য্যা'),
    (98, 23, 'santali', 'বাক্‌লাক'),
    (99, 5, 'santali', 'রর্'), (99, 5, 'chakma', 'রঅ'),
    (99, 6, 'santali', 'রান্‌ডি'), (99, 7, 'santali', 'সল্‌হা'),
]
SPECIAL = {
    (97, 8, 'santali'): 'Visible long-i curl on final র differs from adjacent Chakma গিরা.',
    (97, 17, 'santali'): 'First জ has no u-loop; adjacent Chakma জু has a clear contrasting loop. Preserve both Santali alternatives.',
    (99, 5, 'chakma'): 'Final independent অ shape differs from ব; preserve the literal source form.',
}


def main():
    source = ROOT / 'santali-reviewed-reading-proposal.jsonl'
    assert sha(source) == EXPECTED
    rows = [json.loads(s) for s in source.read_text().splitlines()]
    witnesses = {(r['pdf_page'], r['physical_row'], w['witness_key'].rsplit(':', 1)[1]): w
                 for r in rows for w in r['witnesses']}
    assert len(DECISIONS) == len({d[:3] for d in DECISIONS}) == 32
    decisions = []
    for page, row, field, accepted in DECISIONS:
        key = (page, row, field)
        w = witnesses[key]
        assert accepted == w['reviewed_reading_proposal'] != w['raw_owner_first_reading']
        h_hold = page == 98 and row == 11
        decisions.append({'witness_key': w['witness_key'], 'old': w['raw_owner_first_reading'],
                          'accepted_reading': None if h_hold else accepted,
                          'held_proposal': accepted if h_hold else None, 'original_pdf_page': page,
                          'image_evidence': f'santali-detail/p{page}-1.jpg and p{page}-2.jpg',
                          'basis': 'Earlier approval withdrawn: ordinary হ has a lower stroke too. Font-control adjudication required before adding hasanta.' if h_hold else SPECIAL.get(key, 'Visible lower diagonal is hasanta; retain disjoining only where source consonants are separate, and preserve other letters.')})
    target = ROOT / 'santali-root-glyph-adjudication-20260927.json'
    target.write_text(json.dumps({
        'date': '2026-09-27', 'reviewer': 'root',
        'status': '30 correction proposals accepted at transcription stage; two h-class approvals withdrawn after font-control challenge; residual readings remain',
        'reviewed_proposal_sha256': EXPECTED,
        'source_pdf_sha256': 'f022a2a448445cdc9840891478bd5864e2ded1ae98985332b51f0938c075842d',
        'method': 'All eight original 3x half-page images inspected; exact proposed changes individually matched above.',
        'decisions': decisions,
        'remaining_evidence': [
            'All remaining_class_exceptions in independent-santali-hasanta-class-addendum-20260927.json remain held.',
            'All original owner transcription notes not explicitly resolved by these decisions remain available; no blanket certification.',
            'PDF98 row19 Chakma final cluster remains unresolved in the base review.',
            'PDF99 row16 proposed final hasanta is unconfirmed: independent reviewer identifies a similar stroke on ordinary initial হ in adjacent row17 হাগা. Entire h-class requires font controls; no automatic correction.'
        ],
        'accepted_for_installation': False, 'canonical_rows_changed': 0,
        'pending': ['remaining glyph holds and additional row16 proposal', 'whole-book integration and remaining ingestion gates'],
    }, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'decisions': len(decisions), 'accepted': sum(d['accepted_reading'] is not None for d in decisions), 'held': sum(d['accepted_reading'] is None for d in decisions), 'output': target.name, 'sha256': sha(target)}))


if __name__ == '__main__':
    main()
