"""Regenerate the reviewed Mahali source CSV and audit; never build the database."""
import argparse
import csv
import json
from pathlib import Path

from prepare_analysis import prepare as prepare_analysis
from prepare_draft import prepare as prepare_draft

ROOT = Path(__file__).resolve().parent
INSTALLED = ROOT.parents[1] / '20260925-cfel-mahali.csv'


def run(install=False):
    analyses = prepare_analysis()
    assert len(analyses) == 2451
    assert all(row['native'] for row in analyses), 'native review incomplete'
    (ROOT/'analysis-proposals.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in analyses))
    rows, audit = prepare_draft()
    assert len(rows) == len(audit) == 2451
    assert all(row[0] == 'Mahali' and row[4] and len(row) == 15 for row in rows)
    with (ROOT/'review-draft.csv').open('w', newline='') as stream:
        csv.writer(stream).writerows(rows)
    (ROOT/'draft-audit.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in audit))

    # Both independently inspected samples are frozen. If a parser edit changes
    # an audited output, review the PDF again and update the corresponding audit.
    from audit_draft import verify
    verify()
    if install:
        with INSTALLED.open('w', newline='') as stream:
            csv.writer(stream).writerows(rows)
    return len(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true', help='write source CSV only')
    args = parser.parse_args()
    print(f'{run(args.install)} reviewed rows; '
          f'{"source CSV installed" if args.install else "review artifacts regenerated"}; '
          'no database build')
