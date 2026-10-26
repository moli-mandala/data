"""Read-only canonical reconciliation; writes only whole-recovery evidence."""
import csv
import hashlib
import json
from pathlib import Path

from import_koya import AUDIT, HERE, INSTALLED, MANIFEST, build


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    forms, audit, manifest = build()
    with INSTALLED.open(newline='') as stream:
        installed = list(csv.reader(stream))
    with AUDIT.open(newline='') as stream:
        installed_audit = list(csv.DictReader(stream))
    expected_audit = [{key: str(value) for key, value in row.items()} for row in audit]
    checks = {
        'canonical_rows_equal_regeneration': installed == forms,
        'canonical_audit_equal_regeneration': installed_audit == expected_audit,
        'manifest_equal_regeneration': json.loads(MANIFEST.read_text()) == manifest,
        'all_existing_keys_unique': len({row[10] for row in installed}) == len(installed),
        'expected_counts': len(forms) == 1438 and len(audit) == 1890,
        'printed_wordlist_pages_covered': {int(row['PDF_Page']) for row in audit} == set(range(82, 124)),
    }
    result = {
        'status': 'existing wordlist regeneration reconciled' if all(checks.values()) else 'reconciliation failed',
        'checks': checks,
        'counts': manifest['counts'],
        'canonical_sha256': digest(INSTALLED),
        'audit_sha256': digest(AUDIT),
        'manual_ledger_sha256': digest(HERE / 'manual_review_data.py'),
        'source_sha256': manifest['source_pdf_sha256'],
        'scope': 'Appendix E printed wordlist cells PDF82–123. PDF80–81 alphabet and introductions require separate whole-scope review.',
        'limits': 'Reuses historical source-image ledger; not a fresh image audit or full-source installation approval. No canonical files changed.',
    }
    (HERE / 'whole-source-recovery/existing-wordlist-reconciliation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))
    assert all(checks.values()), checks
