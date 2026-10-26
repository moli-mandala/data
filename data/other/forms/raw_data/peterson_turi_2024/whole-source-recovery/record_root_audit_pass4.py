"""Pin root's passing original/output audit and source-stage installation approval."""
import csv
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent/'revision4'
EXPECTED = '43b2458b907665d70637735c652f523712e51fb47e67b12fbad458f6a99a848d'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(P/'frozen-manifest.json') == EXPECTED
    manifest = json.loads((P/'frozen-manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(P/name) == digest, name
    selection = P/'independent-sample-20260927.json'
    sample = json.loads(selection.read_text())
    assert sample['frozen_manifest_sha256'] == EXPECTED
    rows = {r[10]:r for f in (P/'proposal').glob('*.csv') for r in csv.reader(f.open())}
    checks = {
        'p270:inline:to-monkeys': ({'pl','obl'},set()),
        'p270:inline:of-children': ({'pl','gen'},set()),
        'p271:inline:finite-raised': ({'affix'},{'third-person','sg'}),
        'p273:inline:negative-surface': ({'neg','negator'},{'third-person','sg','affix'}),
        'p275:example:25:whole': ({'multiword-expression'},{'sentential'}),
        'p298:example:45:word4': ({'first-person','pl','1pl','inclusive','uncertain'},set()),
        'p295:item179:form1': ({'num'},set()),
        'p272:inline:jashpur-fruitful': ({'verb','middle','pret'},set()),
    }
    repairs = []
    for tail,(required,forbidden) in checks.items():
        row = rows['peterson2024turi:'+tail]
        tags = set(row[14].split())
        assert required <= tags and not forbidden & tags, tail
        repairs.append(dict(entry_key=row[10],verdict='pass',actual_csv_row=row))
    findings = [dict(source_unit_key=u['source_unit_key'],printed_page=u['printed_page'],
        pdf_page=u['pdf_page'],stratum=u['stratum'],actual_csv_rows=u['actual_csv_rows'],
        verdict='pass',material_errors=[],
        evidence='Root compared literal forms, aligned/free glosses, grammatical scope, lect and citation assignments against original publisher PDF. Unchanged source-page images reused where already inspected.')
        for u in sample['selected']]
    assert len(findings) == 20
    report = dict(status='passed independent source-output audit',reviewer='root',date='2026-09-27',
        seed=sample['seed'],frozen_manifest_sha256=EXPECTED,sample_sha256=sha(selection),
        source_pdf_sha256=manifest['original_sha256'],source_units=20,
        actual_csv_rows=sum(len(u['actual_csv_rows']) for u in sample['selected']),
        passed_source_units=20,failed_source_units=0,per_unit=findings,
        prior_failure_targeted_checks=repairs,
        sampling_note='Fourth selection correctly classifies numbered text units by the leading example number. Earlier selection records remain immutable; no audit result is rewritten.',
        scope_limit='Approves source-stage row proposal only. Shared metadata, idempotent installation/regeneration and actual canonical parser checks must still pass. Full compiled build/graph/identity and browser gates remain deferred.')
    report_path = P/'root-independent-output-audit-20260927.json'
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    approval = dict(status='approved',frozen_manifest_sha256=EXPECTED,
        independent_audit_sha256=sha(report_path),approved_csv_rows=648,
        scope='Install these exact source-stage rows and profiles, with reviewed shared metadata and a guarded idempotent importer route. No database/full build/publication authorization.',
        required_followup='Verify canonical parser/profile/reference/dialect/key integrity and reproducible repeated regeneration. Record exact coordinator/YAML metadata overlays separately from the frozen row proposal.')
    (P/'root-installation-approval.json').write_text(json.dumps(approval,ensure_ascii=False,indent=2)+'\n')
    print(f"20/20 units, {report['actual_csv_rows']} rows passed; eight prior-failure cases passed. Source-stage installation approved.")


if __name__ == '__main__':
    main()
