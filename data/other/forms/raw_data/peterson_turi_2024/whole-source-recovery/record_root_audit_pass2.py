"""Record the original-page independent review of frozen revision 2."""
import csv
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent / 'revision2'
EXPECTED = '1f6f78c88e421cd10d94710fa7fd71bb2cc3ca71a510924c9e6e98378b40c986'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(P / 'frozen-manifest.json') == EXPECTED
    manifest = json.loads((P / 'frozen-manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(P / name) == digest, name
    sample_path = P / 'independent-sample-20260927.json'
    sample = json.loads(sample_path.read_text())
    assert sample['frozen_manifest_sha256'] == EXPECTED
    errors = {
        'peterson2024turi:p275:example:25': [dict(
            type='expression-scope',
            reason='Original has explicit ellipses; Notes correctly identify an excerpt, but the whole row is incorrectly tagged sentential. Retain multiword-expression without asserting a complete sentence.')],
        'peterson2024turi:p298:example:45': [dict(
            type='grammatical-metadata',
            reason='Aligned pɑɽh=ɑ=bu retains encoded ipl.incl uncertainty and Notes explicitly identify first-person plural inclusive. Its Tags omit first-person, pl and 1pl. Preserve the raw typography and typed uncertainty while structuring the explicitly documented analysis.')],
    }
    findings = []
    for unit in sample['selected']:
        failure = errors.get(unit['source_unit_key'], [])
        findings.append(dict(source_unit_key=unit['source_unit_key'],
            printed_page=unit['printed_page'], pdf_page=unit['pdf_page'],
            stratum=unit['stratum'], actual_csv_rows=unit['actual_csv_rows'],
            literal_form_alignment_and_translation='pass',
            verdict='fail' if failure else 'pass', material_errors=failure,
            evidence='Original publisher PDF viewed at 2x; all 13 sampled pages inspected against actual frozen output rows.'))
    baseline = list(csv.reader((P / 'proposal/20260921-peterson-turi.csv').open()))
    numerals = [r for r in baseline if any(f':item{n}:' in r[10] for n in range(179, 184))]
    assert len(numerals) == 6 and all('num' not in r[14].split() for r in numerals)
    report = dict(status='failed; source-wide metadata repair and fresh seeded audit required',
        reviewer='root', date='2026-09-27', seed=sample['seed'],
        frozen_manifest_sha256=EXPECTED, sample_sha256=sha(sample_path),
        source_pdf_sha256=manifest['original_sha256'], source_units=20,
        actual_csv_rows=sum(len(u['actual_csv_rows']) for u in sample['selected']),
        passed_source_units=18, failed_source_units=2,
        checks=['literal form', 'alignment and gloss', 'tag scope', 'citation', 'lect assignment', 'variant claim', 'typed uncertainty'],
        per_unit=findings,
        targeted_outside_sample=[dict(type='incomplete-class-repair', printed_page=295,
            reason='Legacy Appendix1 numerals179–183 (six rows including the two forms for three) still lack num tags. Original and frozen CSV checked. Apply systematic metadata review to the complete proposal, including the legacy246 rows.',
            actual_csv_rows=numerals)],
        remediation='Preserve revision2 and this report. Repair complete error classes including legacy metadata, preserve stable keys/literals/IPA, validate, freeze revision3 and select a fresh disjoint sample.',
        scope_limit='Source-output audit only. Does not approve installation or establish full-build, compiled graph/identity or browser checks.')
    (P / 'root-independent-output-audit-20260927.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(f"Reviewed20units/{report['actual_csv_rows']}rows:18pass,2fail; additional legacy numeral metadata failure confirmed.")


if __name__ == '__main__':
    main()
