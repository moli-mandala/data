"""Record root's original-page review of the immutable first whole proposal."""
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
EXPECTED = '909b87f5cf16109cbe64c610d6a3c0dfc6b631d7adea3214e1048dfd122a26c8'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(P/'frozen-manifest.json') == EXPECTED
    manifest = json.loads((P/'frozen-manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(P/name) == digest, name
    selection_path = P/'independent-sample-20260927.json'
    sample = json.loads(selection_path.read_text())
    assert sample['frozen_manifest_sha256'] == EXPECTED
    failures = {
        'peterson2024turi:p298:example:47': [
            dict(type='grammatical-metadata', reason='All five counted numerals lack num tags in actual child rows, although the complete example is explicitly a counting sequence and p268 classifies these forms as numerals. Whole expression and aligned lexical forms match.')],
        'peterson2024turi:p299:example:56': [
            dict(type='grammatical-metadata', reason='Reused miɑʔn one retains the same omitted numeral classification. Review context-dependent numeral/article function and reuse scope across all witnesses; do not merely patch the sampled citation.')],
        'peterson2024turi:p276:example:29': [
            dict(type='grammatical-metadata', reason='Historical miat’ one and baria two lack numeral classification.'),
            dict(type='gloss-normalization', reason='tahi=ken=a=kin has Gloss cop rather than expanded readable copula; source abbreviation COP is explicitly defined on p290.'),
            dict(type='uncertainty-scope', reason='The [h] editorial uncertainty applies to man and the whole expression, but uncertain is propagated to all five child rows, including unaffected one, two, child and copula.')],
        'peterson2024turi:p290:inline:hiatus': [
            dict(type='dialect-scope', reason='The general abbreviation entry -y- has no locality on p290, but is tagged only Sarangarh. The same historical hiatus notation occurs in explicitly Jashpur material in p272 footnote9 as well as Sarangarh example25 on p275. Reconcile actual attestation scope and supporting citations rather than assigning the general definition exclusively to Sarangarh.'),
            dict(type='grammatical-metadata', reason='Source identifies a hiatus-breaking element; epenthetic analysis should be preserved rather than relying only on generic affix classification.')],
    }
    findings = []
    for unit in sample['selected']:
        errors = failures.get(unit['source_unit_key'], [])
        findings.append(dict(source_unit_key=unit['source_unit_key'],
            printed_page=unit['printed_page'], pdf_page=unit['pdf_page'],
            stratum=unit['stratum'], actual_csv_rows=unit['actual_csv_rows'],
            literal_form_alignment_and_translation='pass',
            verdict='fail' if errors else 'pass', material_errors=errors,
            evidence=f"Original publisher PDF printed p.{unit['printed_page']}, inspected in 2x render; exact frozen output rows compared."))
    assert len(findings) == 20
    report = dict(status='failed; source-wide metadata repair and fresh seeded audit required',
        reviewer='root', date='2026-09-27', seed=sample['seed'],
        frozen_manifest_sha256=EXPECTED, sample_sha256=sha(selection_path),
        source_pdf_sha256=manifest['original_sha256'],
        source_units=20, actual_csv_rows=sum(len(u['actual_csv_rows']) for u in sample['selected']),
        failed_source_units=len(failures), passed_source_units=20-len(failures),
        checks=['literal form', 'alignment and gloss', 'tag scope', 'citation', 'lect assignment', 'variant claim', 'typed uncertainty'],
        per_unit=findings,
        remediation='Preserve this freeze and report. Repair each identified metadata class across the complete proposal, validate regeneration/parser/profile/registries, freeze a new revision and select fresh units excluding this and the old Appendix1 sample.',
        scope_limit='A source-output audit, not full-build/compiled identity/graph/browser validation. Canonical source files not approved for installation by this report.')
    (P/'root-independent-output-audit-20260927-pass1.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f"Reviewed20units/{report['actual_csv_rows']}rows:16pass,4fail; no installation approval.")


if __name__ == '__main__':
    main()
