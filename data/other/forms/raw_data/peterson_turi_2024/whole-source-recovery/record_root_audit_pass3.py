"""Record independent original/output review of immutable Turi revision3."""
import csv
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent / 'revision3'
EXPECTED = 'd8c2dad6a73ee1c977fa06d47707c0ee3b8e51b878fb9819d39e1b6e9ed3fe8f'


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
    rows = {r[10]: r for f in (P/'proposal').glob('*.csv') for r in csv.reader(f.open())}
    failures = {'peterson2024turi:p270:inline:to-monkeys': [dict(
        type='grammatical-metadata', reason='Original p270 explicitly prints monkey=PL=OBL. Actual bandra=kun=kɛ row and its Notes preserve this analysis, but Tags lack pl and obl. Preserve its distinction from numbered example4 OBJ.') ]}
    findings = []
    for unit in sample['selected']:
        errors = failures.get(unit['source_unit_key'], [])
        findings.append(dict(source_unit_key=unit['source_unit_key'],
            printed_page=unit['printed_page'], pdf_page=unit['pdf_page'],
            actual_csv_rows=unit['actual_csv_rows'], stratum=unit['stratum'],
            literal_alignment_translation_and_source_scope='pass',
            verdict='fail' if errors else 'pass', material_errors=errors,
            evidence='Original publisher PDF at 2x compared with actual frozen rows; unchanged original-page evidence reused where already inspected.'))
    targeted = []
    for key, kind, reason in [
        ('peterson2024turi:p270:inline:of-children', 'grammatical-metadata',
         'Original explicitly child=PL=GEN; Notes preserve it but pl/gen tags are absent.'),
        ('peterson2024turi:p271:inline:finite-raised', 'grammatical-scope',
         'The ə finite-marker vowel is tagged third-person and sg by parsing the words describing the following =ɛ index. Those features belong to the following index, not to the vowel.'),
        ('peterson2024turi:p273:inline:negative-surface', 'grammatical-scope',
         'The negative allomorph kə receives third-person and sg from the conditioning description before third-person singular animate marker. Do not assign contextual index features to the negative marker.')]:
        assert key in rows, key
        targeted.append(dict(source_unit_key=key,type=kind,reason=reason,actual_csv_row=rows[key]))
    report = dict(status='failed; complete inline metadata review required',
        date='2026-09-27', reviewer='root', seed=sample['seed'],
        frozen_manifest_sha256=EXPECTED, sample_sha256=sha(selection),
        source_pdf_sha256=manifest['original_sha256'], source_units=20,
        actual_csv_rows=sum(len(u['actual_csv_rows']) for u in sample['selected']),
        passed_source_units=19, failed_source_units=1, per_unit=findings,
        targeted_outside_sample=targeted,
        accepted_typed_exception='Appendix1 item58 preserves redundant nasal mark from the native source text layer in Original/Phonemic with a typed transcription reason and display-only collapse; this documented exception is not a newly discovered parser error.',
        remediation='Review all inline metadata against original scope; move explicit analyses out of unparsed Notes into structured input. Do not infer grammatical tags by scanning conditioning or comparison prose. Cover omitted features and false positives with focused regressions, then freeze a new revision and use a fresh disjoint sample.',
        scope_limit='Source-output audit only; canonical installation not approved. Full compiled identity/graph/build and browser checks remain deferred.')
    (P/'root-independent-output-audit-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f"Reviewed20units/{report['actual_csv_rows']}rows:19pass,1fail;3 additional targeted metadata failures.")


if __name__ == '__main__':
    main()
