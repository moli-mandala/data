"""Record root's completed original-page audit, guarding the reviewed revision."""
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
R = P / 'whole-source-recovery'
F = R / 'literal-integration-metadata-r2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path = F / 'review-manifest.json'
    expected = '1d220e49a1c33e752944825ef8767f29b746bd39937055e37817e7f710d46a6c'
    assert sha(manifest_path) == expected
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest['outputs_sha256'].items():
        assert sha(F / name) == digest, name
    for name, digest in manifest['scripts_sha256'].items():
        assert sha(R / name) == digest, name
    selection_path = F / 'root-audit-selection-20260927.json'
    selection = json.loads(selection_path.read_text())
    assert selection['manifest_sha256'] == expected
    pages = [95,84,97,99,110,100,95,92,84,83,88,98,34,35,29,31,29,33,12,11]
    observations = [
        'Dental mark and single medial length dot confirmed in 6x detail.',
        'Handwritten o and single medial dot confirmed in 6x detail.',
        'Wedge vowel, colon, and epsilon confirmed in 6x detail.',
        'Dental mark confirmed in 6x detail; numeral tag matches prompt.',
        'Both n letters and medial dot match; interrogative tag retained.',
        'Colon and final wedge match; interrogative tag retained.',
        'Affricate, doubled length notation, and final wedge match.',
        'Raised i confirmed in 6x detail, not a baseline sequence.',
        'Literal kisu and fire prompt match the Podia column.',
        'Literal kasaɾi and knife prompt match the Podia column.',
        'Second d has dental mark, confirmed in 6x detail.',
        'Dental mark and old prompt match the Bhamani Madia column.',
        'Complete printed sentence and free translation match; sentential scope appropriate.',
        'Complete printed sentence and free translation match; sentential scope appropriate.',
        'Complete printed sentence and free translation match; sentential scope appropriate.',
        'Source segment continues into next unit; multiword-expression scope correctly avoids standalone sentence claim.',
        'Question label and answer hint correctly separated; interr sentential tags match.',
        'Question label and answer hint correctly separated; interr sentential tags match.',
        'Utnoor village comparison naru matches; no inferred etymology edge.',
        'Bhamani Madia salt comparison ovori matches; no inferred etymology edge.',
    ]
    units = selection['selected_units']
    assert len(units) == len(pages) == len(observations) == 20
    findings = []
    for unit, page, note in zip(units, pages, observations):
        findings.append(dict(source_unit_key=unit['source_unit_key'], kind=unit['kind'],
            actual_rows=unit['actual_rows'], pdf_page=page, printed_page=page-5,
            verdict='pass', material_errors=[], observation=note,
            image_evidence=f'tmp/koya-whole/koya-r2-pdf{page}.png'))
    report = dict(status='accepted independent source-output audit; full ingestion gates deferred',
        reviewer='root', date='2026-09-27', seed=selection['seed'],
        manifest_sha256=expected, selection_sha256=sha(selection_path),
        source_pdf='tmp/pdfs/koya/JLSR2021_029.pdf',
        source_pdf_sha256='a6541e0d2397849ce7c36961b3849f3b2c1f1c267036cfa1a3f6025796e14e7d',
        method='Root visually compared exact frozen CSV rows against original 2x page renders and six 6x handwritten-detail crops; decisions recorded after visual review, not inferred by this script.',
        source_units=20, actual_rows=sum(len(u['actual_rows']) for u in units),
        material_errors=0, known_systematic_error_classes=[],
        checks=['literal form', 'gloss scope', 'grammatical tags', 'lect', 'citation', 'graph claim'],
        excluded_prior_source_units=selection['excluded_prior_source_units'], per_unit=findings,
        additional_edge_checks=[
            'PDF12: all four different comparisons retain printed spaces (veGe veGe, vere vere, vere vere, vera vera) and multiword-expression tags.',
            'Numeral comparison rows retain num tags.',
            'Narrative free translations remain segment-level; no fabricated word alignment.',
        ],
        deferred=['canonical installation and focused installed checks', 'full CLDF build and suite',
                  'compiled durable-ID and graph verification', 'user-triggered browser database refresh and QA'])
    target = F / 'root-independent-output-audit-20260927.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(f'{target}: 20/20 source units passed, 0 material errors')


if __name__ == '__main__':
    main()
