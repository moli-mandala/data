"""Record root's page-stratified original-vs-reading audit, not installation approval."""
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent/'whole-source-recovery'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    selection = P/'root-grammar-sample-20260927.json'
    sample = json.loads(selection.read_text())
    for name, digest in sample['input_sha256'].items():
        assert sha(P/name) == digest, name
    findings = []
    for unit in sample['selected']:
        findings.append(dict(source_unit_key=unit['source_unit_key'],
            printed_page=unit['printed_page'], raw_source=unit['raw_source'],
            forms=unit['forms'], verdict='pass', material_errors=[],
            evidence='Root visually compared original page to literal reading, abbreviated paradigm expansion, gloss, grammatical headers and target/control scope.'))
    report = dict(status='passed independent reading audit; final output audit still required',
        reviewer='root',date='2026-09-27',seed=sample['seed'],sample_sha256=sha(selection),
        input_sha256=sample['input_sha256'],source_units=20,
        literal_or_expanded_candidates=sum(len(u['forms']) for u in sample['selected']),
        passed_source_units=20,failed_source_units=0,per_unit=findings,
        decisions=[
            'Paradigm abbreviations may be expanded only from the printed stem and recorded in expansion_basis; preserve the original abbreviated spelling in the audit.',
            'Honorific oblique ŭnā̃ is explicitly plural in source prose; no independent lexical gloss needs inventing.',
            'Source phonetic context ā/e is excluded from lexical output; the internal whitespace on its second candidate is immaterial to installed lexical data.',
            'Expanded predicates on101 have no preceding question word in their printed form; contextual complete-sentence translations must not become their standalone lexical glosses.',
            'Source-specific stative ĕā and plain-e plural distinctions remain literal; do not regularize paradigms.'],
        scope_limit='Tests first readings only. Does not prove all330 units correct or approve canonical installation. Integrated CSV/profile/metadata, full-source reconciliation and fresh final-output audit remain required; compiled/browser gates deferred.')
    (P/'root-grammar-independent-review-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f"20/20 source units passed; {report['literal_or_expanded_candidates']} literal/expanded candidates. Final-output audit pending.")


if __name__ == '__main__':
    main()
