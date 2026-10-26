"""Regenerate reviewed whole CSV/audits in a small isolated source-only sandbox."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
from freeze_full_reread_for_audit import sha

P = Path(__file__).resolve().parent
D = P.parents[4]
S = D.parent/'tmp/rohru-whole-review/pass4-regeneration'


def main():
    freeze_path = P/'whole-paired-pass4-freeze-20260927.json'
    freeze = json.loads(freeze_path.read_text())
    accepted_path = P/'root-independent-whole-output-audit-20260927-pass4.json'
    accepted = json.loads(accepted_path.read_text())
    assert accepted['material_errors'] == 0 and accepted['sample_size'] >= 20
    assert accepted['freeze_sha256'] == sha(freeze_path)
    # Verify every producer/input named in all three retained revision manifests.
    for filename in ['whole-paired-output-freeze-20260927-pass2.json',
                     'whole-paired-full-reread-freeze-20260927.json',
                     'whole-paired-pass4-freeze-20260927.json']:
        manifest = json.loads((P/filename).read_text())
        for relative, digest in manifest['input_sha256'].items():
            assert sha(P/relative) == digest, relative
        for name, info in manifest['packages'].items():
            for relative, digest in info.get('files',info).items():
                assert sha(P.parent/f'bailey_{name}_1920'/relative) == digest, relative
    fake_data = S/'data'
    fake_forms = fake_data/'data/other/forms'
    fake_raw = fake_forms/'raw_data'
    (fake_data/'conversion').mkdir(parents=True,exist_ok=True)
    copied = {}
    for name in ['rampur','baghi']:
        source = P.parent/f'bailey_{name}_1920'
        target = fake_raw/source.name
        target.mkdir(parents=True,exist_ok=True)
        for path in source.iterdir():
            if path.is_file() and path.suffix in {'.py','.json','.jsonl','.csv','.tsv','.txt'}:
                assert path.stat().st_size < 20_000_000, path
                shutil.copyfile(path,target/path.name)
                copied[f'{source.name}/{path.name}'] = sha(path)
        shutil.copyfile(source/'legacy-before-whole-recovery.csv',fake_forms/f'20260925-bailey-{name}.csv')
        shutil.copyfile(source/'legacy-before-whole-recovery-profile.txt',fake_data/f'conversion/bailey-{name}-1920.txt')
    scripts = ['prepare_paired_reviewed.py','prepare_paired_expansion.py','prepare_paired_proposals.py',
               'prepare_full_reread_preview.py','prepare_pass4_preview.py']
    for script in scripts:
        result = subprocess.run([sys.executable,str(fake_raw/'bailey_rampur_1920'/script)],
                                cwd=fake_data,capture_output=True,text=True)
        assert result.returncode == 0, (script,result.stdout,result.stderr)
    comparisons = {}
    for name in ['rampur','baghi']:
        comparisons[name] = {}
        for filename in ['pass4-preview.csv','pass4-preview-audit.jsonl']:
            actual = sha(fake_raw/f'bailey_{name}_1920'/filename)
            assert actual == freeze['packages'][name][filename], (name,filename)
            comparisons[name][filename] = actual
    report = dict(status='source-only regeneration matches accepted CSV and audit bytes',
                  freeze_sha256=sha(freeze_path), accepted_audit_sha256=sha(accepted_path),
                  producer_sha256=sha(Path(__file__)), scripts=scripts, comparisons=comparisons,
                  copied_input_sha256=copied, canonical_changes=False,
                  profile_gate='Accepted profile bytes are pinned by freeze and already tested with actual parser; this check regenerates CSV/audit only')
    (P/'pass4-regeneration-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Both complete CSVs and audits regenerate byte-identically from reviewed source inputs.')


if __name__ == '__main__':
    main()
