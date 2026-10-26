"""Guarded paired source-stage installer proposal; requires root's accepted audit.

Default action is verification only. No database/CLDF build or remote action.
Both packages regenerate in a bounded source-only sandbox before any canonical
write. Bibliography/YAML/checklist installation is a separate coordinated step.
"""
import argparse,csv,hashlib,json,shutil,subprocess,sys
from pathlib import Path

P=Path(__file__).resolve().parent
D=P.parents[4]
FREEZE=P/'whole-paired-output-freeze-20260927.json'
SANDBOX=D.parent/'tmp/rohru-whole-review/paired-install-validation'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def require_local_name(name):
    path=Path(name)
    if path.is_absolute() or '..' in path.parts:raise ValueError(f'Nonlocal manifest path: {name}')
    return path


def verify_approval(audit_path):
    report=json.loads(audit_path.read_text())
    if report.get('material_errors')!=0 or report.get('units_reviewed',0)<20:
        raise ValueError('Whole-source root audit has not passed at least20 reviewed source units')
    if report.get('freeze_sha256')!=sha(FREEZE):
        raise ValueError('Root audit does not pin the current paired output freeze')
    freeze=json.loads(FREEZE.read_text())
    for name,expected in freeze['input_sha256'].items():
        path=P/require_local_name(name)
        if sha(path)!=expected:raise ValueError(f'Frozen source input changed: {name}')
    for package,info in freeze['packages'].items():
        directory=P.parent/f'bailey_{package}_1920'
        for name,expected in info['files'].items():
            if sha(directory/require_local_name(name))!=expected:
                raise ValueError(f'Frozen output changed: {package}/{name}')
    return freeze,report


def regenerate(freeze):
    # Copies only named small source artifacts. No source PDF, working-tree
    # checkout, external cache, or .git directory is copied.
    fake_data=SANDBOX/'data'
    fake_raw=fake_data/'data/other/forms/raw_data'
    fake_ramp=fake_raw/'bailey_rampur_1920'
    fake_ramp.mkdir(parents=True,exist_ok=True)
    (fake_data/'conversion').mkdir(parents=True,exist_ok=True)
    for name in freeze['input_sha256']:
        relative=require_local_name(name)
        target=fake_ramp/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(P/relative,target)
    for name in ['rampur','baghi']:
        real=P.parent/f'bailey_{name}_1920';fake=fake_raw/f'bailey_{name}_1920'
        fake.mkdir(parents=True,exist_ok=True)
        for filename in ['legacy-before-whole-recovery.csv','legacy-before-whole-recovery-audit.jsonl','legacy-before-whole-recovery-profile.txt']:
            shutil.copyfile(real/filename,fake/filename)
        shutil.copyfile(real/'legacy-before-whole-recovery.csv',fake_data/f'data/other/forms/20260925-bailey-{name}.csv')
        shutil.copyfile(real/'legacy-before-whole-recovery-profile.txt',fake_data/f'conversion/bailey-{name}-1920.txt')
    for script in ['prepare_paired_reviewed.py','prepare_paired_expansion.py','prepare_paired_proposals.py']:
        result=subprocess.run([sys.executable,str(fake_ramp/script)],cwd=fake_data,capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError(f'{script} failed:\n{result.stdout}\n{result.stderr}')
    comparisons={}
    for package,info in freeze['packages'].items():
        fake=fake_raw/f'bailey_{package}_1920'
        comparisons[package]={}
        for name in ['whole-proposed.csv','whole-proposed-audit.jsonl','whole-proposed-profile.txt']:
            actual=sha(fake/name)
            if actual!=info['files'][name]:raise ValueError(f'Regenerated artifact differs: {package}/{name}')
            comparisons[package][name]=actual
    return comparisons


def verify_current(freeze):
    targets=[]
    for name,info in freeze['packages'].items():
        p=P.parent/f'bailey_{name}_1920'
        canonical=D/f'data/other/forms/20260925-bailey-{name}.csv'
        current=list(csv.reader(canonical.open()))
        legacy=list(csv.reader((p/'legacy-before-whole-recovery.csv').open()))
        approved=list(csv.reader((p/'whole-proposed.csv').open()))
        if current not in (legacy,approved):raise ValueError(f'Unexpected canonical CSV changes: {name}')
        current_audit=[json.loads(x) for x in (p/'audit.jsonl').read_text().splitlines()]
        old_audit=[json.loads(x) for x in (p/'legacy-before-whole-recovery-audit.jsonl').read_text().splitlines()]
        new_audit=[json.loads(x) for x in (p/'whole-proposed-audit.jsonl').read_text().splitlines()]
        if current_audit not in (old_audit,new_audit):raise ValueError(f'Unexpected canonical audit changes: {name}')
        profile=D/f'conversion/bailey-{name}-1920.txt'
        if profile.read_bytes() not in ((p/'legacy-before-whole-recovery-profile.txt').read_bytes(),(p/'whole-proposed-profile.txt').read_bytes()):
            raise ValueError(f'Unexpected canonical profile changes: {name}')
        targets.extend([(p/'whole-proposed.csv',canonical),(p/'whole-proposed-audit.jsonl',p/'audit.jsonl'),(p/'whole-proposed-profile.txt',profile)])
    return targets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True,help='Root accepted independent report with material_errors, units_reviewed and freeze_sha256')
    parser.add_argument('--install',action='store_true')
    args=parser.parse_args()
    freeze,review=verify_approval(args.audit)
    regenerated=regenerate(freeze)
    targets=verify_current(freeze)
    before={str(target.relative_to(D)):sha(target) for _,target in targets}
    if args.install:
        # Every target is preflighted before the first write; a repeated run
        # accepts only exact old or approved content, allowing safe resumption.
        for source,target in targets:
            temporary=target.with_name(target.name+'.paired-install-tmp')
            temporary.write_bytes(source.read_bytes());temporary.replace(target)
    report=dict(status='installed exact approved CSV/audit/profile; metadata and canonical-parser gates still required' if args.install else 'verified only; canonical unchanged',audit_path=str(args.audit),audit_sha256=sha(args.audit),freeze_sha256=sha(FREEZE),regenerated=regenerated,before_sha256=before,after_sha256={str(target.relative_to(D)):sha(target) for _,target in targets},deferred=['bibliography/YAML/checklist reconciliation if not separately completed','actual canonical parser and identity checks','fullCLDF/database/build/browser gates'])
    (P/'whole-guarded-installation-result-20260927.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
