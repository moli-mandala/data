"""Prepare exact disabled approval draft. Root alone reviews and activates it."""
import importlib.util
import json
from pathlib import Path
from pybtex.database import parse_string

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('installer',HERE/'install_source.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def main():
    inputs={
      'legacy_csv':'legacy-review/legacy-repaired-candidate.csv',
      'addition_csv':'integration-proposal/all-additions.csv',
      'legacy_yaml':'routing-proposal/20260828-sil-korwa-kodaku.yaml',
      'addition_yaml':'routing-proposal/20260927-behera-whole-additions.yaml',
      'languages':'integration-proposal/registry-drafts/languages.csv',
      'dialects':'integration-proposal/registry-drafts/dialects.csv',
      'bibliography':'integration-proposal/sources.bib',
      'audit':'installation/accepted-audit.jsonl'}
    planned={m.DATA/m.TARGETS[r]:(m.HERE/n).read_bytes() for r,n in inputs.items()}
    old={r[10]:r for r in m.csv_rows((m.DATA/m.TARGETS['legacy_csv']).read_bytes())}
    new={r[10]:r for r in m.csv_rows(planned[m.DATA/m.TARGETS['legacy_csv']])}
    extra=m.csv_rows(planned[m.DATA/m.TARGETS['addition_csv']])
    files=set(inputs.values())|{'installation/install_source.py','installation/prepare_accepted_audit.py','installation/prepare_manifest_draft.py'}
    # Pin reviewed evidence, not caches or mutable pytest fixtures.
    for folder in ('continuous-independent-review','interlinear-independent-review','identity-review','comparisons','outside-review','legacy-review','integration-proposal','interlinear-review','routing-proposal'):
        for p in (m.HERE/folder).glob('*.json'):
            files.add(str(p.relative_to(m.HERE)))
    for p in (m.HERE/'installation').glob('*check.json'):files.add(str(p.relative_to(m.HERE)))
    a={'installation_approved':False,'result':'pass','review_scope':'whole-report',
       'draft_warning':'Not authorized for execution; root must review and activate exact hashes',
       'engine_sha256':m.file_hash(m.ENGINE),
       'files':{n:m.file_hash(m.HERE/n) for n in sorted(files)},
       'assets':{r:{'input':n,'output_sha256':m.file_hash(m.HERE/n)} for r,n in inputs.items()},
       'canonical_baselines':{n:m.file_hash(m.DATA/n) for n in m.TARGETS.values()},
       'durable_identity_sha256':m.file_hash(m.DATA/m.IDENTITY),
       'legacy_rows':len(new),'addition_rows':len(extra),'rows':len(new)+len(extra),
       'audit_units':6823,'audit_schema':{'key_field':'audit_key','entry_keys_field':'proposed_entry_keys'},
       'legacy_changed_keys':sorted(k for k in old if old[k]!=new[k]),
       'legacy_added_keys':sorted(set(new)-set(old)), 'registry_added_ids':{}}
    for role in ('languages','dialects'):
        _,before=m.registry((m.DATA/m.TARGETS[role]).read_bytes())
        _,after=m.registry(planned[m.DATA/m.TARGETS[role]])
        a['registry_added_ids'][role]=sorted(set(after)-set(before))
    before=parse_string((m.DATA/m.TARGETS['bibliography']).read_text(),'bibtex').entries
    after=parse_string(planned[m.DATA/m.TARGETS['bibliography']].decode(),'bibtex').entries
    a['bibliography_changed_keys']=sorted(k for k in before if before[k]!=after[k])
    a['bibliography_added_keys']=sorted(set(after)-set(before))
    pdf=m.DATA.parent/'tmp/pdfs/korwa-kodaku-2022/source.pdf'
    a['original_inputs']={str(pdf.relative_to(m.DATA.parent)):m.file_hash(pdf)}
    m.validate_assets(a,planned)
    (HERE/'root-acceptance-draft.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'disabled draft validated','rows':a['rows'],'audit_units':a['audit_units'],'pinned_files':len(a['files']),'installation_approved':False}))

if __name__=='__main__':main()
