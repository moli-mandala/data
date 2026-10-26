"""Prepare own-file settings/reference/audit scaffold, never shared baseline assets."""
import csv,hashlib,json
from pathlib import Path
import yaml
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
FINAL=PACKAGE/'integration/revision6'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    freeze=json.loads((FINAL/'review-freeze.json').read_text())
    for name,pin in freeze['files'].items():assert sha(PACKAGE/name)==pin,('Frozen input drift',name)
    settings=yaml.safe_load((FINAL/'combined-review-candidate.yaml').read_text())
    settings['file']='20260927-grierson-kolami-basim.csv'
    settings['defaults']['importer']={'commands':[['data/other/forms/raw_data/grierson_kolami_basim_1906/installation/install_source.py','--install']]}
    (HERE/'20260927-grierson-kolami-basim.yaml').write_text(yaml.safe_dump(settings,sort_keys=False,allow_unicode=True))
    refs=(FINAL/'references-review-candidate.bib').read_text()
    # Inclusion and source authority stay identical. Only planned canonical
    # provenance changes after source-stage acceptance; final root reviews it.
    refs=refs.replace('Planned canonical data/other/forms/','Canonical source-stage data/other/forms/',1)
    refs=refs.replace('integration/revision5/','integration/revision6/')
    (HERE/'staged-references.bib').write_text(refs)
    (HERE/'staged-dialects.csv').write_bytes((PACKAGE/'metadata-profile-proposal/dialects-proposed.csv').read_bytes())
    audit=[json.loads(s) for s in (FINAL/'combined-review-audit.jsonl').read_text().splitlines()]
    assert len(audit)==1570
    # Do not label rows accepted before root final actual-output acceptance.
    assert all(a['accepted_for_installation'] is False for a in audit)
    with (FINAL/'combined-review-candidate.csv').open() as f:rr=list(csv.reader(f))
    ownkeys={r[10] for r in rr}
    reused={k for a in audit for k in a['entry_keys']}-ownkeys
    assert len(ownkeys)==827 and len(reused)==3
    counts={'status':'Own source assets staged; accepted audit/shared baseline/root activation pending',
      'rows':827,'audit_records':1570,'language_counts':{},'uncertain_emitted_rows':sum('uncertain' in r[14].split() for r in rr),
      'reused_existing_keys':sorted(reused),'freeze_sha256':sha(FINAL/'review-freeze.json'),
      'staged_assets':{p.name:sha(p) for p in [HERE/'20260927-grierson-kolami-basim.yaml',HERE/'staged-references.bib',HERE/'staged-dialects.csv']},
      'shared_baseline_captured':False,'audit_acceptance_ready':False,'canonical_changes':False}
    for r in rr:counts['language_counts'][r[0]]=counts['language_counts'].get(r[0],0)+1
    (HERE/'own-source-assets-status.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(counts,ensure_ascii=False))
if __name__=='__main__':main()
