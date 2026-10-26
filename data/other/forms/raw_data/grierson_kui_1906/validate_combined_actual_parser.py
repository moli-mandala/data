"""Read-only combined candidate checks using an unmodified isolated parser."""
import ast
import collections
import csv
import io
import json
import os
import re
import shutil
import sys
import unicodedata as ud
from urllib.parse import unquote
import yaml
from validate_profile_actual_parser import HERE, DATA, sha, read

def main():
    integration=HERE/'integration'
    status=json.loads((integration/'status.json').read_text())
    pins=status['outputs']
    for name,expected in pins.items():assert sha(integration/name)==expected,(name,'stale output manifest')
    rows=list(csv.reader((integration/'combined-candidate.csv').open()))
    audit=read(integration/'combined-audit.jsonl')
    assert all(len(r)==15 for r in rows)
    keys={r[10] for r in rows}
    assert len(keys)==len(rows)
    langids={r['ID'] for r in csv.DictReader((DATA/'cldf/languages.csv').open())}
    tree=ast.parse((DATA/'tags.py').read_text())
    allowed=set()
    for name in ('GRAMMATICAL_TAGS','GENDER_TAGS'):
        allowed |= next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
    dialects=set()
    for row in rows:
        assert row[0] in langids,(row[10],row[0])
        for tag in row[14].split():
            if tag.startswith('dialect:'):
                parts=tag.split(':')
                assert len(parts)==4 and unquote(parts[1])==row[0],(row[10],tag)
                dialects.add(tag)
            else:
                assert tag in allowed,(row[10],'unregistered tag',tag)
        for field in (11,12,13):
            assert all(k in keys for k in row[field].split('|') if k),(row[10],field,row[field])
    accounted={k for r in audit for k in r.get('entry_keys',[]) if k in keys}
    assert accounted==keys,('unaccounted emitted keys',keys-accounted)
    assert len({r['source_unit_key'] for r in audit})==len(audit)
    reuse_checks=[]
    external_tables={}
    for item in audit:
        if item['disposition']!='reused-existing-occurrence':
            continue
        witnesses=item['source_record']['verified_existing_occurrence_reuse']
        verified=[]
        for witness in witnesses:
            path=(DATA.parent/witness['file']).resolve()
            assert path.is_relative_to(DATA.resolve()),path
            if path not in external_tables:
                with path.open() as f:
                    external_tables[path]={r[10]:r for r in csv.reader(f)}
            expected=witness['row']
            assert external_tables[path][expected[10]]==expected,(item['source_unit_key'],'existing reuse witness drift')
            verified.append(expected[10])
        assert set(verified)==set(item['entry_keys'])
        reuse_checks.append(dict(source_unit_key=item['source_unit_key'],existing_keys=verified))
    atomic=read(HERE/'specimens/atomic-emissions-reviewed.jsonl')
    expressions=read(HERE/'specimens/expression-emissions-reviewed.jsonl')
    emitted={r[10]:r for r in rows}
    for atom in atomic:
        key=atom['emission_key']
        assert key in emitted,key
        assert emitted[key][2]==atom['emission_form'],key
        if len(atom['physical_source_keys'])>1:
            assert atom['physical_source_keys'][1] not in emitted
    for e in expressions:
        assert emitted[e['source_unit_key']][2]==e['emission_form']
        assert 'sentential' in emitted[e['source_unit_key']][14].split()
    scratch=DATA.parent/'tmp/kui-specimens/combined-parser-sandbox'
    target=scratch/'data/other/forms'
    target.mkdir(parents=True,exist_ok=True)
    (scratch/'conversion').mkdir(exist_ok=True)
    for name in ('make_cldf.py','source_meta.py'):shutil.copyfile(DATA/name,scratch/name)
    meta=json.loads((integration/'metadata-proposal.json').read_text())
    stem='combined-candidate'
    profile_name=meta['defaults']['transcription']['profile']
    shutil.copyfile(integration/'literal-preservation.txt',scratch/'conversion'/f'{profile_name}.txt')
    (target/f'{stem}.yaml').write_text(yaml.safe_dump(meta,sort_keys=False))
    os.chdir(scratch)
    sys.path[:0]=[str(scratch),str(DATA)]
    import make_cldf
    checks={}
    field_names=('lang','param','old_form','gloss','native','ipa','notes','source','cognateset','etymology','entry_key','variant_of_key','borrowed_from_key','derivation_parent_keys','tags')
    for normalization in ('NFC','NFD'):
        inputs=[[ud.normalize(normalization,v) if i==2 else v for i,v in enumerate(r)] for r in rows]
        path=target/f'{stem}.csv'
        with path.open('w',newline='') as f:csv.writer(f).writerows(inputs)
        errors=io.StringIO()
        parsed,stats=make_cldf.parse_file(str(path),errors,name=stem)
        assert not errors.getvalue(),errors.getvalue()
        assert len(parsed)==stats['converted']==len(rows)
        for actual,expected in zip(parsed,inputs):
            assert [getattr(actual,n) for n in field_names]==expected,(actual.entry_key,'field changed')
            display=ud.normalize('NFC',expected[2].strip('1234⁴5⁵67⁷,;.')).lower().replace('ṅ','ŋ')
            assert ud.normalize('NFC',actual.form)==display,(actual.entry_key,actual.form,display)
        checks[normalization]=dict(rows=len(parsed),errors=0,all15source_fields_retained=True,ordered_keys=True,no_unintended_splits=True,approved_display=True)
    for name,expected in pins.items():assert sha(integration/name)==expected,(name,'changed during validation')
    report=dict(status='combined actual-parser and scoped source accounting pass; independent acceptance and canonical metadata pending',
        output_pins=pins,status_sha256=sha(integration/'status.json'),checks=checks,
        parser_code={n:sha(DATA/n) for n in ('make_cldf.py','source_meta.py','tags.py')},
        accounting=dict(rows=len(rows),audit_units=len(audit),atomic_emissions_verified=len(atomic),expressions_verified=len(expressions),
                        dispositions=dict(collections.Counter(r['disposition'] for r in audit))),
        dialect_tokens=sorted(dialects),existing_occurrence_reuses_verified=reuse_checks,
        pending=['Canonical dialect/source declaration and bibliography registration still need guarded integration; encoded tokens checked for parent/syntax only.',
            'Graph claims may still be revised by independent source review; this report covers only pinned candidate.',
            'Final independent output audit and approved freeze; no full build/database/browser run.'],canonical_changes=False)
    destination=HERE/'metadata-profile-proposal/combined-actual-parser-review.json'
    if destination.exists():
        previous=json.loads(destination.read_text())
        previous_hash=previous['output_pins']['combined-candidate.csv']
        if previous_hash!=pins['combined-candidate.csv']:
            archive=destination.with_name('combined-actual-parser-review-'+previous_hash[:16]+'.json')
            if archive.exists():
                assert archive.read_bytes()==destination.read_bytes()
            else:
                shutil.copyfile(destination,archive)
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(checks=checks,accounting=report['accounting']),indent=2))

if __name__=='__main__':main()
