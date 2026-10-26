"""Guarded exact installation/revalidation of the approved whole source."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from pybtex.database import parse_file, parse_string
import prepare_source

P = Path(__file__).resolve().parent
ROOT = next(p for p in P.parents if (p/'source_meta.py').exists())
CSV = ROOT/'data/other/forms/20260927-kera-mundari-living.csv'
YAML = CSV.with_suffix('.yaml')
PROFILE = ROOT/'conversion/kera-mundari-living-literal.txt'

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def validated():
    manifest = json.loads((P/'review-manifest-r1.json').read_text())
    for name,h in manifest['hashes'].items(): assert digest(P/name) == h, name
    review = json.loads((P/'independent-review-202609271210.json').read_text())
    approval = json.loads((P/'installation-authorization-evidence.json').read_text())
    assert review['material_errors'] == 0 and review['reviewed_source_units'] == 20
    assert digest(P/'proposal.csv') == review['proposal_csv_sha256'] == approval['approved_csv_sha256']
    rows,audit,_ = prepare_source.prepare()
    assert prepare_source.serialize(rows).encode() == (P/'proposal.csv').read_bytes()
    assert ''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audit) == (P/'proposal-audit.jsonl').read_text()
    return rows

def install():
    rows = validated()
    identity = ROOT/'data/form-identities.csv'; old_identity = digest(identity)
    original_yaml = (P/'proposal.yaml').read_text()
    yaml = original_yaml.replace('file: proposal.csv','file: '+CSV.name).replace(
        'defaults:\n','defaults:\n  importer:\n    commands:\n      - [data/other/forms/raw_data/kera_mundari_living_2026/import_source.py, --install]\n    note: Complete approved public dictionary and about-page target-family attestations.\n',1)
    targets = {CSV:(P/'proposal.csv').read_bytes(), YAML:yaml.encode(),
               PROFILE:(P/'proposal-profile.txt').read_bytes(), P/'installed-audit.jsonl':(P/'proposal-audit.jsonl').read_bytes()}
    for path, content in targets.items():
        assert not path.exists() or path.read_bytes() == content, f'Unexpected existing edits: {path}'
    bib = ROOT/'cldf/sources.bib'; old_bib = bib.read_bytes(); before = parse_file(str(bib))
    additions = parse_file(str(P/'proposed-reference.bib'))
    source = additions.entries[prepare_source.SOURCE]
    source.fields['included'] = 'Whole pinned dictionary: 446 entries and 446 senses, 462 readings; complete about-page target-family evidence, 17 further readings; total 479 forms'
    source.fields['provenance'] = 'data/other/forms/20260927-kera-mundari-living.csv; data/other/forms/raw_data/kera_mundari_living_2026/installed-audit.jsonl'
    for key,entry in additions.entries.items():
        entry.fields['jambu_editor'] = 'OpenAI Codex'
        entry.fields['etymology_provenance'] = 'source' if key == prepare_source.SOURCE else 'none'
        if key != prepare_source.SOURCE:
            entry.fields['included'] = 'Auxiliary background attribution only; original chapter not independently consulted'
        assert key not in before.entries or before.entries[key] == entry, f'Unexpected reference: {key}'
    suffix = ''.join('\n'+entry.to_string('bibtex')+'\n' for key,entry in additions.entries.items() if key not in before.entries)
    new_bib = old_bib + suffix.encode()
    dialects = ROOT/'cldf/dialects.csv'; old_dialects = dialects.read_bytes()
    actual = list(csv.DictReader(io.StringIO(old_dialects.decode())))
    expected = list(csv.DictReader((P/'proposed-dialect.csv').open()))
    extra = []
    for d in expected:
        same = [a for a in actual if a['ID'] == d['ID'] or a['Tag'] == d['Tag'] or a['Source_Language_ID'] == d['Source_Language_ID']]
        assert not same or same == [d], f'Unexpected existing dialect: {d["ID"]}'
        if not same: extra.append(d)
    out = io.StringIO(newline=''); writer=csv.DictWriter(out,fieldnames=list(actual[0])); writer.writerows(extra)
    new_dialects = old_dialects + out.getvalue().encode()
    assert bib.read_bytes() == old_bib and dialects.read_bytes() == old_dialects
    for path,content in targets.items(): path.write_bytes(content)
    if suffix: bib.write_bytes(new_bib)
    if extra: dialects.write_bytes(new_dialects)
    after = parse_file(str(bib))
    assert all(after.entries[k] == entry for k,entry in before.entries.items())
    assert digest(identity) == old_identity
    report = {'status':'source-stage installed; compiled/database/browser gates deferred',
        'rows':len(rows), 'legacy_source_keys':0, 'canonical_sha256':digest(CSV),
        'audit_sha256':digest(P/'installed-audit.jsonl'), 'profile_sha256':digest(PROFILE),
        'yaml_sha256':digest(YAML),'durable_identity_sha256_unchanged':old_identity,
        'unrelated_bibliography_entries_unchanged':len(before.entries),
        'new_bibliography_keys':sorted(set(after.entries)-set(before.entries)),
        'new_dialects':[r['ID'] for r in extra], 'frozen_manifest_sha256':digest(P/'review-manifest-r1.json')}
    receipt = P/'installation-report-20260927.json'
    if not receipt.exists(): receipt.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--install',action='store_true'); args=ap.parse_args()
    if args.install: install()
    else: print(f'{len(validated())} approved rows verified; no writes')
