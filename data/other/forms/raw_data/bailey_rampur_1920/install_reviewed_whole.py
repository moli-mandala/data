"""Install the independently accepted paired source files; never build the database."""
import argparse
import csv
import json
import re
from pathlib import Path
import yaml
from pybtex.database import parse_string
from freeze_full_reread_for_audit import sha
from verify_pass4_regeneration import main as verify_regeneration

P = Path(__file__).resolve().parent
D = P.parents[4]
FREEZE = P/'whole-paired-pass4-freeze-20260927.json'


def verify():
    freeze = json.loads(FREEZE.read_text())
    audit = json.loads((P/'root-independent-whole-output-audit-20260927-pass4.json').read_text())
    assert audit['freeze_sha256'] == sha(FREEZE)
    assert audit['material_errors'] == 0 and audit['sample_size'] >= 20
    for filename, digest in freeze['input_sha256'].items():
        assert sha(P/filename) == digest, filename
    for name, files in freeze['packages'].items():
        for filename, digest in files.items():
            assert sha(P.parent/f'bailey_{name}_1920'/filename) == digest, filename
    return freeze


def generate(name):
    verify()
    package = P.parent/f'bailey_{name}_1920'
    with (package/'pass4-preview.csv').open(newline='') as stream:
        rows = list(csv.reader(stream))
    audit = [json.loads(line) for line in (package/'pass4-preview-audit.jsonl').read_text().splitlines()]
    return rows, audit


def targets():
    verify()
    metadata = json.loads((P/'pass4-metadata-reconciliation.json').read_text())
    assert metadata['freeze_sha256'] == sha(FREEZE)
    pending = []
    bib_path = D/'cldf/sources.bib'
    original_bib = bib_path.read_text()
    bib = original_bib
    for name in ['rampur','baghi']:
        package = P.parent/f'bailey_{name}_1920'
        stem = f'20260925-bailey-{name}'
        for approved, destination, baseline in [
            ('pass4-preview.csv', D/f'data/other/forms/{stem}.csv', 'legacy-before-whole-recovery.csv'),
            ('pass4-preview-audit.jsonl', package/'audit.jsonl', 'legacy-before-whole-recovery-audit.jsonl'),
            ('pass4-profile.txt', D/f'conversion/bailey-{name}-1920.txt', 'legacy-before-whole-recovery-profile.txt'),
        ]:
            content = (package/approved).read_bytes()
            assert destination.read_bytes() in [(package/baseline).read_bytes(),content], destination
            pending.append((destination,content))
        yaml_path = D/f'data/other/forms/{stem}.yaml'
        proposal = package/'pass4-proposed.yaml'
        assert sha(proposal) == metadata['packages'][name]['proposed_yaml_sha256']
        meta = yaml.safe_load(proposal.read_text())
        meta['defaults']['importer']['commands'] = [[
            'data/other/forms/raw_data/bailey_rampur_1920/install_reviewed_whole.py', '--install']]
        content = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).encode()
        assert sha(yaml_path) == metadata['packages'][name]['current_yaml_sha256'] or yaml_path.read_bytes() == content
        pending.append((yaml_path,content))
        reference = metadata['packages'][name]['reference']
        source = reference['citation_id']
        pattern = rf'(@book\{{{source},\n)(.*?)(\n\}})'
        match = re.search(pattern,bib,re.S)
        assert match, source
        body = match.group(2)
        for field, value in reference['changed_fields'].items():
            assert not any(char in value for char in '{}\n'), field
            body, count = re.subn(rf'(?m)^  {field}\s*= \{{[^\n]*\}},?$',
                                  f'  {field:<13}= {{{value}}},',body)
            assert count == 1, field
        bib = bib[:match.start(2)] + body + bib[match.end(2):]
    before, after = parse_string(original_bib,'bibtex'), parse_string(bib,'bibtex')
    assert set(before.entries) == set(after.entries)
    for key, entry in before.entries.items():
        new = after.entries[key]
        assert entry.persons == new.persons and entry.type == new.type
        fields = set(entry.fields) | set(new.fields)
        changed = {field for field in fields if entry.fields.get(field) != new.fields.get(field)}
        assert changed <= ({'included','provenance'} if key in {'bailey1920rampur','bailey1920baghi'} else set())
    pending.append((bib_path,bib.encode()))
    return pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install',action='store_true')
    parser.add_argument('--check-pdf',action='store_true')
    args = parser.parse_args()
    if args.check_pdf:
        frozen = verify()
        assert sha(D.parent/frozen['source_pdf']) == frozen['source_pdf_sha256']
    verify_regeneration()
    pending = targets()
    before = {str(path.relative_to(D)):sha(path) for path,_ in pending}
    if args.install:
        for path, content in pending:
            temp = path.with_name(path.name+'.whole-install-tmp')
            temp.write_bytes(content)
            temp.replace(path)
    report = dict(status='source files installed; full ingestion gates deferred' if args.install else 'preflight passed; no canonical writes',
                  freeze_sha256=sha(FREEZE), installer_sha256=sha(Path(__file__)),
                  before_sha256=before, after_sha256={str(path.relative_to(D)):sha(path) for path,_ in pending},
                  pending=['Focused installed-source validation', 'Full database/graph/build/browser gates require user-authorized rebuild'])
    (P/'pass4-installation-result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(report['status'])


if __name__ == '__main__':
    main()
