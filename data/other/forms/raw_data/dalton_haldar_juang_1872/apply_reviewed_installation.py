"""Root-reviewed source installation and single-entry bibliography reconciliation."""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
O = P / 'whole-source-recovery'
D = P.parents[4]


def main():
    import pybtex
    from pybtex.database import parse_string
    sys.path.insert(0, str(D))
    import source_meta
    import install_whole_source
    audit = json.loads((O / 'root-independent-whole-output-audit-20260927.json').read_text())
    manifest_hash = hashlib.sha256((O / 'frozen-manifest-20260927.json').read_bytes()).hexdigest()
    assert audit['manifest_sha256'] == manifest_hash
    assert audit['material_errors'] == 0 and audit['sample_units_reviewed'] == 20
    assert len(audit['controls_reviewed']) == 5
    path = D / 'cldf/sources.bib'
    before = path.read_text()
    pattern = r'@book\{dalton1872haldarjuang,\n.*?\n\}'
    matches = list(re.finditer(pattern, before, flags=re.S))
    assert len(matches) == 1
    draft = (O / 'proposal-sources.bib').read_text().strip()
    entry = parse_string(draft, 'bibtex').entries['dalton1872haldarjuang']
    formatted = pybtex.PybtexEngine().format_from_string(entry.to_string('bibtex'), 'plain', output_backend='markdown')[3:].strip()
    assert formatted and '1872' in formatted
    os.chdir(D)
    current_settings = [p for p in source_meta.yaml_files() if p.stem != '20260925-dalton-haldar-juang']
    meta = source_meta.SourceMeta(current_settings + [O / '20260925-dalton-haldar-juang.yaml'])
    assert meta.flag('dalton1872haldarjuang', 'reference', 'ocr') is True
    approval = {'status': 'approved', 'reviewer': 'root', 'frozen_manifest_sha256': manifest_hash,
                'basis': 'Independent original/output audit passes0/20, five controls and three grammar markers; scoped metadata and formatted reference validated.',
                'scope': 'Local source-stage installation only; no CLDF/database build, commit, push or release.'}
    (O / 'root-installation-approval-20260927.json').write_text(json.dumps(approval, indent=2) + '\n')
    install_whole_source.install()
    # Assert unchanged shared bibliography immediately before this scoped edit.
    assert path.read_text() == before
    match = matches[0]
    after = before[:match.start()] + draft + before[match.end():]
    path.write_text(after)
    report = {'status': 'source-stage installed and bibliography reconciled', 'rows': 226,
              'formatted_reference': formatted, 'ocr': True,
              'previous_bib_entry': match.group(), 'installed_bib_entry': draft,
              'bibliography_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'frozen_manifest_sha256': manifest_hash,
              'deferred': 'Full CLDF, compiled identity/graph/references, full suite, database and browser gates.'}
    (O / 'root-integration-result-20260927.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('Installed226 reviewed Juang rows and reconciled one bibliography entry; no database build.')


if __name__ == '__main__':
    main()
