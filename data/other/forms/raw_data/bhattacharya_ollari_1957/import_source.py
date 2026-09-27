"""Rebuild reviewed Ollari inputs; --install requires the passing pinned audit."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import yaml
from pybtex.database import parse_file
import integration_preview

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
STEM = '20260921-bhattacharya-ollari'


def install(output):
    sample_path = HERE / 'audits/integration-sample-2026092105.json'
    sample = json.loads(sample_path.read_text())
    result = json.loads((HERE / 'audits/integration-results-2026092105.json').read_text())
    if result['material_errors'] or result['physical_records_reviewed'] != 20:
        raise ValueError('Passing acceptance audit required')
    if result['sample_sha256'] != hashlib.sha256(sample_path.read_bytes()).hexdigest():
        raise ValueError('Acceptance sample changed')
    for relative, expected in sample['input_sha256'].items():
        path = output / Path(relative).name if relative.startswith('integration-preview/') else HERE / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Audited input changed: {relative}')
    settings = yaml.safe_load((output / f'{STEM}.yaml').read_text())
    settings['defaults']['identity'] = {'legacy_ids': 'stem', 'append_order': 32}
    settings['defaults']['importer'] = {'commands': [[str((HERE / 'import_source.py').relative_to(ROOT)), '--install']]}
    registry_path = ROOT / 'cldf/sources.bib'
    registry = parse_file(str(registry_path)).entries
    additions = []
    for filename in ['source-reference.bib', 'auxiliary-references.bib', 'bibliography-only-references.bib']:
        entries = parse_file(str(HERE / filename)).entries
        for key, entry in entries.items():
            if key == 'bhattacharya1957ollari':
                entry.fields['included'] = 'Complete vocabulary, printed pp. 48--77: 657 physical entries, 880 lexical rows and 509 entry-prose passages; comparative languages retained as prose, not separately ingested.'
                entry.fields['provenance'] = f'data/other/forms/{STEM}.csv; data/other/entry_texts/{STEM}.csv; data/other/forms/raw_data/bhattacharya_ollari_1957/manifest.json; data/other/forms/raw_data/bhattacharya_ollari_1957/integration-preview/physical-record-audit.jsonl'
            if key in registry:
                if registry[key] != entry:
                    raise ValueError(f'Existing bibliography differs: {key}')
            else:
                # Keep URL/path underscores literal, as in the source templates.
                additions.append(entry.to_string('bibtex').replace('\\_', '_'))
    forms = ROOT / 'data/other/forms'
    texts = ROOT / 'data/other/entry_texts'
    texts.mkdir(parents=True, exist_ok=True)
    (forms / f'{STEM}.csv').write_bytes((output / f'{STEM}.csv').read_bytes())
    (forms / f'{STEM}.yaml').write_text(yaml.safe_dump(settings, sort_keys=False, allow_unicode=True))
    (texts / f'{STEM}.csv').write_bytes((output / f'{STEM}-entry-texts.csv').read_bytes())
    if additions:
        with registry_path.open('a') as stream:
            stream.write('\n' + '\n\n'.join(additions) + '\n')
    return {'installed_rows': 880, 'installed_prose_passages': 509,
            'bibliography_added': len(additions), 'compiled_validation': 'pending'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='ollari-import-') as temp:
        output = args.output or Path(temp)
        summary = integration_preview.build(output)
        if args.install:
            summary = install(output)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
