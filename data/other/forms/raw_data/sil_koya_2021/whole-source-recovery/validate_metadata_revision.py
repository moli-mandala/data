"""Reuse frozen validator implementations with explicit revision output paths."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/'literal-integration-metadata-r2'


def main():
    for name in ['validate_literal_integration','validate_literal_metadata']:
        path=HERE/f'{name}.py'
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        # Configuration only: the actual parser and policies remain unmodified.
        module.OUT=OUT
        if hasattr(module,'SANDBOX'):module.SANDBOX=module.DATA.parent/'tmp/rohru-whole-review/koya-parser-r2'
        if hasattr(module,'TMP'):module.TMP=module.DATA.parent/'tmp/rohru-whole-review/koya-metadata-r2'
        module.main()
    manifest=dict(status='metadata_revision_ready_for_independent_review_not_installed',
                  rows={'wordlist':1437,'narrative':210,'comparative':18,'total':1665},
                  prior_manifest_sha256=hashlib.sha256((HERE/'literal-integration/review-manifest.json').read_bytes()).hexdigest(),
                  outputs_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='review-manifest.json'},
                  scripts_sha256={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['prepare_metadata_revision.py','validate_metadata_revision.py','validate_literal_integration.py','validate_literal_metadata.py']},
                  canonical_modified=False,required_next='Fresh independent original-to-output audit; no installation or full build authorized by these checks.')
    (OUT/'review-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
