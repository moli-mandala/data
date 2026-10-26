"""Guarded registration of the source-reported adaptation reference only."""
import hashlib
import json
from pathlib import Path

from pybtex.database import parse_file, parse_string
from pybtex import PybtexEngine
from source_meta import SourceMeta

P = Path(__file__).resolve().parent
ROOT = next(p for p in P.parents if (p / 'source_meta.py').exists())
KEY = 'dewan_chakma2003chakma_first_lesson'
BIB = ROOT / 'cldf/sources.bib'
YAML = ROOT / 'data/other/forms/20260921-niranjan-chakma.yaml'

def sha(data): return hashlib.sha256(data).hexdigest()

def main():
    old_bib, old_yaml = BIB.read_bytes(), YAML.read_bytes()
    before = parse_file(str(BIB))
    assert KEY not in before.entries, 'Duplicate guard: reference already registered.'
    assert KEY not in SourceMeta().sources, 'Duplicate guard: source metadata already registered.'
    proposed = (P / 'grammar-auxiliary-reference-proposed.bib').read_text()
    expected = parse_string(proposed, 'bibtex').entries[KEY]
    expected.fields['included'] = 'Auxiliary adaptation reference only; no independent lexical ingestion or original-article collation'
    expected.fields['jambu_editor'] = 'OpenAI Codex'
    expected.fields['etymology_provenance'] = 'none'
    text = proposed.rstrip().removesuffix('}') + (
        '  included = {Auxiliary adaptation reference only; no independent lexical ingestion or original-article collation},\n'
        '  jambu_editor = {OpenAI Codex},\n'
        '  etymology_provenance = {none}\n}\n')
    assert parse_string(text, 'bibtex').entries[KEY] == expected
    yaml_addition = ('  ' + KEY + ':\n    reference:\n      editor: OpenAI Codex\n'
                     '      ocr: false\n      etymology_provenance: none\n')
    # Refuse concurrent changes rather than rewrite a shared file from stale data.
    assert BIB.read_bytes() == old_bib and YAML.read_bytes() == old_yaml
    BIB.write_bytes(old_bib.rstrip() + b'\n\n' + text.encode())
    YAML.write_bytes(old_yaml.rstrip() + b'\n' + yaml_addition.encode())
    after = parse_file(str(BIB))
    assert set(after.entries) - set(before.entries) == {KEY}
    assert all(after.entries[k] == v for k,v in before.entries.items())
    assert after.entries[KEY] == expected
    meta = SourceMeta()
    assert meta.flag(KEY, 'reference', 'editor') == 'OpenAI Codex'
    assert meta.flag(KEY, 'reference', 'ocr') is False
    assert meta.flag(KEY, 'reference', 'etymology_provenance') == 'none'
    rendered = PybtexEngine().format_from_string(text, 'plain', output_backend='markdown')
    assert '2003' in rendered and 'সদরক' in rendered
    report = {'status':'passed', 'key':KEY, 'scope':'auxiliary adaptation attribution only',
              'original_article_consulted':False, 'unrelated_bibliography_entries_unchanged':len(before.entries),
              'bibliography_before_sha256':sha(old_bib), 'bibliography_after_sha256':sha(BIB.read_bytes()),
              'yaml_before_sha256':sha(old_yaml), 'yaml_after_sha256':sha(YAML.read_bytes()),
              'source_proposal_sha256':sha((P/'grammar-auxiliary-reference-proposed.bib').read_bytes()),
              'formatted_reference':rendered, 'checks':['duplicate guards','Pybtex exact expected entry',
                 'all unrelated entries unchanged','SourceMeta editor/OCR/provenance','actual reference formatter'],
              'no_database_build':True}
    (P/'grammar-auxiliary-registration-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
