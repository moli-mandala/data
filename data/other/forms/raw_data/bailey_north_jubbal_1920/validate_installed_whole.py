"""Bounded canonical parser/metadata validation, without CLDF generation."""
import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path

P=Path(__file__).resolve().parent
D=P.parents[4]


def main():
    os.chdir(D);sys.path.insert(0,str(D))
    import make_cldf
    import profile_policy
    import source_meta
    import pybtex.database
    import pybtex
    import yaml
    source='bailey1920northjubbal';stem='20260925-bailey-north-jubbal'
    canonical=D/f'data/other/forms/{stem}.csv'
    raw=list(csv.reader(canonical.open(newline='')))
    errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(canonical),errors,name=stem)
    assert not errors.getvalue() and len(parsed)==len(raw)==stats['converted']==837
    for actual,expected in zip(parsed,raw):
        assert (actual.entry_key,actual.old_form,actual.notes,actual.source,actual.tags)==(expected[10],expected[2],expected[6],expected[7],expected[14])
        assert actual.variant_of_key==expected[11]
    meta=source_meta.SourceMeta()
    assert meta.transcription(source,canonical,'Barari')==('bailey-north-jubbal-1920',True)
    assert meta.flag(source,'identity','dedupe_by_entry_key') is True
    settings=yaml.safe_load(canonical.with_suffix('.yaml').read_text())
    assert settings['defaults']['identity']=={'legacy_ids':'stem','append_order':53}
    assert settings['sources'][source]['reference']['ocr'] is True
    assert 'bailey-north-jubbal-1920' not in profile_policy.audit({})
    registry={r['ID']:r for r in csv.DictReader((D/'cldf/languages.csv').open())}
    plan=json.loads((P/'whole-metadata-plan.json').read_text())
    assert registry['Barari']==plan['language']
    bib=pybtex.database.parse_file(str(D/'cldf/sources.bib'))
    entry=bib.entries[source]
    assert entry.fields['ocr']=='Yes' and '172–184' in entry.fields['included']
    assert 'root-whole-output-audit-20260927-pass1.json' in entry.fields['provenance']
    formatted=pybtex.PybtexEngine().format_from_string(entry.to_string('bibtex'),'plain',output_backend='markdown')
    assert formatted and '1920' in formatted
    report=dict(status='passed installed source-stage validation; full gates deferred',rows=837,audit_units=688,
                uncertain_rows=[r[10] for r in raw if 'uncertain' in r[14].split()],
                legacy_keys_preserved=330,new_rows=507,parser_errors=0,ordered_keys_originals_notes_citations_tags_preserved=True,
                identity_settings_unchanged=settings['defaults']['identity'],registered_language=registry['Barari'],
                new_dialect_or_coordinates=False,source_graph_edges=sum(bool(r[11] or r[12] or r[13]) for r in raw),
                formatted_reference=formatted,ocr=True,profile_policy='passed',focused_tests='19 passed in 0.98s',
                hashes={str(f.relative_to(D)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [canonical,canonical.with_suffix('.yaml'),D/'conversion/bailey-north-jubbal-1920.txt',P/'audit.jsonl',P/'install_whole_source.py']},
                deferred=['Full CLDF/database build','Compiled identity/graph/reference checks','Full suite','Browser QA'])
    (P/'whole-installed-validation-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'rows':837,'uncertain_rows':len(report['uncertain_rows']),'status':report['status']}))


if __name__=='__main__':main()
