"""Installed-only verifier: real SourceMeta, all fields and normalization checks.

Refuses to run before an approved transaction and receipt exist. It performs
no canonical mutation or build; the importer invocation is guarded as a no-op.
"""
import csv
import io
import json
import sys
import tempfile
import unicodedata
from pathlib import Path
from pybtex.database import parse_file
import pybtex
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
sys.path.insert(0,str(DATA))
import source_meta
import make_cldf
from tags import GRAMMATICAL_TAGS,GENDER_TAGS
from install_source import TARGETS,STEM,file_hash,main as installer_main
FIELDS=('lang','param','old_form','gloss','native','ipa','notes','source','cognateset','etymology','entry_key','variant_of_key','borrowed_from_key','derivation_parent_keys','tags')

def main():
    approval_path=HERE/'root-acceptance.json'
    receipt_path=HERE/'installation-receipt.json'
    assert approval_path.exists() and receipt_path.exists(),'Approved installation and receipt required before canonical checks'
    approval=json.loads(approval_path.read_text());receipt=json.loads(receipt_path.read_text())
    assert approval['installation_approved'] is True and receipt['approval_sha256']==file_hash(approval_path)
    paths={role:DATA/name for role,name in TARGETS.items()}
    for role in ('csv','yaml','audit'):
        assert file_hash(paths[role])==approval['assets'][role]['output_sha256']
    identity_before=file_hash(DATA/'data/form-identities.csv')
    assert identity_before==approval['durable_identity_sha256']
    source_meta.load.cache_clear();meta=source_meta.load()
    assert source_meta.importer_commands(meta,STEM)==[['data/other/forms/raw_data/nair_irula_2013/installation/install_source.py','--install']]
    assert meta.file_defaults(STEM+'.csv')['transcription']==[{'convert':False,'preserve_hyphens':True}]
    assert meta.transcription('nair2013irula',str(paths['csv']),'Irula')==(None,False)
    expected_flags={'identity':{'dedupe_by_entry_key':True},'forms':{'split_alternates':False},'gloss':{'grammar_tags':False}}
    for section,flags in expected_flags.items():
        for key,value in flags.items():assert meta.file_flag(STEM+'.csv',section,key) is value
    refs=('nair2013irula','zvelebil1982irulapart3','perialwar1978irula')
    for key in refs:
        assert meta.flag(key,'reference','editor')=='OpenAI Codex'
        assert meta.flag(key,'reference','ocr') is False
        assert meta.flag(key,'reference','etymology_provenance')==('source' if key==refs[0] else 'none')
    language={r['ID']:r for r in csv.DictReader((DATA/'cldf/languages.csv').open())}
    assert language['Irula']==approval['existing_irula_identity']
    rows=list(csv.reader(paths['csv'].open(newline='')))
    by={r[10]:r for r in rows};assert len(rows)==len(by)==256
    assert {r[0] for r in rows}=={'Irula'}
    assert sum(bool(r[11]) for r in rows)==12 and sum(bool(r[13]) for r in rows)==4
    for r in rows:
        assert not r[11] or r[11] in by
        assert not r[13] or all(k in by for k in r[13].split('|'))
        assert not r[12]
        assert set(r[14].split())<=GRAMMATICAL_TAGS|GENDER_TAGS
        assert unicodedata.is_normalized('NFC',r[2])
    assert not by['nair2013irula:p23:proximate-demonstratives:02'][11]
    history=list(map(json.loads,(PACKAGE/'proposal-audit.jsonl').read_text().splitlines()))
    accepted=list(map(json.loads,paths['audit'].read_text().splitlines()))
    assert len(history)==len(accepted)==261
    assert sum(not a['proposed_entry_keys'] for a in accepted)==16
    assert {k for a in accepted for k in a['proposed_entry_keys']}==set(by)
    for old,new in zip(history,accepted):
        assert all(new[k]==v for k,v in old.items() if k!='disposition')
        assert new['historical_disposition']==old['disposition'] and new['accepted_for_installation'] is True
    checks={}
    for normalization in ('canonical','NFC','NFD'):
        expected=rows if normalization=='canonical' else [[unicodedata.normalize(normalization,v) if i==2 else v for i,v in enumerate(r)] for r in rows]
        with tempfile.TemporaryDirectory(prefix='irula-parser-',dir=DATA.parent/'tmp') as folder:
            path=paths['csv']
            if normalization!='canonical':
                manual_dir=Path(folder)/'other/forms';manual_dir.mkdir(parents=True)
                path=manual_dir/(STEM+'.csv')
                with path.open('w',newline='') as f:csv.writer(f).writerows(expected)
            errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(path),errors,name=STEM)
        assert not errors.getvalue() and len(parsed)==256 and stats=={'converted':0,'for_conversion':0}
        for actual,raw in zip(parsed,expected):
            assert [getattr(actual,f) for f in FIELDS]==raw,actual.entry_key
            assert unicodedata.normalize('NFC',actual.form)==unicodedata.normalize('NFC',raw[2])
            assert make_cldf.source_entry_dedupe_key(actual,meta)==actual.entry_key
        assert len({(r.lang,r.param,r.form,make_cldf.source_entry_dedupe_key(r,meta)) for r in parsed})==256
        checks[normalization]={'rows':256,'all15fields_exact':True,'form_preserved':True,'unique_source_entry_dedupe_keys':256,'zero_conversion':True}
    current_bib=parse_file(str(paths['bib'])).entries
    baseline=parse_file(str(HERE/'baselines/bib.asset'),bib_format='bibtex').entries
    expected_refs=parse_file(str(HERE/'staged-references.bib')).entries
    assert all(current_bib[k]==v for k,v in baseline.items())
    assert all(current_bib[k]==v for k,v in expected_refs.items())
    formatted={k:pybtex.PybtexEngine().format_from_string(current_bib[k].to_string('bibtex'),'plain',output_backend='markdown')[3:].strip() for k in refs}
    assert all(formatted.values())
    before={role:file_hash(path) for role,path in paths.items()}
    assert installer_main(install=True)['canonical_mutations'] is False
    assert before=={role:file_hash(path) for role,path in paths.items()}
    assert file_hash(DATA/'data/form-identities.csv')==identity_before
    report={'status':'passed canonical source-stage checks; compiled/browser gates deferred','rows':256,'physical_units':261,'audit_only':16,'variants':12,'derived':4,'checks':checks,'canonical_sha256':before,'durable_identity_sha256':identity_before,'actual_registered_SourceMeta':True,'parser_monkeypatch':False,'repeat_install':'exact no-op','reference_flags':3,'formatted_references':formatted,'unrelated_bibliography_members_unchanged':len(baseline),'verifier_corrections':'Initial checks used an invalid file_flag accessor for normalized transcription rules, copies without the manual other/forms path marker, and omitted an explicit BibTeX format for the .asset baseline. Only verifier code changed; canonical inputs remained exact. All three 256-row parser checks passed before bibliography format refinement. Failed attempts preserved separately.','deferred':['full CLDF/DB build','compiled identities/graphs/reference survival','full suite','browser QA']}
    (PACKAGE/'canonical-source-stage-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='formatted_references'},indent=2))
    return report

if __name__=='__main__':main()
