"""Actual installed parser/metadata check, executable only after activation."""
import csv
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unicodedata
from pathlib import Path

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
sys.path.insert(0,str(DATA))
import source_meta
import make_cldf
from pybtex.database import parse_file as parse_bib
from pybtex.style.formatting.plain import Style
from tags import GRAMMATICAL_TAGS,GENDER_TAGS
spec=importlib.util.spec_from_file_location('mullu_approved_canonical_guard',HERE/'install_source.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
FIELDS=('lang','param','old_form','gloss','native','ipa','notes','source','cognateset','etymology','entry_key','variant_of_key','borrowed_from_key','derivation_parent_keys','tags')
STEM='20260927-nair-mullu-kurumba'


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main(initial_identity_check=False):
    assert sha(DATA/guard.TARGETS['csv'])==sha(PACKAGE/'proposed.csv')
    assert sha(DATA/guard.TARGETS['yaml'])==sha(HERE/(STEM+'.yaml'))
    assert sha(DATA/guard.TARGETS['audit'])==sha(HERE/'accepted-audit-proposal.jsonl')
    rows=list(csv.reader((DATA/guard.TARGETS['csv']).open(newline='')))
    assert len(rows)==len({r[10] for r in rows})==135
    source_meta.load.cache_clear();meta=source_meta.load()
    assert meta.file_flag(STEM+'.csv','identity','dedupe_by_entry_key') is True
    rule=meta.transcription_rule('nair2013mullukurumba',STEM+'.csv','Mullu Kurumba')
    assert rule['convert'] is False and rule['preserve_hyphens'] is True
    for key in ('nair2013mullukurumba','robertsathyajoseph1982mullukurumba'):
        assert meta.flag(key,'reference','editor')=='OpenAI Codex'
        assert meta.flag(key,'reference','ocr') is False
        assert meta.flag(key,'reference','etymology_provenance')==('source' if key=='nair2013mullukurumba' else 'none')
    registry={r['ID']:r for r in csv.DictReader((DATA/'cldf/languages.csv').open(newline=''))}
    assert registry['Mullu Kurumba']['Glottocode']=='mull1244'
    assert {r[0] for r in rows}=={'Mullu Kurumba','Malayalam'}
    assert all(set(r[14].split())<=GRAMMATICAL_TAGS|GENDER_TAGS for r in rows)
    keys={r[10] for r in rows}
    assert sum(bool(r[11]) for r in rows)==11 and all(not r[11] or r[11] in keys for r in rows)
    audit=[json.loads(s) for s in (DATA/guard.TARGETS['audit']).read_text().splitlines()]
    raw=[json.loads(s) for s in (PACKAGE/'proposal-audit.jsonl').read_text().splitlines()]
    assert len(audit)==65 and [a['source_record'] for a in audit]==raw
    references=[k for a in audit for k in a['entry_keys']]
    assert len(references)==141 and set(references)==keys
    checks={}
    for normalization in ('NFC','NFD'):
        expected=[[unicodedata.normalize(normalization,v) if i==2 else v for i,v in enumerate(r)] for r in rows]
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            path=DATA/guard.TARGETS['csv'] if normalization=='NFC' else Path(temporary)/(STEM+'.csv')
            if normalization=='NFD':
                with path.open('w',newline='') as stream:csv.writer(stream).writerows(expected)
            errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(path),errors,name=STEM)
            assert not errors.getvalue(),errors.getvalue()
            assert len(parsed)==135 and stats=={'converted':0,'for_conversion':0}
            for actual,wanted in zip(parsed,expected):
                assert [getattr(actual,f) for f in FIELDS]==wanted,actual.entry_key
                assert unicodedata.normalize('NFC',actual.form)==unicodedata.normalize('NFC',wanted[2])
            assert len({(r.lang,r.param,r.form,make_cldf.source_entry_dedupe_key(r,meta)) for r in parsed})==135
            checks[normalization]=dict(rows=135,all15fields_exact=True,initial_dedupe_keys=135,zero_conversion=True)
    identity=DATA/'data/form-identities.csv'
    before=sha(identity)
    if initial_identity_check:
        assert before==json.loads((HERE/'root-acceptance.json').read_text())['durable_identity_sha256']
    assert guard.main(install=True)['canonical_mutations'] is False
    assert sha(identity)==before
    baseline=parse_bib(str(HERE/'baselines/bib.asset'),bib_format='bibtex').entries
    current=parse_bib(str(DATA/'cldf/sources.bib')).entries
    added=parse_bib(str(HERE/'staged-references.bib')).entries
    assert len(baseline)==630 and all(current[k]==v for k,v in baseline.items())
    assert all(current[k]==v for k,v in added.items())
    rendered={e.key:e.text.render_as('plaintext') for e in Style().format_entries([current[k] for k in added])}
    assert 'Tribal Languages of Kerala' in rendered['nair2013mullukurumba']
    assert 'A Descriptive Study of Mullu Kurumba' in rendered['robertsathyajoseph1982mullukurumba']
    report=dict(status='Canonical source-stage checks pass; full compiled/browser gates deferred',checks=checks,
        rows=135,audit_groups=65,scoped_witness_records=164,commentary_reuses=6,variants=11,source_reference_flags=2,
        canonical_sha256={role:sha(DATA/guard.TARGETS[role]) for role in ('csv','yaml','audit')},
        actual_canonical_source_meta=True,parser_monkeypatch=False,repeat_install='no-op; current own assets and shared membership verified',
        prior_bibliography_entries=630,all_prior_bibliography_semantics_unchanged=True,new_references_rendered=rendered,
        durable_identity_sha256=before,identity_unchanged_during_check=True,initial_identity_check=initial_identity_check,
        deferred=['full compiled build','full suite','compiled identity/graph/reference checks','browser QA'])
    (PACKAGE/'canonical-source-stage-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report));return report


if __name__=='__main__':main(initial_identity_check=True)
