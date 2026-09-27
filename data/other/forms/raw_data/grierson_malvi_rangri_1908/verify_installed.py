"""Read-only scoped verification after installation; never builds a database.

From data/: .venv/bin/python data/other/forms/raw_data/grierson_malvi_rangri_1908/verify_installed.py --identity-before SHA256
Capture data/form-identities.csv SHA256 immediately before installing, then pass it here.
Prints a compact JSON report to stdout. No generated CLDF files are written.
"""
import argparse,csv,hashlib,io,json,re,sys
from pathlib import Path
P=Path(__file__).resolve().parent
DATA=P.parents[4]
sys.path.insert(0,str(DATA))


def verify(identity_before):
    assert Path.cwd().resolve()==DATA, 'Run from data repository root'
    import make_cldf, make_refs, source_meta, profile_policy, tags
    from unify_cldf import citation_keys
    from segments import Tokenizer
    import pybtex,pybtex.database
    source='grierson1908malvirangri';stem='20260925-grierson-malvi-rangri';profile='grierson-malvi-rangri-1908'
    canonical=DATA/f'data/other/forms/{stem}.csv'
    assert canonical.read_bytes()==(P/'proposal.csv').read_bytes()
    assert (P/'audit.jsonl').read_bytes()==(P/'proposal-audit.jsonl').read_bytes()
    registered=DATA/f'conversion/{profile}.txt'
    assert registered.read_bytes()==(P/'proposal-profile.txt').read_bytes()
    identity_path=DATA/'data/form-identities.csv'
    assert hashlib.sha256(identity_path.read_bytes()).hexdigest()==identity_before,'Durable identity state changed'
    meta=source_meta.SourceMeta()  # Validates YAML schema and unique source ownership repo-wide.
    assert meta.transcription(source,canonical,'Malw')[0]==profile
    assert meta.flag(source,'identity','dedupe_by_entry_key') is True
    assert meta.flag(source,'forms','split_alternates') is False
    assert profile not in profile_policy.audit({})
    langs={x['ID']:x for x in csv.DictReader((DATA/'cldf/languages.csv').open())}
    dialects={x['ID']:x for x in csv.DictReader((DATA/'cldf/dialects.csv').open())}
    assert langs['Malw']['Glottocode']=='malv1243'
    dialect=dialects['lsi1908-malvi-rangri'];assert dialect['Language_ID']=='Malw'
    assert not dialect['Latitude'] and not dialect['Longitude']
    assert 'Junior Branch' in str(dialect)
    rows=list(csv.reader(canonical.open()));assert len(rows)==1170
    by_key={r[10]:r for r in rows};assert len(by_key)==1170
    tokenizer=Tokenizer(str(registered))
    for row in rows:
        assert len(row)==15 and row[0]=='Malw'
        assert set(row[14].split())<=tags.GRAMMATICAL_TAGS|tags.GENDER_TAGS|{dialect['Tag']}
        assert set(make_refs.source_ids(row[7]))==citation_keys(row[7])=={source}
        assert not re.search(r'\[[^\]]*;',row[7])
        assert '�' not in tokenizer(row[2],column='IPA')
    errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(canonical),errors,name=stem)
    assert not errors.getvalue() and stats['converted']==len(parsed)==1170
    for r in parsed:
        original=by_key[r.entry_key]
        assert r.old_form==original[2] and r.native==original[4] and r.ipa==original[5]
        assert r.source==original[7]
        assert r.form==tokenizer(original[2],column='IPA').replace(' ','').replace('#',' ')
    bibliography=pybtex.database.parse_file(str(DATA/'cldf/sources.bib'))
    entry=bibliography.entries[source]
    formatted=pybtex.PybtexEngine().format_from_string(entry.to_string('bibtex'),'plain',output_backend='markdown')[3:].strip()
    assert formatted and '1908' in formatted and 'Grierson' in formatted
    assert entry.fields.get('included') and 'five secure' not in entry.fields['included']
    assert stem in entry.fields['provenance'] and 'audit.jsonl' in entry.fields['provenance']
    assert entry.fields.get('jambu_editor') and entry.fields.get('ocr','').lower()=='no'
    assert meta.source(source)['reference']['ocr'] is False
    assert meta.source(source)['reference']['etymology_provenance']=='none'
    return {'status':'passed-source-stage-only','rows':len(rows),'native_rows':sum(bool(r[4]) for r in rows),'reference_short':make_refs.create_short_ref(entry),'reference_formatted':formatted,'identity_sha256':identity_before,'csv_sha256':hashlib.sha256(canonical.read_bytes()).hexdigest(),'deferred':['full CLDF build','compiled identity/alias/graph/reference output verification','full suite','browser refresh not requested']}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--identity-before',required=True);args=parser.parse_args()
    print(json.dumps(verify(args.identity_before),ensure_ascii=False,indent=2))
