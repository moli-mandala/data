"""Review proposal routing and registered identities without compiled writes."""
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

HERE=Path(__file__).resolve().parent
OUT=HERE/'literal-integration'
DATA=HERE.parents[5]
TMP=DATA.parent/'tmp/koya-notation/metadata'
KEY='devagnanavaram-et-al2021koya'
STEMS={'wordlist':'20260828-sil-koya','narrative':'20260927-sil-koya-narrative','comparative':'20260927-sil-koya-comparative'}

def main():
    sys.path[:0]=[str(DATA),str(DATA/'tests')]
    from source_meta import SourceMeta,yaml_files
    from coordinate_policy import assert_reviewed_point
    from tags import GRAMMATICAL_TAGS, GENDER_TAGS
    import pybtex
    import pybtex.database
    TMP.mkdir(parents=True,exist_ok=True)
    paths=[]
    for kind,stem in STEMS.items():
        path=TMP/f'{stem}.yaml';shutil.copyfile(OUT/f'proposed-{kind}.yaml',path);paths.append(path)
    meta=SourceMeta([p for p in yaml_files() if p.stem not in STEMS.values()]+paths)
    languages={r['ID'] for r in csv.DictReader((DATA/'cldf/languages.csv').open())}
    dialects={r['Tag']:r for r in csv.DictReader((DATA/'cldf/dialects.csv').open())}
    seen=set();keys=set();edges=[];counts={}
    for kind,stem in STEMS.items():
        rows=list(csv.reader((OUT/f'proposed-{kind}.csv').open()))
        profile='sil-koya-literal' if kind=='wordlist' else f'sil-koya-{kind}'
        for row in rows:
            assert row[0] in languages,row[10]
            assert re.fullmatch(re.escape(KEY)+r'\[[^;\[\]]+\]',row[7]),(row[10],row[7])
            assert meta.transcription(KEY,stem,row[0])==(profile,True)
            assert meta.flag(KEY,'identity','dedupe_by_entry_key') is True
            assert row[10] not in keys;keys.add(row[10])
            if row[11]:edges.append((row[10],row[11]))
            for tag in row[14].split():
                if tag.startswith('dialect:'):
                    d=dialects[tag];assert d['Language_ID']==row[0];assert_reviewed_point(d);seen.add(tag)
                else:assert tag in GRAMMATICAL_TAGS | GENDER_TAGS,(row[10],tag)
        counts[kind]=len(rows)
    assert len(seen)==7
    assert all(a!=b and b in keys for a,b in edges)
    bib=pybtex.database.parse_file(str(DATA/'cldf/sources.bib'))
    entry=bib.entries[KEY]
    fields=json.loads((OUT/'bibliography-update-proposal.json').read_text())['fields']
    entry.fields.update(fields)
    formatted=pybtex.PybtexEngine().format_from_string(entry.to_string('bibtex'),'plain',output_backend='markdown')[3:].strip()
    assert formatted and '2021' in formatted
    report={'status':'passed_proposal_metadata_checks','rows':counts,'registered_dialects':sorted(seen),'variant_edges':len(edges),'variant_edges_closed':True,'all_source_keys_identity_protected':True,'citations_delimiter_safe':True,'formatted_proposed_reference':formatted,'scope':'Only small source registry, language/dialect registry and in-memory Bib formatting; no shared metadata or compiled outputs changed.','sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('proposed-*')) if p.is_file()}}
    (OUT/'metadata-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='sha256'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
