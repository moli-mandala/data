"""Final scoped reference/lect/profile readiness proposal; no shared writes."""
import csv,hashlib,json,sys,unicodedata as ud
from collections import Counter,defaultdict
from pathlib import Path
import yaml
from pybtex.database import parse_file
from pybtex import PybtexEngine

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
sys.path.insert(0,str(DATA))
import source_meta
from dialects import dialect_tag,normalize_dialect
from tags import GRAMMATICAL_TAGS,GENDER_TAGS
from segments.tokenizer import Tokenizer
REVISION=HERE/'review-assembly-r2'
OUT=HERE/'metadata-ready-r2'
SOURCE='grierson1908ahirwati'
PROFILE='grierson-ahirwati-1908-literal-preliminary'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,text):
    if p.exists():assert p.read_text()==text,('Immutable metadata proposal differs',p)
    else:p.write_text(text)
def main():
    status=json.loads((REVISION/'status.json').read_text())
    for name,h in status['outputs'].items():assert sha(REVISION/name)==h
    rows=list(csv.reader((REVISION/'combined-review-candidate.csv').open()))
    audit=[json.loads(s) for s in (REVISION/'combined-review-audit.jsonl').read_text().splitlines()]
    languages={r['ID']:r for r in csv.DictReader((DATA/'cldf/languages.csv').open())}
    current_lects={r['ID'] for r in csv.DictReader((DATA/'cldf/dialects.csv').open())}
    lects=list(csv.DictReader((REVISION/'dialects-proposed.csv').open()))
    assert len(lects)==7
    for r in lects:
        assert r['Language_ID'] in languages and r['ID'] not in current_lects
        assert r['Tag']==dialect_tag(r['Language_ID'],r['Source_Language_ID'],r['Name'])
        assert normalize_dialect(r['Source_Language_ID'],r['Tag'],{},set(languages))==(r['Language_ID'],r['Tag'])
        assert all(r[f]=='' for f in ('Glottocode','Latitude','Longitude','Quality'))
    tags={r['Tag'] for r in lects}
    assert all(r[0] in languages and set(r[14].split())<=GRAMMATICAL_TAGS|GENDER_TAGS|tags for r in rows)
    OUT.mkdir(exist_ok=True)
    settings=yaml.safe_load((REVISION/'combined-review-candidate.yaml').read_text());settings['file']='20260927-grierson-ahirwati.csv'
    settings_path=OUT/'20260927-grierson-ahirwati.yaml'
    save(settings_path,yaml.safe_dump(settings,sort_keys=False))
    meta=source_meta.SourceMeta([*source_meta.yaml_files(),settings_path])
    reference=meta.source(SOURCE)['reference']
    assert reference=={'editor':'OpenAI Codex','ocr':False,'etymology_provenance':'source'}
    assert sum(SOURCE in (yaml.safe_load(p.read_text()) or {}).get('sources',{}) for p in [*source_meta.yaml_files(),settings_path])==1
    assert meta.file_flag(settings['file'],'identity','dedupe_by_entry_key',False) is True
    assert meta.flag(SOURCE,'forms','split_alternates',True) is False and meta.flag(SOURCE,'gloss','grammar_tags',True) is False
    for language in {r[0] for r in rows}:assert meta.transcription(SOURCE,settings['file'],language)==(PROFILE,True)
    sourcebib=PACKAGE/'metadata-profile-proposal/references-proposed.bib'
    bib=parse_file(str(sourcebib));assert set(bib.entries)=={SOURCE}
    entry=bib.entries[SOURCE]
    entry.fields['included']='Complete source-defined Ahirwati grammar pp. 49--51; Gurgaon specimen 37 pp. 233--236; mixed Jhajjar/Rohtak specimen 38 pp. 237--239 with explanatory quotations and complete documentary translation; eligible Bagri comparison witnesses pp. 149 and 151. 653 lexical/expression rows with 1,970 audit records; 173 structural parents and all 1,059 specimen physical Roman/Native groups accounted. No Ahirwati comparative-table column is printed.'
    entry.fields['provenance']='data/other/forms/20260927-grierson-ahirwati.csv; data/other/forms/raw_data/grierson_ahirwati_1908/installed-audit.jsonl; data/other/forms/raw_data/grierson_ahirwati_1908/import_source.py (guarded installation route pending approval); frozen source-local integration/review-assembly-r2; original scope/verified-scope-manifest.json and scope/source-and-errata-review.json.'
    save(OUT/'references-proposed.bib',bib.to_string('bibtex'))
    current=parse_file(str(DATA/'cldf/sources.bib'));assert SOURCE not in current.entries
    formatted=PybtexEngine().format_from_string(bib.to_string('bibtex'),'plain',output_backend='markdown')
    assert all(term in formatted for term in ('1908','Rajasthani','Gujarati','Grierson'))
    tokenizer=Tokenizer(str(REVISION/(PROFILE+'.txt')))
    inventories=defaultdict(Counter);cases=0
    emitted={a['source_unit_key']:a for a in audit if a.get('actual_row')}
    for r in rows:
        inventories[emitted[r[10]]['kind']].update(r[2])
        expected=ud.normalize('NFC',r[2].lower()).replace('ṅ','ŋ')
        for norm in ('NFC','NFD'):
            display=ud.normalize('NFC',tokenizer(ud.normalize(norm,r[2]),column='IPA').replace(' ','').replace('#',' '))
            assert display==expected,(r[10],norm,display,expected)
            assert '�' not in display and not any(ud.category(c)=='Co' for c in display)
            cases+=1
    inventory={kind:[{'literal':c,'codepoint':f'U+{ord(c):04X}','name':ud.name(c,'UNNAMED'),'count':n} for c,n in sorted(counts.items())] for kind,counts in inventories.items()}
    save(OUT/'unicode-inventory-by-section.json',json.dumps(inventory,ensure_ascii=False,indent=2)+'\n')
    result={'status':'pass scoped metadata/reference/profile readiness; canonical installation and final acceptance pending',
      'inputs':{'review_status':sha(REVISION/'status.json'),'candidate':sha(REVISION/'combined-review-candidate.csv'),'audit':sha(REVISION/'combined-review-audit.jsonl'),
        'lects':sha(REVISION/'dialects-proposed.csv'),'profile':sha(REVISION/(PROFILE+'.txt')),'raw_references':sha(sourcebib),'language_registry':sha(DATA/'cldf/languages.csv')},
      'outputs':{p.name:sha(p) for p in OUT.iterdir() if p.name!='status.json'},
      'rows':653,'audit_records':1970,'lects':7,'source_declarations':1,'registered_base_IDs':sorted({r[0] for r in rows}),
      'reference_flags':reference,'formatted_reference':formatted,'profile_NFC_NFD_cases':cases,'Phonemic_blank':True,
      'uncertain_rows':sum('uncertain' in r[14].split() for r in rows),'canonical_changes':False,
      'pending':['Root final actual-output original audit/metadata acceptance','Guarded importer command and exact staged activation by installer owner','Seven proposed historical lects and source Bib registration','Canonical parser/dedupe/no-op/identity checks','Compiled/full-suite/browser gates deferred']}
    save(OUT/'status.json',json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
