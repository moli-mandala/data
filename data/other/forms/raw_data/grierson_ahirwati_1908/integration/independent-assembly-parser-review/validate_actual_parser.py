"""Bounded unchanged production modules in an isolated filesystem fixture.

No provider/convertor monkeypatches. Only local copied inputs/modules are written.
"""
import collections,csv,hashlib,io,json,os,re,shutil,sys,unicodedata as ud
from pathlib import Path
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1]
ROOT=next(p for p in HERE.parents if (p/'data/make_cldf.py').exists())
SOURCE=HERE.parent/'review-assembly-r1'
FIXTURE=HERE/'fixture/modules'
FORMS=FIXTURE/'data/other/forms'
FORMS.mkdir(parents=True,exist_ok=True)
(FIXTURE/'conversion').mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
status=json.loads((SOURCE/'status.json').read_text())
for name,digest in status['outputs'].items():assert sha(SOURCE/name)==digest
pins={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE/'status.json',*(SOURCE/n for n in status['outputs'])]}
csv_path=SOURCE/'combined-review-candidate.csv'
raw=list(csv.reader(csv_path.open(newline='')))
assert len(raw)==653 and all(len(r)==15 for r in raw)
assert len({r[10] for r in raw})==653
for name in ['make_cldf.py','source_meta.py']:
    shutil.copyfile(ROOT/'data'/name,FIXTURE/name)
    assert sha(FIXTURE/name)==sha(ROOT/'data'/name)
for name in ['combined-review-candidate.csv','combined-review-candidate.yaml']:
    shutil.copyfile(SOURCE/name,FORMS/name)
profile_name='grierson-ahirwati-1908-literal-preliminary'
shutil.copyfile(SOURCE/(profile_name+'.txt'),FIXTURE/'conversion'/(profile_name+'.txt'))
sys.path[:0]=[str(FIXTURE),str(ROOT/'data')]
os.chdir(FIXTURE)
import make_cldf,source_meta
from tags import GENDER_TAGS,GRAMMATICAL_TAGS
from dialects import load_dialect_aliases,normalize_dialect,dialect_tag
assert Path(make_cldf.__file__).resolve()==FIXTURE/'make_cldf.py'
assert Path(source_meta.__file__).resolve()==FIXTURE/'source_meta.py'
meta=source_meta.load()
assert len(source_meta.yaml_files())==1
assert meta.source('grierson1908ahirwati')['reference']==dict(editor='OpenAI Codex',ocr=False,etymology_provenance='source')
canonical_data=ROOT/'data'
paths=sorted((canonical_data/'data/other/forms').glob('*.yaml'))+sorted((canonical_data/'data/other/params').glob('*.yaml'))
paths += [canonical_data/'data'/n/'source.yaml' for n in source_meta.DICTIONARY_DIRS if (canonical_data/'data'/n/'source.yaml').exists()]
combined_meta=source_meta.SourceMeta([*paths,FORMS/'combined-review-candidate.yaml'])
issues,display_deltas,cases=[],[],[]
attrs={0:'lang',1:'param',3:'gloss',4:'native',5:'ipa',6:'notes',7:'source',8:'cognateset',9:'etymology',10:'entry_key',11:'variant_of_key',12:'borrowed_from_key',13:'derivation_parent_keys',14:'tags'}
def expected_display(value):
    return ud.normalize('NFC',value.strip('1234⁴5⁵67⁷,;.').lower()).replace('ṅ','ŋ')
for norm in ['NFC','NFD']:
    inputs=[list(r) for r in raw]
    for r in inputs:r[2]=ud.normalize(norm,r[2])
    fixture_path=FORMS/'combined-review-candidate.csv'
    with fixture_path.open('w',newline='') as stream:csv.writer(stream).writerows(inputs)
    errors=io.StringIO()
    parsed,stats=make_cldf.parse_file(str(fixture_path),errors,name='combined-review-candidate',file_num=999)
    if errors.getvalue():issues.append(dict(kind='conversion-errors',normalization=norm,text=errors.getvalue()))
    if len(parsed)!=653:issues.append(dict(kind='split-or-drop',normalization=norm,expected=653,actual=len(parsed)))
    dedupe_keys=[]
    for expected,actual in zip(inputs,parsed):
        key=expected[10]
        for i,attribute in attrs.items():
            if getattr(actual,attribute)!=expected[i]:issues.append(dict(kind='field-preservation',normalization=norm,key=key,index=i,attribute=attribute,expected=expected[i],actual=getattr(actual,attribute)))
        if actual.old_form!=expected[2]:issues.append(dict(kind='Original',normalization=norm,key=key,expected=expected[2],actual=actual.old_form))
        display=ud.normalize('NFC',actual.form)
        if display!=expected_display(expected[2]):issues.append(dict(kind='display-policy',normalization=norm,key=key,expected=expected_display(expected[2]),actual=display))
        if norm=='NFC' and actual.form!=expected[2]:display_deltas.append(dict(key=key,Original=expected[2],display=actual.form,expected_display=expected_display(expected[2]),reason='House lowercase/ṅ→ŋ plus generic documented boundary trimming; Original unchanged.'))
        dedupe_key=make_cldf.source_entry_dedupe_key(actual,meta)
        if dedupe_key!=key:issues.append(dict(kind='immutable-key-dedupe',normalization=norm,key=key,actual=dedupe_key))
        dedupe_keys.append((actual.lang,actual.param,actual.form,dedupe_key))
    cases.append(dict(normalization=norm,raw_rows=len(inputs),parsed_rows=len(parsed),statistics=stats,unique_cleanup_keys=len(set(dedupe_keys)),splits_or_drops=len(parsed)-len(inputs)))
shutil.copyfile(csv_path,FORMS/'combined-review-candidate.csv')
assert sha(FORMS/'combined-review-candidate.csv')==sha(csv_path)
languages={r['ID']:r for r in csv.DictReader((canonical_data/'cldf/languages.csv').open())}
aliases=load_dialect_aliases(canonical_data/'cldf/dialects.csv')
registered_tags={r['Tag'] for r in aliases.values()}
proposed=list(csv.DictReader((PACKAGE/'metadata-profile-proposal/dialects-proposed.csv').open()))
proposed_tags={r['Tag']:r for r in proposed}
unknown_tags,missing_lects,route_errors=[],{},[]
symbol_count=collections.Counter()
for r in raw:
    for token in r[14].split():
        if token.startswith('dialect:'):
            parts=token.split(':',3)
            if len(parts)!=4:issues.append(dict(kind='dialect-token-shape',key=r[10],token=token))
            elif token not in registered_tags:
                missing_lects.setdefault(token,dict(proposed_registry_row=proposed_tags.get(token),example_key=r[10],rows=0))['rows']+=1
        elif token not in GENDER_TAGS|GRAMMATICAL_TAGS:unknown_tags.append(dict(key=r[10],tag=token))
    mapped=make_cldf.change.get(r[0],r[0])
    mapped=make_cldf.unidecode.unidecode(mapped).replace('.','')
    normalized,tags=normalize_dialect(mapped,r[14],aliases,set(languages))
    if normalized not in languages:route_errors.append(dict(key=r[10],raw_language=r[0],normalized=normalized))
    symbol_count.update(r[2])
    if '\ufffd' in ''.join(r) or any(ud.category(c)=='Co' for c in ''.join(r)):issues.append(dict(kind='replacement-or-PUA',key=r[10]))
    if any(ud.category(c)=='Cc' and c not in '\t\n\r' for c in ''.join(r)):issues.append(dict(kind='control-character',key=r[10]))
regional=[r[10] for r in raw if ':Bawal-Kot-Kasam-regional-copulas:' in r[10]]
assert len(regional)==4 and all(not any(t.startswith('dialect:') for t in r[14].split()) for r in raw if r[10] in regional)
for path,digest in pins.items():assert sha(ROOT/path)==digest
for name,records in [('all-display-deltas.jsonl',display_deltas),('field-or-conversion-issues.jsonl',issues),('unicode-inventory.jsonl',[dict(character=c,codepoint=f'U+{ord(c):04X}',name=ud.name(c,'UNNAMED'),count=n) for c,n in sorted(symbol_count.items())])]:
    (HERE/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
report=dict(status='PASS isolated actual parser compatibility' if not issues and not unknown_tags and not route_errors else 'FAIL isolated actual parser compatibility',
            source_acceptance='Pending grammar/No37 metadata decisions, final graph/lect registration and whole-output audit; this compatibility result is not scholarly acceptance.',
            input_hashes=pins,production_module_hashes={n:sha(canonical_data/n) for n in ['make_cldf.py','source_meta.py']},
            method='Byte-identical production modules copied into isolated filesystem root. SourceMeta.load/yaml discovery and profile loading run normally; no parser, conversion, metadata-provider or tokenizer monkeypatch. All653 rows checked in NFC and NFD.',
            counts=dict(input_rows=653,normalization_cases=1306,field_or_conversion_issues=len(issues),unknown_tags=len(unknown_tags),unregistered_base_routes=len(route_errors),display_deltas=len(display_deltas),used_unregistered_lect_tags=len(missing_lects),regional_route_pending_rows=len(regional),phonemic_nonblank=sum(bool(r[5]) for r in raw),graph_fields_nonblank=sum(any(r[i] for i in [11,12,13]) for r in raw)),
            cases=cases,reference_flags=meta.source('grierson1908ahirwati')['reference'],canonical_plus_staged_sourceMeta_valid=True,
            unknown_tags=unknown_tags,base_route_errors=route_errors,lect_registration_needed=missing_lects,
            regional_routing_pending=dict(keys=regional,requirement='Two source-qualified historical Bawal/Kot Kasam contributions jointly attribute these four physical witnesses; blank geo/glotto; no Gurgaon tag. Final root-approved registry contribution not yet in this assembly.'),
            profile_symbols=len(symbol_count),profile_source_only=True,
            output_hashes={p.name:sha(p) for p in [HERE/'all-display-deltas.jsonl',HERE/'field-or-conversion-issues.jsonl',HERE/'unicode-inventory.jsonl']},
            canonical_changes=False)
(HERE/'actual-parser-compatibility-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status=report['status'],counts=report['counts'],cases=cases)))
