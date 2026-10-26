"""Immutable whole-source review assembly, never canonical installation or acceptance."""
import argparse,copy,csv,hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
import yaml
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
sys.path.insert(0,str(DATA))
from tags import GRAMMATICAL_TAGS,GENDER_TAGS
SOURCE='grierson1908ahirwati'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def records(p):return [json.loads(s) for s in p.read_text().splitlines()]
def save(p,text):
    if p.exists():assert p.read_text()==text,('Immutable output differs; use a new revision directory',str(p))
    else:p.write_text(text)
def main(out):
    out=out.resolve();assert out.parent==HERE.resolve(),'Output must be a new integration revision'
    out.mkdir(exist_ok=True)
    pins={}
    def consume(name):
        p=PACKAGE/name;pins[name]=sha(p);return p
    def read(name):return records(consume(name))
    def proof(name,prefix=''):
        p=consume(name);a=json.loads(p.read_text());assert a['status']=='pass',name
        for field in ['inputs','input_hashes','input_sha256']:
            values=a.get(field)
            if isinstance(values,dict):
                for n,h in values.items():
                    target=PACKAGE/(prefix+n)
                    assert sha(target)==h,('Acceptance input drift',name,n)
                    pins[prefix+n]=h
        return a
    languages={r['ID']:r['Name'] for r in csv.DictReader((DATA/'cldf/languages.csv').open())}
    dialect_rows=list(csv.DictReader(consume('metadata-profile-proposal/dialects-proposed.csv').open()))
    dialects={r['ID']:r for r in dialect_rows};dialect_tags={r['Tag'] for r in dialect_rows}
    candidate=[];audit=[];counts=Counter();physical_maps=defaultdict(set)
    def emit(row,section,record,accepted_metadata,typed=None,physical=None):
        assert len(row)==15 and row[2] and row[10] and not row[1] and not row[5]
        assert row[0] in languages.values(),('Unknown base display',row[0])
        assert set(row[14].split())<=GRAMMATICAL_TAGS|GENDER_TAGS|dialect_tags
        assert not any(row[i] for i in [8,11,12,13]),'No pending graph is silently serialized'
        candidate.append(row);counts[section]+=1
        audit.append({'source_unit_key':row[10],'kind':section,'entry_keys':[row[10]],
          'disposition':'review_candidate_emission','actual_row':row,'metadata_accepted_for_source_stage':accepted_metadata,
          'accepted_for_installation':False,'typed_uncertainties':typed or [],'source_record':record})
        for script,key in physical or []:physical_maps[(section.split('-')[0],script,key)].add(row[10])
    def rich(language,key,form,gloss,page,tags,notes='',native='',dialect=None,etymology=''):
        assert language in languages,language
        if dialect:
            assert dialects[dialect]['Language_ID']==language
            tags=[*tags,dialects[dialect]['Tag']]
        return [languages[language],'',form,gloss,native,'',notes,f'{SOURCE}[p. {page}]','',etymology,key,'','','',' '.join(dict.fromkeys(tags))]
    grammar_proof=json.loads(consume('grammar/root-reviewed-grammar-acceptance.json').read_text())
    for name,h in grammar_proof['input_sha256'].items():assert sha(PACKAGE/name)==h; pins[name]=h
    grammar=read('grammar/child-witnesses-reviewed.jsonl')
    assert len(grammar)==98 and sha(PACKAGE/'grammar/child-witnesses-reviewed.jsonl')=='5ac645b13ae1a3132ac094431fbce9376037705f93c0c59d9861e6e7ef999ee7'
    proposed=read('integration/grammar-metadata-proposal-r2/all98-keyed-grammar-metadata-proposal.jsonl')
    assert sha(PACKAGE/'integration/grammar-metadata-proposal-r2/all98-keyed-grammar-metadata-proposal.jsonl')=='89025ac3ebf122a8db117de76dd408116eaf7f88c499f46c98cf95fc007ff9a3'
    metadata={r['source_unit_key']:r for r in proposed};assert set(metadata)=={r['witness_key'] for r in grammar}
    for r in grammar:
        d=metadata[r['witness_key']]
        assert (r['source_form'],r['source_gloss'],r['language_label'],r['disposition'])==(d['exact_form_guard'],d['exact_gloss_guard'],d['exact_language_label_guard'],d['exact_disposition_guard'])
        if d['proposed_language_id'] is None:
            audit.append({'source_unit_key':r['witness_key'],'kind':'grammar-documentary','entry_keys':[],
              'disposition':r['disposition'],'source_record':r,'accepted_for_installation':False});continue
        dialect='lsi1908_ahirwati_gurgaon' if r['language_label']=='Ahirwati' and 'Bawal-Kot-Kasam' not in r['witness_key'] else None
        row=rich(d['proposed_language_id'],r['witness_key'],r['source_form'],r['source_gloss'],r['printed_page'],d['proposed_tags'],
           '; '.join(d['proposed_notes']),dialect=dialect,etymology=d['proposed_etymology'])
        emit(row,'grammar',{'reviewed_literal':r,'pending_keyed_metadata':d},False,r['typed_uncertainties'])
    n37='specimens/specimen37/independent-review/projection-revision3/'
    projection=proof(n37+'root-projection-acceptance.json')
    source=consume(n37+'candidate.csv');assert sha(source)==projection['output_sha256']=='a8141e9f2d2291d7f50929b3cb9fc3574cbf77669a2a19a92f4b4ee0d670fd99'
    with source.open() as stream:n37rows=list(csv.reader(stream))
    n37audit=read(n37+'candidate-audit.jsonl');assert len(n37rows)==len(n37audit)==390
    for row,a in zip(n37rows,n37audit):
        assert row==a['row'] and row[10]==a['source_unit_key']
        physical=[('Roman',k) for k in a.get('roman_physical_keys',[])]+[('Native',k) for k in a.get('native_physical_keys',[])]
        emit(row,'No37-'+a['kind'],a,False,a.get('typed_uncertainties',[]),physical)
    n38='integration/specimen38-reviewed-root-r3/'
    derivative=json.loads(consume(n38+'status.json').read_text())
    for name,h in derivative['inputs'].items():assert sha(PACKAGE/name)==h;pins[name]=h
    for name,h in derivative['outputs'].items():assert sha(PACKAGE/(n38+name))==h
    atoms=read(n38+'atomic-emissions-reviewed.jsonl');whole=read(n38+'expression-emissions-reviewed.jsonl')
    assert len(atoms)==132 and len(whole)==23
    for a in atoms:
        row=rich(a['language'],a['emission_key'],a['emission_form'],a['emission_gloss'],a['printed_page'],a['tags'],
          a['emission_notes'],native=a['emission_native'],dialect=a['proposed_dialect_id'])
        emit(row,'No38-atomic',a,True,[*a['typed_uncertainties'],*a['typed_metadata_uncertainties']], [('Roman',k) for k in a['physical_source_keys']])
    for a in whole:
        assert a['key']==a['first_word_key']+':expression'
        row=rich(a['language'],a['key'],a['emission_form'],a['emission_gloss'],238,a['tags'],native=a['emission_native'],dialect=a['proposed_dialect_id'])
        row[7]=f'{SOURCE}[p. 238, specimen 38]'
        emit(row,'No38-expression',a,True,a['typed_uncertainties'], [('Roman',k) for k in a['physical_source_keys']])
    quote_proof=proof('specimen38/root-documentary-review/root-reviewed-quoted-metadata-acceptance.json')
    quotations=read('specimen38/native-review/quoted-emissions-reviewed-derivative.jsonl');assert len(quotations)==18
    for q in quotations:
        dialect='lsi1908_ahirwati_rohtak_jhajjar' if q['language']=='mewati_jhambaus' else None
        row=rich(q['language'],q['source_unit_key'],q.get('emission_form',q.get('raw_form')),
          q.get('emission_gloss',q.get('raw_gloss','')),q['printed_page'],q['proposed_tags'],q['notes'],dialect=dialect)
        emit(row,'quoted-commentary',q,True,q.get('typed_uncertainties',[]))
    outside_proof=proof('outside-review/root-reviewed-comparison-continuity.json',prefix='outside-review/')
    outside=read('outside-review/child-witnesses-reviewed.jsonl');assert len(outside)==62
    for r in outside:
        eligible=r['reviewed_disposition']=='eligible source-comparand emission proposal'
        if not eligible:
            audit.append({'source_unit_key':r['key'],'kind':'outside-control','entry_keys':[],
              'disposition':r['reviewed_disposition'],'source_record':r,'accepted_for_installation':False});continue
        tags=r['proposed_tags'][:]
        if r['typed_uncertainties'] and 'uncertain' not in tags:tags.append('uncertain')
        row=rich(r['proposed_language_id'],r['key'],r['source_form'],r['source_gloss'],r['printed_page'],tags,
           r['source_note'],native=r.get('raw_native') or '',dialect=r['proposed_dialect_id'])
        emit(row,'outside-eligible',r,True,r['typed_uncertainties'])
    extra=read('outside-review/Hissar-whole-physical-atoms.jsonl');assert len(extra)==2
    for r in extra:audit.append({'source_unit_key':r['key']+':physical-accounting','kind':'outside-physical-components','entry_keys':[],
      'disposition':'Physical provenance only; no additional separately authorized lexical rows','source_record':r,'accepted_for_installation':False})
    # Raw physical records are conserved separately from overlapping lexical/
    # expression views. No broad Persian span is inserted as narrow Native.
    physical_inputs=[
      ('No37','Roman','specimens/specimen37/roman-atoms-first-reading.jsonl','key'),
      ('No37','Native','specimens/specimen37/native-atoms-first-reading.jsonl','key'),
      ('No38','Roman','specimen38/root-reviewed-revision/p238-atomic-reviewed.jsonl','source_unit_key'),
      ('No38','Native','specimen38/native-review/native-atoms-first-reading.jsonl','key')]
    physical_counts={}
    for section,script,path,keyfield in physical_inputs:
        raw=read(path);physical_counts[f'{section}-{script}']=len(raw)
        for r in raw:
            mapped=sorted(physical_maps[(section,script,r[keyfield])])
            if script=='Roman':assert mapped,('Unaccounted Roman physical token',r[keyfield])
            audit.append({'source_unit_key':r[keyfield]+':physical-accounting','kind':section+'-physical-'+script,
              'physical_key':r[keyfield],'entry_keys':mapped,'disposition':'Source physical witness accounted; overlapping whole-expression views deliberate' if mapped else 'Broad Native physical provenance only; narrow word equivalence not certified',
              'source_record':r,'accepted_for_installation':False})
    parents=read('scope/physical-source-unit-census.jsonl');assert len(parents)==173
    for r in parents:
        key=r.get('source_unit_key',r.get('key'))
        assert key
        audit.append({'source_unit_key':key+':structural-accounting','kind':'structural-parent','entry_keys':[],
          'disposition':'Complete source layout/documentary scope retained; lexical decisions separately keyed','source_record':r,'accepted_for_installation':False})
    # Free translation/commentary remains documentary, without manufacturing
    # separate lexical glosses for interlinear word rows.
    documentary=read('specimen38/documentary-first-reading.jsonl')
    for r in documentary:
        key=r.get('source_unit_key',r.get('key'));assert key
        audit.append({'source_unit_key':key+':documentary-accounting','kind':'documentary-source-block','entry_keys':[],
          'disposition':'Source documentary/free-translation context, not independent lexical head','source_record':r,'accepted_for_installation':False})
    assert len(candidate)==len({r[10] for r in candidate})==653
    assert sum(counts[k] for k in counts if k.startswith('No37-'))==390
    assert counts['grammar']==75 and counts['No38-atomic']==132 and counts['No38-expression']==23
    assert counts['quoted-commentary']==18 and counts['outside-eligible']==15
    assert len(audit)==len({r['source_unit_key'] for r in audit}), 'Duplicate audit identity'
    import io
    buffer=io.StringIO(newline='');csv.writer(buffer).writerows(candidate)
    save(out/'combined-review-candidate.csv',buffer.getvalue())
    save(out/'combined-review-audit.jsonl',''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audit))
    settings=yaml.safe_load(consume('metadata-profile-proposal/20260927-grierson-ahirwati.yaml').read_text())
    settings['file']='combined-review-candidate.csv'
    settings['sources'][SOURCE]['forms']={'split_alternates':False}
    settings['sources'][SOURCE]['gloss']={'grammar_tags':False}
    save(out/'combined-review-candidate.yaml',yaml.safe_dump(settings,sort_keys=False,allow_unicode=True))
    profile=consume('metadata-profile-proposal/grierson-ahirwati-1908-literal-preliminary.txt')
    save(out/'grierson-ahirwati-1908-literal-preliminary.txt',profile.read_text())
    pins[str(Path(__file__).relative_to(PACKAGE))]=sha(Path(__file__))
    status={'status':'Whole source-local assembly review scaffold; not accepted or installed',
      'inputs':pins,'outputs':{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='status.json'},
      'counts':{'rows':653,'audit_records':len(audit),'sections':dict(counts),'physical_records':physical_counts,'structural_parents':173,
        'grammar_interpretations':98,'grammar_documentary':19,'unattested_hypotheses':4,'outside_controls':47,'outside_unauthorized_physical_components':2},
      'language_counts':dict(Counter(r[0] for r in candidate)),
      'pending':['Root full98 grammar tag/graph/routing acceptance','Root complete390 No37 residual metadata acceptance','Final combined profile/display decision and full symbol parser coverage',
        'Final combined original-output audit','No38 broad Native documentary review reconciliation','Complete settings/reference/lect registration readiness','Guarded installation approval and canonical checks','Full compiled/full-suite/browser gates deferred'],
      'accepted_for_installation':False,'canonical_changes':False,
      'notes':'Proposed grammar tags are visibly pending in audit; all graph columns stay blank until exact root graph acceptance. Four regional copulas have no guessed Gurgaon dialect.'}
    save(out/'status.json',json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(status,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'review-assembly-r1');a=p.parse_args();main(a.output)
