"""Apply exact root-approved layers to a new immutable source-local assembly."""
import copy,csv,hashlib,io,json,os,sys
from collections import Counter
from pathlib import Path
import yaml

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
ROOT=DATA.parent
sys.path.insert(0,str(DATA))
from tags import GRAMMATICAL_TAGS,GENDER_TAGS
OLD=HERE/'review-assembly-r1'
OUT=HERE/'review-assembly-r2'
RECEIPTS={
 'integration/independent-grammar-metadata-review/root-grammar-metadata-graph-acceptance.json':'7efd2522848622371525c593d200d666c0fb9bd65e38c3cc58ba299f110dc5ca',
 'specimens/specimen37/residual-peer-review/root-residual-metadata-acceptance.json':'4f23786359463695ca0ada127827576dad515a508a5adf68e7d9d29cb0eb9a0c',
 'specimen38/independent-documentary-review/root-documentary-acceptance.json':'0ec92da8946d0c06e9eba4431cf38dde98bd5e22f3a9446834f21d756afb51be'}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def records(p):return [json.loads(s) for s in p.read_text().splitlines()]
def rows(p):return list(csv.reader(p.open()))
def save(p,text):
    if p.exists():assert p.read_text()==text,('Immutable revision differs',str(p))
    else:p.write_text(text)
def rowdigest(row):return hashlib.sha256(json.dumps(row,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def main():
    pins={};receipts={}
    def pin(p):
        assert p.is_relative_to(ROOT)
        rel=os.path.relpath(p,PACKAGE);pins[rel]=sha(p);return p
    def read(rel):return records(pin(PACKAGE/rel))
    for name,h in RECEIPTS.items():
        path=pin(PACKAGE/name);assert pins[name]==h
        r=json.loads(path.read_text());assert r['status'].startswith('accepted');receipts[name]=r
        for rel,digest in r['input_hashes'].items():
            p=ROOT/rel;assert sha(p)==digest,rel;pin(p)
    oldstatus=json.loads(pin(OLD/'status.json').read_text())
    for name,h in oldstatus['outputs'].items():assert sha(pin(OLD/name))==h
    for name,h in oldstatus['inputs'].items():assert sha(pin(PACKAGE/name))==h
    before=rows(OLD/'combined-review-candidate.csv');candidate=copy.deepcopy(before)
    by={r[10]:r for r in candidate};assert len(by)==653
    audit=records(OLD/'combined-review-audit.jsonl');audit_by={a['source_unit_key']:a for a in audit}
    grammar_receipt=receipts[next(n for n in RECEIPTS if 'grammar-metadata-graph' in n)]
    dialect_path=pin(PACKAGE/'metadata-profile-proposal/dialects-proposed.csv')
    dialects=list(csv.DictReader(dialect_path.open()));fields=list(dialects[0])
    dialects+=copy.deepcopy(grammar_receipt['regional_lects'])
    assert len(dialects)==len({r['ID'] for r in dialects})==7
    lect_tags={r['Tag'] for r in dialects}
    grammar=read('integration/grammar-metadata-proposal-r2/all98-keyed-grammar-metadata-proposal.jsonl')
    grammar_by={r['source_unit_key']:r for r in grammar}
    refinements=read('integration/independent-grammar-metadata-review/keyed-tag-scope-proposals.jsonl')
    assert len(refinements)==5
    for d in refinements:
        g=grammar_by[d['source_unit_key']]
        assert (g['exact_form_guard'],g['exact_gloss_guard'],g['proposed_tags'],g['proposed_notes'])==(d['exact_form_guard'],d['exact_gloss_guard'],d['old_tags'],d['old_notes'])
        g['historical_proposed_metadata']={'tags':g['proposed_tags'],'notes':g['proposed_notes']}
        g['proposed_tags']=d['new_tags'];g['proposed_notes']=d['new_notes'];g['accepted_peer_refinement']=d
    for key,g in grammar_by.items():
        a=audit_by[key]
        a['historical_assembly_record']=copy.deepcopy(a)
        if g['proposed_language_id'] is None:continue
        row=by[key]
        assert (row[2],row[3])==(g['exact_form_guard'],g['exact_gloss_guard'])
        tags=[t for t in row[14].split() if t.startswith('dialect:')]
        if key in grammar_receipt['regional_keys']:
            assert tags==[]
            tags=[r['Tag'] for r in grammar_receipt['regional_lects']]
        row[14]=' '.join(dict.fromkeys([*g['proposed_tags'],*tags]))
        row[6]='; '.join(g['proposed_notes']);row[9]=g['proposed_etymology']
        a['source_record']={'reviewed_literal':a['source_record']['reviewed_literal'],'accepted_keyed_metadata':g}
        a['metadata_accepted_for_source_stage']=True
        a['root_approval']=next(n for n in RECEIPTS if 'grammar-metadata-graph' in n)
    graph=read('integration/grammar-metadata-proposal-r2/source-graph-proposal.jsonl')
    assert Counter(g['relation'] for g in graph)=={'variant':12,'component':1}
    for g in graph:
        child=by[g['child_key']];parents=[by[k] for k in g['parent_keys']]
        assert child[2]==g['exact_child_form_guard'] and [p[2] for p in parents]==g['exact_parent_form_guards']
        assert all(p[0]==child[0] for p in parents)
        index=11 if g['relation']=='variant' else 13
        assert child[index]=='';child[index]='|'.join(g['parent_keys'])
        audit_by[g['child_key']]['accepted_graph_claim']=copy.deepcopy(g)
        audit_by[g['child_key']]['accepted_graph_claim']['accepted']=True
    # No37: exact initial projection -> reviewed r4 -> 109 accepted peer deltas.
    n37='specimens/specimen37/'
    old37=rows(pin(PACKAGE/(n37+'independent-review/projection-revision3/candidate.csv')))
    r4=rows(pin(PACKAGE/(n37+'independent-metadata-review/residual-revision4-proposal/candidate.csv')))
    allr4=read(n37+'independent-metadata-review/residual-revision4-proposal/all390-keyed-decisions.jsonl')
    protected=[i for i in range(15) if i not in (6,7,14)]
    assert len(old37)==len(r4)==len(allr4)==390
    for old,new,d in zip(old37,r4,allr4):
        assert old==d['old_row'] and new==d['proposed_row']
        assert all(old[i]==new[i] for i in protected)
        assert by[old[10]]==old
    peer=read(n37+'residual-peer-review/guarded-refinement-proposals.jsonl');assert len(peer)==109
    r4by={r[10]:r for r in r4}
    for d in peer:
        row=r4by[d['source_unit_key']]
        assert (row[2],row[3])==(d['exact_form_guard'],d['exact_gloss_guard'])
        for col,change in d['changes'].items():
            col=int(col);assert col in (6,7,14) and row[col]==change['old'],(d['source_unit_key'],col)
            row[col]=change['new']
    r4decisions={d['source_unit_key']:d for d in allr4};peer_by={d['source_unit_key']:d for d in peer}
    for row in r4:
        key=row[10];by[key][:]=row
        a=audit_by[key];a['historical_assembly_record']=copy.deepcopy(a)
        a['accepted_metadata_layers']={'source_class_review':r4decisions[key],'peer_refinement':peer_by.get(key)}
        a['metadata_accepted_for_source_stage']=True
        a['root_approval']=next(n for n in RECEIPTS if 'residual-metadata' in n)
    # Preserve the historical owner's paraphrase; the field explicitly called
    # exact_keyed_decision must use the authoritative receipt's exact packet.
    n38receipt_path=pin(PACKAGE/'specimen38/independent-metadata-review/root-metadata-boundary-acceptance-r3.json')
    assert sha(n38receipt_path)=='f95fc51bb6c057bd4a9af28369fde0865995e03d2b6d4f1b1be0ef1156301889'
    n38receipt=json.loads(n38receipt_path.read_text())
    key='lsi9.2:p238:aligned-line:10:word:3'
    decision=next(d for d in n38receipt['decisions'] if d['source_unit_key']==key)
    a=audit_by[key];record=a['source_record']
    assert record['emission_form']==decision['exact_form_guard'] and record['emission_gloss']==decision['exact_gloss_guard']
    assert record['tags']==decision['proposed_tags']
    record['historical_metadata_annotation']={'typed_metadata_uncertainties':copy.deepcopy(record['typed_metadata_uncertainties']),
      'exact_keyed_decision':copy.deepcopy(record['root_metadata_review']['exact_keyed_decision'])}
    record['typed_metadata_uncertainties']=copy.deepcopy(decision['additional_scope_holds'])
    record['root_metadata_review']['exact_keyed_decision']=copy.deepcopy(decision)
    a['typed_uncertainties']=[*record['typed_uncertainties'],*record['typed_metadata_uncertainties']]
    # Use registered base IDs, avoiding later cleanup's ambiguous name aliases.
    languages=list(csv.DictReader(pin(DATA/'cldf/languages.csv').open()))
    ids={r['ID'] for r in languages};name_to_id={'Ahirwadi':'mewati_jhambaus','Haryanvi':'kaithal','Punjabi':'P','Mewati':'mewati_akera','Khandesi':'Khandesi','Bagri':'bagri_fatehabad'}
    for row in candidate:
        row[0]=name_to_id.get(row[0],row[0]);assert row[0] in ids,row[0]
        assert set(row[14].split())<=GRAMMATICAL_TAGS|GENDER_TAGS|lect_tags
        audit_by[row[10]]['actual_row']=row
    native_status=json.loads(pin(HERE/'specimen38-native-provenance-r1/status.json').read_text())
    assert native_status['status']=='pass'
    for name,h in native_status['inputs'].items():assert sha(pin(PACKAGE/name))==h
    for name,h in native_status['outputs'].items():assert sha(pin(HERE/'specimen38-native-provenance-r1'/name))==h
    native=read('integration/specimen38-native-provenance-r1/native-atoms-reviewed.jsonl')
    for r in native:
        a=audit_by[r['key']+':physical-accounting']
        assert a['kind']=='No38-physical-Native' and a['entry_keys']==[]
        a['historical_assembly_record']=copy.deepcopy(a);a['source_record']=r
        a['disposition']='Accepted qualified physical Native provenance only; narrow lexical equivalence not certified'
        a['root_approval']='specimen38/independent-native-review/root-native-provenance-acceptance.json'
    for r in read('integration/specimen38-native-provenance-r1/broad-script-spans-reviewed.jsonl'):
        audit.append({'source_unit_key':r['key']+':provenance-accounting','kind':'No38-broad-script-provenance','entry_keys':[],
          'disposition':'Qualified broad script provenance, no narrow Native assignment','source_record':r,'accepted_for_installation':False})
    census=read('scope/root-current-census-reconciliation-r2.jsonl')
    for r in census:
        a=audit_by[r['physical_census_key']+':structural-accounting']
        a['historical_assembly_record']=copy.deepcopy(a);a['source_record']=r
    # Replace old documentary views, retaining their history under current text.
    historical_docs={a['source_unit_key']:a for a in audit if a['kind']=='documentary-source-block'}
    audit=[a for a in audit if a['kind']!='documentary-source-block']
    docs=read('specimen38/independent-documentary-review/documentary-reviewed-proposal.jsonl');assert len(docs)==5
    pin(PACKAGE/'scope/p233-specimen-introduction-first-reading.json')
    for r in docs:
        key=r['source_unit_key']+':documentary-accounting'
        audit.append({'source_unit_key':key,'kind':'documentary-source-block','entry_keys':[],
          'disposition':'Accepted documentary text/free-translation context; no synthetic lexical alignment',
          'source_record':r,'historical_assembly_record':historical_docs.get(key),
          'root_approval':next(n for n in RECEIPTS if 'documentary-acceptance' in n),'accepted_for_installation':False})
    assert len(candidate)==653 and [r[10] for r in candidate]==[r[10] for r in before]
    for old,new in zip(before,candidate):
        assert all(old[i]==new[i] for i in (1,2,3,4,5,8,9,10,12)),new[10]
    assert len(audit)==len({a['source_unit_key'] for a in audit})
    assert {k for a in audit for k in a['entry_keys']}==set(by)
    adjacency={r[10]:[p for p in [r[11],*r[13].split('|')] if p] for r in candidate}
    def visit(k,active):
        assert k not in active,('cycle',k)
        for p in adjacency[k]:assert p in by;visit(p,active|{k})
    for key in by:visit(key,set())
    OUT.mkdir(exist_ok=True)
    stream=io.StringIO(newline='');csv.writer(stream).writerows(candidate);save(OUT/'combined-review-candidate.csv',stream.getvalue())
    save(OUT/'combined-review-audit.jsonl',''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audit))
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(dialects);save(OUT/'dialects-proposed.csv',stream.getvalue())
    for name in ['combined-review-candidate.yaml','grierson-ahirwati-1908-literal-preliminary.txt']:save(OUT/name,(OLD/name).read_text())
    continuity={'status':'pass','source_revision':str(OLD.relative_to(PACKAGE)),
      'rows':653,'ordered_keys_and_source_literals_Native_Phonemic_exact':True,'registered_base_ID_routing':True,
      'changed_rows':sum(a!=b for a,b in zip(before,candidate)),
      'field_change_counts':dict(Counter(str(i) for a,b in zip(before,candidate) for i in range(15) if a[i]!=b[i])),
      'graph':{'variants':12,'component_sets':1,'ordered_edges':2,'closed':True,'acyclic':True},
      'receipt_hashes':RECEIPTS,'canonical_changes':False}
    save(OUT/'review-continuity.json',json.dumps(continuity,ensure_ascii=False,indent=2)+'\n')
    pins[str(Path(__file__).relative_to(PACKAGE))]=sha(Path(__file__))
    status={'status':'Reviewed source-local whole assembly; final combined acceptance pending','inputs':pins,
      'outputs':{p.name:sha(p) for p in OUT.iterdir() if p.name!='status.json'},
      'counts':{**oldstatus['counts'],'audit_records':len(audit),'lect_proposals':7,'reviewed_documentary_blocks':5,'Native_broad_spans':8},
      'language_counts':dict(Counter(r[0] for r in candidate)),
      'pending':['Final combined original-output audit','Final profile/reference/lect metadata approval and source parser readiness','Guarded installation approval/canonical checks','Full compiled/full-suite/browser gates deferred'],
      'accepted_for_installation':False,'canonical_changes':False}
    save(OUT/'status.json',json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':status['status'],'counts':status['counts'],'outputs':status['outputs']}))

if __name__=='__main__':main()
