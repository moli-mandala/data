"""Independent bounded read-only audit of the combined source candidate."""
import collections, csv, hashlib, json, re, sys, unicodedata
from pathlib import Path
P = Path(__file__).resolve().parent.parent
I = P/'integration'
O = Path(__file__).resolve().parent
DATA = next(p for p in P.parents if (p/'tags.py').exists())
sys.path.insert(0, str(DATA))
from tags import GRAMMATICAL_TAGS, GENDER_TAGS
from segments import Tokenizer
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def jl(p): return [json.loads(x) for x in p.read_text().splitlines()]
def js(p): return json.loads(p.read_text())
pins = {p.name: sha(p) for p in I.iterdir() if p.is_file()}
status=js(I/'status.json')
issues=[]
def check(ok, field, key='', detail=None):
    if not ok: issues.append(dict(field=field,key=key,detail=detail))
for name,h in status['outputs'].items(): check(sha(I/name)==h,'output hash',name)
for name,h in status['input_hashes'].items(): check(sha(P/name)==h,'input hash',name)
rows=list(csv.reader((I/'combined-candidate.csv').open()))
check(all(len(r)==15 for r in rows),'15 columns')
bykey={r[10]:r for r in rows}
check(len(bykey)==len(rows),'unique entry keys')
audit=jl(I/'combined-audit.jsonl')
check(len({a['source_unit_key'] for a in audit})==len(audit),'unique physical keys')
atom={r['emission_key']:r for r in jl(P/'specimens/atomic-emissions-reviewed.jsonl')}
routes={r['key']:r for r in js(P/'source-lect-mapping-approved-20260927.json')['mappings']}
external={'external-Gondi':'Gondi','external-Kanarese':'Kannada','external-Tamil':'Tamil','external-Telugu':'Telugu'}
covered=set(); counts=collections.Counter()
for a in audit:
    k=a['source_unit_key']; s=a['source_record']; sec=a['section']; d=a['disposition']; keys=a['entry_keys']
    counts[(sec,d)]+=1
    if d not in ('emitted','joined-physical-fragment','reused-existing-occurrence'):
        check(not keys,'nonemission has keys',k);continue
    if d=='reused-existing-occurrence':
        check(len(keys)==1 and a.get('reuse_evidence'),'reuse evidence',k);continue
    check(all(e in bykey for e in keys),'entry resolution',k)
    if not all(e in bykey for e in keys): continue
    covered.update(keys)
    expected=[]
    if sec=='table':
        rr=s['reviewed_reading']; forms=rr['forms']; qualifiers=rr.get('form_qualifiers',[])
        tags=rr['tags'][:]
        if a.get('cross_reference_source_key'): tags=['fut' if t=='pres' else t for t in tags]
        for n,f in enumerate(forms):
            gloss=(qualifiers[n]+' ' if len(qualifiers)==len(forms) else '')+s['english_prompt']
            expected.append((f,gloss,'Kui',tags,'grierson1906_kui_khondmals'))
    elif sec=='grammar':
        forms=a['alternative_decision']['forms'] if a.get('alternative_decision') else [s['form']]
        route=routes.get(s['route_key'],{})
        expected=[(f,s['gloss'],route.get('proposed_base') or external.get(s['route_key']),s['tags'],route.get('proposed_dialect_id')) for f in forms]
    elif sec=='grammar-supplement':expected=[(s['form'],s['gloss'],s['language_id'],s['tags'],None)]
    elif sec=='specimen-physical-atom':
        r=atom[keys[0]]
        expected=[(r['emission_form'],r['emission_gloss'],'Kui',r['tags'],r['source_dialect'])]
        check(k in r['physical_source_keys'],'joined physical provenance',k)
    elif sec=='specimen-expression':
        dialect={40:'gumsur_bod',41:'khondmals',42:'khondmals',43:'kalahandi'}[s['specimen']]
        expected=[(s['emission_form'],s['emission_gloss'],'Kui',s['tags'],'grierson1906_kui_'+dialect)]
        check(all(x in bykey for x in a['aligned_atom_entry_keys']),'expression atom links',k)
    elif sec in ('outside-target','outside-comparand'):
        expected=[(s['form'],s['gloss'],'Kui' if sec=='outside-target' else None,[],None)]
    check(len(expected)==len(keys),'expansion count',k)
    for ek,(form,gloss,lang,tags,dialect) in zip(keys,expected):
        row=bykey[ek]
        check(row[2]==form,'literal',ek,[form,row[2]])
        check(row[3]==gloss,'gloss',ek,[gloss,row[3]])
        if lang:check(row[0]==lang,'language',ek,[lang,row[0]])
        got=set(row[14].split())
        clear_alternates={
            'grierson1906kui:p652:khondmals-table:prompt47:v1',
            'grierson1906kui:p652:khondmals-table:prompt51:v1',
            'grierson1906kui:p660:khondmals-table:prompt99:v2',
            'grierson1906kui:p672:khondmals-table:prompt175:v1',
            'grierson1906kui:p672:khondmals-table:prompt175:v2'}
        required=set(tags)-({'uncertain'} if ek in clear_alternates else set())
        check(required<=got,'dropped source tags',ek,sorted(required-got))
        if ek in clear_alternates:check('uncertain' not in got and 'Source uncertainty:' not in row[6],'clear alternate wrongly uncertain',ek)
        if dialect:check(any(t.startswith('dialect:Kui:'+dialect+':') for t in got),'lect',ek,dialect)
        if 'printed_page' in s:check(str(s['printed_page']) in row[7],'source page',ek,row[7])
check(covered==set(bykey),'orphan CSV keys',detail=sorted(set(bykey)-covered))
languages={r['ID'] for r in csv.DictReader((DATA/'cldf/languages.csv').open())}
unknown=set(); internal_separators=[]; procedural=[]
for r in rows:
    k=r[10];check(r[0] in languages,'unknown language',k,r[0])
    check(all(not r[n] for n in (1,4,5,8,9,11,12)),'unexpected nonempty field',k)
    unknown.update(t for t in r[14].split() if not t.startswith('dialect:') and t not in GRAMMATICAL_TAGS|GENDER_TAGS)
    check(not any(';' in x for x in re.findall(r'\[([^]]*)\]',r[7])),'internal citation separator',k)
    check(r[7].count('[')==r[7].count(']'),'citation brackets',k)
    if re.search(r'Do not |do not |endpoint|not an invented|currently|exporter|serialize',r[6]):procedural.append(k)
check(not unknown,'unregistered tags',detail=sorted(unknown))
approval=js(P/'root-component-relations-review-20260927.json')
expected_graph={a['child_key']:a['ordered_parent_keys'] for a in approval['accepted']}
actual_graph={r[10]:r[13].split('|') for r in rows if r[13]}
check(actual_graph==expected_graph,'exact approved graph',detail=actual_graph)
for k,parents in actual_graph.items():check(all(p in bykey for p in parents),'graph endpoint',k)
tok=Tokenizer(str(I/'literal-preservation.txt')); profile_fail=[]
for r in rows:
    for norm in ('NFC','NFD'):
        f=unicodedata.normalize(norm,r[2]);display=tok(f,column='IPA').replace(' ','').replace('#',' ')
        expected=unicodedata.normalize('NFC',r[2]).lower().replace('ṅ','ŋ')
        if display!=expected:profile_fail.append([r[10],norm,display,expected])
check(not profile_fail,'complete profile coverage',detail=profile_fail[:10])
route_review=js(P/'outside-comparands-routing-reuse-review-20260927.json')
route_keys={r['source_unit_key']:r for r in route_review['decisions']}
new_labels={'literary Telugu':'Telugu','Malayalam':'Malayalam','Tamil dialects (unspecified)':'Tamil','Kaikadi of Berar':'Kaikadi','Tulu':'Tulu'}
for a in audit:
    if a['section']!='outside-comparand':continue
    s=a['source_record'];k=a['source_unit_key'];route=route_keys.get(k)
    if a['disposition']=='emitted':
        lang=(route or {}).get('language_id') or new_labels.get(s['source_language_label'])
        if not lang:
            candidates={r.get('language_id') for r in route_review['decisions'] if r['input_label']==s['source_language_label'] and r.get('language_id')}
            if len(candidates)==1:lang=next(iter(candidates))
        check(lang==bykey[a['entry_keys'][0]][0],'independent comparator routing',k,lang)
    if 'unresolved-reference' in s.get('status',''):check(a['disposition']=='reference-hold','ambiguous reference emitted',k)
    if s['source_language_label'] in ['Bhili of Basim','Burgandi','Korava']:
        check(a['disposition']=='identity-hold','held identity emitted',k)
check(pins=={p.name:sha(p) for p in I.iterdir() if p.is_file()},'stable candidate during review')
result={'reviewer':'/root/asur_finish','status':'field audit passed' if not issues else 'corrections required',
 'candidate_hashes':pins,'rows':len(rows),'audit_units':len(audit),'section_dispositions':{' / '.join(k):v for k,v in counts.items()},
 'issues':issues,'procedural_notes_screen_candidates':procedural,
 'multianswer_uncertainty_review':[{'key':a['source_unit_key'],'forms':a['source_record']['reviewed_reading']['forms'],'uncertainty':a['source_record']['reviewed_reading'].get('uncertainty')} for a in audit if a['section']=='table' and a['disposition']=='emitted' and len(a['entry_keys'])>1 and a['source_record']['reviewed_reading'].get('current_uncertainty_tag')],
 'checks':'Every CSV row is compared with its reviewed input literal/gloss, known section language, source tags and required lect; all dispositions, physical joins, expression links, graph endpoints and profile NFC/NFD coverage checked.',
 'limits':['Not a fresh visual re-reading of all originals. Outside-comparand exact routing and scholarly Notes require the companion reviewer findings.', 'No actual YAML, source reference formatter or generic parser fixture is certified by this script.', 'No canonical edits, full build, database or browser validation.']}
(O/'field-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(result['status'],len(rows),'rows;',len(issues),'issues;',len(procedural),'Notes screen candidates')
