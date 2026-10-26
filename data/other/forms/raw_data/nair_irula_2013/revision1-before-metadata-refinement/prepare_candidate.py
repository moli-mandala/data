"""Complete Irula proposal from keyed source evidence; no canonical mutation."""
import csv
import hashlib
import json
import unicodedata
from pathlib import Path
HERE=Path(__file__).resolve().parent
units=[json.loads(s) for s in (HERE/'physical-units-first-reading.jsonl').read_text().splitlines()]
rows=[];audit=[]
MARKER_SECTIONS={'plural-markers','plural-common-repeat','plural-human-marker-repeat','plural-pronominal-marker','plural-prevocalic','plural-prepause','archaic-plural','masculine-markers','feminine-markers','accusative-markers','instrumental-markers','sociative-marker','dative-markers','purposive-marker','ablative-markers','locative-markers','genitive-markers','adjectival-marker','past-in-marker','past-nd-marker','past-t-marker','past-d-marker','negative-markers','negative-construction','infinitive-markers','prohibitive-marker','personal-ending-table','relative-participle-markers','verbal-participle-marker','negative-verbal-participle-marker','permissive-markers','hortative-marker','hortative-comparison-context','participial-noun-marker','neuter-participial-noun-marker','optative-markers','polite-imperative-marker','polite-imperative-base-marker','imperative-plural-marker','injunctive-markers','conditional-markers','concessive-marker','temporal-marker','zvelebil-temporal-comparison','derived-adverb-marker'}

def tags_for(u):
    s=u['section'];label=u['grammatical_label'];tags=[]
    def add(*values):
        for t in values:
            if t and t not in tags:tags.append(t)
    if s in MARKER_SECTIONS:add('suffix')
    if s in {'interrogative-morphs','plural-human-bases','derived-adverb-bases'}:
        add('affix' if s=='interrogative-morphs' else 'stem')
    if 'plural' in s and s not in {'plural-human-bases','plural-examples'}:add('pl')
    if s=='archaic-plural':add('archaic')
    if s=='masculine-markers':add('m')
    if s=='feminine-markers':add('f')
    for part,tag in [('accusative','acc'),('instrumental','instr'),('sociative','comitative'),('dative','dat'),('purposive','purposive'),('ablative','abl'),('locative','loc'),('genitive','gen'),('vocative','voc')]:
        if part in s:add(tag)
    # Source case examples identify nominal/pronominal lexical bases explicitly.
    nominal={'accusative-examples':'noun','instrumental-examples':'noun','sociative-example':'noun','dative-examples':'noun','ablative-examples':'noun','locative-examples':'noun','genitive-examples':'noun','vocative-example':'noun','purposive-example':'verb'}
    if s in nominal:add(nominal[s])
    if s=='instrumental-examples' and u['section_index']==1:tags.remove('noun');add('pron','1pl')
    if s=='dative-examples' and u['section_index']==1:tags.remove('noun');add('pron','3pl')
    if s=='ablative-examples' and u['section_index']==2:tags.remove('noun');add('proper-noun')
    if s in {'locative-examples','genitive-examples'} and ((s=='locative-examples' and u['section_index']==4) or (s=='genitive-examples' and u['section_index']==1)):
        tags.remove('noun');add('pron','1pl')
    if s=='plural-examples':
        i=u['section_index']
        if i in (1,2,3):add('pron', '1pl' if i in (1,3) else '2pl')
        elif i in (4,5):add('pron','demonstrative','3pl','dist' if i==4 else 'prox')
        elif i==6:add('pron','interr')
        else:add('noun','pl')
    if s=='personal-pronoun-table':add('pron',*[x for x in label.split() if x in {'1sg','1pl','2sg','2pl','3sg','3pl'}]);add('refl' if 'reflexive' in label else '')
    if s in {'remote-demonstratives','proximate-demonstratives','masculine-demonstratives'}:
        add('pron','demonstrative')
        add('dist' if s=='remote-demonstratives' or (s=='masculine-demonstratives' and u['section_index']==1) else 'prox')
        if s!='masculine-demonstratives':
            i=u['section_index'];add('3sg' if i<=3 else '3pl');add({1:'m',2:'f',3:'n',5:'n'}.get(i,''))
        else:add('m')
    if s in {'interrogative-morphs','interrogative-examples'}:add('interr');add('pron' if s=='interrogative-examples' else '')
    if s=='indefinite-example':add('pron','indef')
    if s=='universal-examples':add('pron','quantifier')
    if s=='noun-modifiers':add('noun')
    if s in {'adjective-categories','adjectival-examples','appellative-adjectives','noun-participle-adjectives'}:add('adj')
    if s=='adjective-categories':
        if 'demonstrative' in label:add('demonstrative')
        if 'interrogative' in label:add('interr')
    if s=='adjectival-marker':add('adjectivizer')
    if s.startswith('past-'):add('pret')
    if s.startswith('past-') and 'example' in s:add('verb')
    if s=='past-in-examples':add('2pl')
    if s=='past-nd-examples':add('1pl' if u['section_index']==1 else 'relative','' if u['section_index']==1 else 'participle')
    if s=='past-t-examples':
        if u['section_index']==1:add('participle')
        elif u['section_index']==2:add('3pl')
        else:add('1pl')
    if s=='past-d-example':add('1pl')
    if s=='nonpast-examples':
        add('verb','non-past');add({1:'1pl',2:'1pl',3:'3pl',4:'3sg'}[u['section_index']]);add('f' if u['section_index']==4 else '')
    if s in {'negative-markers','negative-examples','negative-verbs','negative-whole-expressions','negative-comparison-examples','negative-construction','negative-verbal-participle-marker','negative-verbal-participle-examples'}:add('neg')
    if s in {'negative-examples','negative-verbs','negative-whole-expressions','negative-comparison-examples'}:add('verb')
    if s=='periphrastic-negative-examples':
        add('verb')
        if u['section_index'] in (1,3):add('1sg')
        else:add('neg')
        if u['section_index']==4:add('pret')
    if s=='infinitive-markers' or s=='infinitive-examples':add('inf');add('verb' if s=='infinitive-examples' else '')
    if s in {'prohibitive-marker','prohibitive-example'}:add('prohibitive','neg');add('verb' if s=='prohibitive-example' else '')
    if s=='personal-ending-table':
        for part,tag in [('first-person','first-person'),('second-person','second-person'),('third-person','third-person'),('inclusive','inclusive'),('exclusive','exclusive'),('masculine','m'),('feminine','f'),('neuter','n')]:
            if part in label:add(tag)
    if s.startswith('relative-participle'):add('relative','participle');add('verb' if s=='relative-participle-examples' else '')
    if s=='relative-participle-markers' and u['section_index']==2:add('neg')
    if s=='relative-participle-examples':
        if u['section_index']<=2:add('pret')
        if u['section_index']==3:add('non-past')
        if u['section_index']==4:add('neg')
    if s in {'verbal-participle-marker','verbal-participle-examples','negative-verbal-participle-marker','negative-verbal-participle-examples'}:
        add('participle');add('verb' if s.endswith('examples') else '')
    if s.startswith('permissive-'):add('permissive');add('verb' if s.endswith('examples') else '')
    if s in {'hortative-marker','hortative-examples','hortative-comparison-context'}:add('hortative');add('verb','1pl') if s=='hortative-examples' else None
    if s=='relative-noun-expression':add('noun','relative','participle')
    if s in {'participial-noun-marker','neuter-participial-noun-marker'}:add('nominalizer','participle');add('n','sg') if s.startswith('neuter') else None
    if s in {'participial-noun-examples','neuter-participial-noun-examples'}:
        add('noun','participle','relative')
        if s.startswith('neuter'):add('n','sg')
        else:add('3sg' if u['section_index']<4 else '3pl');add('f' if u['section_index']==3 else ('m' if u['section_index']<3 else ''))
    if s.startswith('optative-'):add('opt');add('verb' if s.endswith('examples') else '')
    if s.startswith('polite-imperative'):add('impv');add('verb','polite') if s.endswith('examples') else None
    if s in {'imperative-plural-marker','imperative-examples'}:add('impv');add('pl') if s=='imperative-plural-marker' else add('verb','2pl' if u['section_index']==2 else '2sg')
    if s.startswith('injunctive-'):
        add('injunctive');add('sg' if label.endswith('singular') else ('pl' if label.endswith('plural') else ''))
        if s in {'injunctive-examples','injunctive-continuation'}:add('verb');add('2pl' if u['gloss'].find('(Pl.)')!=-1 else 'second-person')
    if s.startswith('conditional-'):add('conditional');add('verb' if s.endswith('examples') else '')
    if s.startswith('concessive-'):add('concessive');add('verb' if s.endswith('example') else '')
    if s in {'temporal-marker','temporal-example','zvelebil-temporal-comparison'}:add('temporal');add('verb' if s=='temporal-example' else '')
    if s=='additional-injunctive-temporal-examples':
        add('verb','injunctive' if u['section_index']<=3 else 'temporal')
        if u['section_index'] in (1,3):add('2pl')
        if u['section_index']==2:add('second-person')
    if s in {'simple-adverbs','derived-adverb-examples'}:add('adv')
    if s=='simple-adverbs' and u['section_index']==1:add('temporal')
    if s=='derived-adverb-marker':add('adverbializer')
    if s=='derived-adverb-bases':add('noun','stem')
    if ' ' in (u['surface'] or u['form']).strip():add('multiword-expression')
    if s in {'negative-whole-expressions','concessive-example'}:add('sentential')
    if u['section']=='noun-participle-adjectives' and u['section_index']==2:add('uncertain')
    if u['section']=='personal-ending-table' and u['section_index']==10:add('uncertain')
    return ' '.join(tags)

def gloss_for(u):
    g=u['gloss'];s=u['section']
    if s in MARKER_SECTIONS:return ''
    if s=='past-in-examples':return g.split('-you')[0]
    if s=='past-nd-examples' and u['section_index']==1:return 'hanged'
    if s=='past-t-examples' and u['section_index']>1:return g.split('-')[0]
    if s=='past-d-example':return 'worked'
    if s=='nonpast-examples':return g.split('-')[0]
    if s=='imperative-examples' and u['section_index']==2:return 'say'
    return g

for u in units:
    emitted=[];state='audit-only';reason=''
    if u['role']=='target':
        forms=(u['surface'] or u['form']).split('/')
        for i,f in enumerate(forms):
            f=unicodedata.normalize('NFC',f)
            key=u['unit_key']+(f':v{i+1}' if len(forms)>1 else '')
            notes=[u['source_claim'],u['note']]
            if u['surface']:notes.append('Printed segmented analysis: '+u['form'])
            if len(forms)>1:notes.append('Printed grouped alternatives: '+u['form'])
            row=['Irula','',f,gloss_for(u),'','','; '.join(n for n in notes if n),'nair2013irula[p. '+str(u['printed_page'])+']','','',key,'','','',tags_for(u)]
            rows.append(row);emitted.append({'entry_key':key,'form':f,'gloss':row[3],'notes':row[6],'tags':row[14]})
        state='proposed-target'
    else:reason={'documentary':'Source contextual claim without a lexical target attribution','zero-exponent':'Unpronounced zero evidence','comparison-control':'Explicit Old Tamil comparison deliberately excluded from target chapter rows','bibliographic-context':'Reported work, no directly collated target forms','non-attested-control':'Source explicitly rejects this construction as target usage'}[u['role']]
    uncertainties=[]
    if u['section']=='noun-participle-adjectives' and u['section_index']==2:uncertainties.append({'type':'source-character-encoding','detail':'Source text layer uses Cyrillic ї for visible i with diaeresis; preserved literally without IPA.'})
    if u['section']=='personal-ending-table' and u['section_index']==10:uncertainties.append({'type':'source-notation','detail':'Visible final backslash after -ina has no explained function.'})
    audit.append({'typed_uncertainties':uncertainties,'unit_key':u['unit_key'],'disposition':state,'source_record':u,'proposed_emissions':emitted,'proposed_entry_keys':[e['entry_key'] for e in emitted],'exclusion_reason':reason})
# Only the explicitly freely varying conditional suffix pair has a proposed graph.
by={r[10]:r for r in rows}
child='nair2013irula:p27:conditional-markers:02';parent='nair2013irula:p27:conditional-markers:01'
by[child][11]=parent
graph=json.loads((HERE/'graph-decisions-proposal.json').read_text())
for relation in graph['relations']:
    assert relation['child'] in by and relation['parent'] in by
    by[relation['child']][13]=relation['parent']
for a in audit:
    for e in a['proposed_emissions']:
        e['variant_of_key']=by[e['entry_key']][11]
        e['derivation_parent_keys']=by[e['entry_key']][13]
assert len(by)==len(rows) and all(len(r)==15 for r in rows)
with (HERE/'proposed.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
(HERE/'proposal-audit.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audit))
status={'status':'whole chapter candidate; independent review and metadata gates pending','rows':len(rows),'physical_units':len(units),'audit_only':sum(not a['proposed_entry_keys'] for a in audit),'variant_edges':1,'derived_edges':len(graph['relations']),'csv_sha256':hashlib.sha256((HERE/'proposed.csv').read_bytes()).hexdigest(),'audit_sha256':hashlib.sha256((HERE/'proposal-audit.jsonl').read_bytes()).hexdigest()}
(HERE/'proposal-status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2))
