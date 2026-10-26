# Research preparation before graph validation. Final decisions additionally hold yellow8233 due to a pre-existing ancestry cycle; see pass035-existing-cycle.json. Do not rerun over saved decisions.
import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
rs=[r for r in csv.DictReader(open('cldf/forms.csv')) if 'lehr' in r['Source'].split(';')]
linked={r['Form_ID'] for r in read_assignments()};linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
(['mār-'],['kill'],'10066','mārayati','Turner explicitly gives Pashai mār- under mārayati kills, alongside regional kill/strike continuations. This is the inherited lexical verb, not a new synchronic causative edge to a reconstructed die stem.'),
(['ka-','karik'],['do','to do'],'2814','karoti','Turner explicitly gives both Pashai kar- and ka- under karoti. Lehr separately records the lexical stem ka- and citation form karik to do; both belong to this verb. Do not include kave- cause to do or other suppletive tense forms by blanket matching.'),
(['pac-'],['ripen'],'7654','pacyate','Turner explicitly gives Pashai pač- to be boiled, ripen under pacyate, also used transitively to boil/cook. The source intransitive ripen sense selects this entry rather than a bare active pacati comparison.'),
(['pēlā'],['yellow'],'8233','pītala','Turner section1 explicitly gives Darrai Pashai pēla yellow, beside ar. piālo and Prakrit pīala. Choose pītala8233, not the pīthala/pīvala branches.'),
(['de-'],['give'],'6141','dadāti','Turner explicitly lists Pashai weg. chil. dē- under the remodeled deti present discussed in dadāti6141. Preserve that analogical history rather than deriving e directly from the Sanskrit reduplicated present.'),
(['cōr'],['thief'],'4931','cora','Turner section1 lists Pashai lauṛ. čūr thief, beside regional čor/čōr. The source simple masculine lexical noun belongs here, not feminine corī or the separately listed stealing verb.'),
(['aṛā'],['half'],'644','ardha','Turner explicitly gives Pashai aḍḍ(h)ā and ar. aṛo half-full under ardha half. The regional retroflex-flap stem supports Lehr aṛā half; the printed Pashai-to-Shumashti loan arrow does not label the Pashai form itself a loan.'),
(['pin-'],['know'],'8157','*pijānāti','Turner explicitly gives Darrai Pashai pīn- to know, recognize, hear under *pijānāti, alongside piēn/piān elsewhere in Pashai. Select this contracted prefixed stem rather than bare jānāti5193; leave the separately inflected pineken for morphological review.')]
a=[];lines=['# Pashai verbs and descriptive vocabulary','','Nine records proposed under standing autonomous authorization.','','| # | Pashai | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(forms,glosses,parent,head,ev) in enumerate(groups,1):
 hits=[r for r in rs if r['Form'] in forms and r['Gloss'] in glosses and r['ID'] not in linked];assert len(hits)==len(forms),(forms,len(hits))
 for r in hits:a.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=parent,kind='reflex',citation=f'CDIAL[{parent}];lehr',evidence=ev+' Existing source citations provide no page locator; none is invented.'))
 lines.append(f'| {i} | **{" / ".join(forms)}** ‘{" / ".join(glosses)}’ | **{head.replace("*",chr(92)+"*")}**, CDIAL **{parent}** | {ev} |')
held=[dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},reason='CDIAL1111 explicitly labels Pashai āṇḍa egg and āṇḍī scrotum as borrowed from Indic, alongside a distinct inherited *āṇḍra history. Source aṇā is insufficient to decide the route or identify an existing immediate donor; no direct reflex assignment.') for r in rs if r['Form']=='aṇā' and r['Gloss']=='egg']
assert len(a)==9
(P/'pass035-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n')
lines+=['','Evidence: homepage search005–006; full entries and addenda pashai-primary003.json. Egg remains held with a stronger loan-route reason. Earlier ear, seed, sugar, eye and skin candidates remain unassigned; no borrowing route inferred from word resemblance.','Validation: pending.']
(P/'pass035-review.md').write_text('\n'.join(lines)+'\n')
wanted={x['record']['Gloss'] for x in a};aud=[r for r in csv.reader(open('data/other/forms/20230705-pashai.csv')) if len(r)>3 and r[3] in wanted];(P/'pass035-source-audit.json').write_text(json.dumps(aud,ensure_ascii=False,indent=2)+'\n')
print(len(a),'accepted',len(held),'held')
