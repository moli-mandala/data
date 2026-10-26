import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
rs=[r for r in csv.DictReader(open('cldf/forms.csv')) if 'lehr' in r['Source'].split(';')]
linked={r['Form_ID'] for r in read_assignments()};linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
(['aŋār'],'fire','125','aṅgāra','Turner explicitly lists Pashai aṅgār fire under aṅgāra section1, beside reduced Torwali aṅā. The source aŋār retains the nasal and long vowel, with loss of the stop. The explicit borrowing statement in this entry concerns Shumashti from Pashai, not Pashai itself.'),
(['gā̌ns','gāns'],'grass','4471','ghāsa','Turner lists Darrai Pashai gāas and Chilas gõs, with Shumashti gãs. These support the Pashai grass family including nasalization; retain the source n and accent spellings without normalization.'),
(['ā̌st'],'hand','14024','hasta','Turner explicitly lists Pashai hāst/hōst and h-less āstu-m, ōst, ōs. The retained st cluster identifies this family. Exclude the separate merged copula/hand record and the unexplained das record.'),
(['āṛ','ā̌ṛ'],'heart','14064','hārdi','Turner explicitly lists Darrai Pashai āṛ heart under hārdi, beside lauṛ. hāṛ and other h-less Pashai forms. This is an exact regional comparison; preserve the two installed accent spellings.'),
(['nās'],'nose','7031','nasta','Turner section1 explicitly gives Pashai nāst and nās nose, with final t loss. Choose nasta7031, not the nakka6909 family or another numbered nasal-stem branch.'),
(['nandi'],'river','6943','nadī','Turner explicitly lists Pashai nandī under the early nasalized *nandī continuation of nadī, together with Ningalam nandī. Nasalization is part of the printed analysis, not a new sound rule inferred from spelling.'),
(['sev-'],'sew','13444','sīvyati','Turner section1 lists Pashai ar. sīw- and weg. sew- in this sewing family. Retain his explicit alternative derivation from section2 sīvayati: the Pashai stem alone does not settle that older formation. It is not automatically assigned to the distinct causative sēvayati added for Western Pahari.'),
(['sor'],'sun','13574-4','sūrī','Turner explicitly assigns Pashai sur in all dialects to section4 sūrī, feminine, rather than section3 sūriya used for Ningalam and Shumashti. The source o vowel does not justify overriding this Pashai-specific branch.'),
(['vāṛ'],'stone','11348-2','*vārta','Turner section2 explicitly gives Pashai lauṛ. wāṛ and Shumashti wāṛ stone under *vārta. Choose the long-vowel subsection, not the bare varta11348 used for other regional stone forms.'),
(['se','sai'],'sister','13913','svasṛ','Turner explicitly gives Pashai sāī and säi sister, explaining the local formations from svasā mē and citing sayā-m and related possessed forms. The source se/sai belong to this family; the historical possessive development is retained as Turner’s account.')]
a=[];lines=['# Pashai: additional basic vocabulary','','13 proposed records / 13 reflex rows across ten etyma; standing autonomous authorization.','','| # | Pashai | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(forms,gloss,parent,head,ev) in enumerate(groups,1):
 hits=[r for r in rs if r['Form'] in forms and r['Gloss']==gloss and r['ID'] not in linked];assert len(hits)==len(forms),(forms,len(hits))
 for r in hits:a.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=parent,kind='reflex',citation=f'CDIAL[{parent}];lehr',evidence=ev+' Existing Lehr source citations have no page locator; none is invented.'))
 lines.append(f'| {i} | **{" / ".join(forms)}** ‘{gloss}’ | **{head.replace("*",chr(92)+"*")}**, CDIAL **{parent}** | {ev} |')
assert len(a)==13
held=[]
for r in rs:
 if r['ID'] in linked:continue
 if r['Form'] in ['bēc','kāṛ','aṇā','ainc','goṛ']:
  held.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},reason='Shortlisted but excluded from this batch: current comparison does not yet resolve exact stem, branch or possible loan route. Seed bēc does not match the bare Pashai bī comparison under9250; review bījya and competing seed families before assigning.'))
(P/'pass034-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n')
lines+=['','Discovery: homepage search002–003. Full shortlisted dictionary articles/addenda: pashai-primary002.json. Existing installed source records and dialect tags retained; no source edits, new nodes or rebuild.',f'{len(held)} shortlisted records deferred; no mixed-gloss, compound or ambiguous copula record included.','Validation: pending.']
(P/'pass034-review.md').write_text('\n'.join(lines)+'\n')
print(len(a),'accepted',len(held),'held')
