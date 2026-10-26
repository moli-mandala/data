import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
rs=[{k:r[k] for k in ('ID','Language_ID','Form','Gloss','Source','Tags')} for r in csv.DictReader(open('cldf/forms.csv')) if 'rezaibaghbidi2003zargari[' in r['Source']]
assigned={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
(['beś','beśd','beśdo','beśipey'],['to sit','sat'],'2245','upaviśati','Turner explicitly gives European Romani beš and bešto under upaviśati; the entry includes upaviṣṭa and discusses contamination with vasati and past-stem remodelling. The existing unsplit2245 covers this family. Source3.6.2 pairs beś-/beśd-,3.14.1 gives beśdo, and3.17.3.1 defines -ipey infinitives.'),
(['xā','xayeni','xayenisa'],['to eat','eating'],'3865','khādati','Turner section1 gives Romani xa- eat and MIA khāyati/khāaï. Source3.17.2.1 explicitly forms present participles with -eni/-enisa; these remain with the present family, preserving the source glide. The separate perfect stem is excluded from this row.'),
(['xāl'],['to eat'],'3865-2','khādita','Turner section2 explicitly gives European Romani xalo ate under khādita. The source identifies xāl- as its perfect stem in3.6.2. Assign this stem to the specific inherited past family, not automatically to khādati on the basis of the productive -l pattern alone; no claim of a separate synchronic lexeme is made.'),
(['piyipey'],['to drink'],'8209','pibati','Turner explicitly gives European Romani pi drink beside MIA pi(v)aï. Source3.15.1 identifies piy- and piyipey before describing its distinct causative. The ordinary infinitive is retained without absorbing the make-drink forms.'),
(['nāś','nāśipey'],['to run'],'7027','naśyati','Turner section1 gives European Romani naš run away/go away; source3.15.1 gives nāś- and nāśipey run, then contrasts the causative. The running sense is an attested Romani continuation of this family, not a dhāvati assignment based solely on English gloss.'),
(['d','d-e','deyipey'],['to give','give!'],'6141','dadāti','Turner explicitly gives Romani del under the dēti series within dadāti, explaining interaction with nēti and imperative dehi. Source3.6.2 identifies d-;3.13.2 gives d-e;3.17.3.2 explicitly identifies deyipey as its irregular infinitive. The very short stem is supported by the grammar, not matched by one-letter similarity.'),
(['l','l-e','leyipey'],['to take; to get; to buy','buy!; get!','to buy; to get'],'10948','labhate','Turner gives European Romani lel takes and Armenian li take/buy under labhate, explicitly noting remodelling with MIA dēti/nēti. Source3.6.2 identifies l- take/get/buy;3.13.2 and3.17.3.2 connect l-e and the irregular infinitive leyipey. The source buy sense is retained within that verb family.')]
a=[]
for fs,gs,pa,label,ev in groups:
 for r in rs:
  if r['ID'] not in assigned and r['Form'] in fs and r['Gloss'] in gs:a.append(dict(record=r,parent=pa,kind='reflex',citation=f'CDIAL[{pa} and addenda];'+r['Source'],evidence=ev+' Exact grammar citation, morphology and source spelling preserved.'))
assert len(a)==17,len(a)
(P/'pass029-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=2)+'\n')
locs={s for x in a for s in x['record']['Source'].split(';')};aud=[r for r in csv.DictReader(open('data/other/forms/raw_data/20260825-rezai-baghbidi-zargari-audit.csv')) if r['Source'] in locs];(P/'pass029-source-audit.json').write_text(json.dumps(aud,ensure_ascii=False,indent=2)+'\n')
lines=['# Zargari: further basic verbs','','17 records / 17 reflex rows, six verb families and seven specific parents. Compiled rank1 and accepted sidecar links excluded.','','| # | Zargari forms and meanings | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(fs,gs,pa,label,ev) in enumerate(groups,1):
 n=sum(x['parent']==pa for x in a);lines.append(f'| {i} | **{" / ".join(fs)}** ‘{" / ".join(gs)}’ ({n}) | **{label}**, CDIAL **{pa}** | {ev} |')
lines+=['','Primary grammar: zargari-primary-pages.json, printed pp138,141–142. Source definitions are the basis for grouping stems, imperatives and infinitives. Historical subentries still take precedence where Turner explicitly assigns the past family: xāl goes to3865-2. The existing2245 node includes both upaviśati/upaviṣṭa discussion; no new node or split is created.','', 'Excluded: analytic beśdo-som/beśdi-som predicates, causative run/drink, mediopassive eat, and negative forms. Earlier holds remain. Search008–010 and primary007–008 contain discovery and full dictionary evidence. No source re-ingestion or rebuild.','', 'Validation and save status: pending.'];(P/'pass029-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted')
