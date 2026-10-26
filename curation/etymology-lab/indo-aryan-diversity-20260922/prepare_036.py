import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
rs=[r for r in csv.DictReader(open('cldf/forms.csv')) if 'herin2012domari[' in r['Source']];linked={r['Form_ID'] for r in read_assignments()};linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
(['ōśt'],'2563','oṣṭha','lip','Turner explicitly gives Palestinian Domari ōšt under oṣṭha. The Aleppo source preserves the same št cluster; no general lip comparison is substituted.'),
(['cərī́'],'3727','kṣura','knife','Turner lists Palestinian Domari čiri knife/razor in the ch-/č- knife series under kṣura, including feminine churī/churikā. Herin §1.1 explicitly describes short-vowel centralization and čərī~črī; the central vowel does not require another etymon.'),
(['kūkár'],'3208','kukkuṭa','cock','Turner gives Asian Gypsy kukar hen/bird in the kukkuṭa family and widespread regional cock forms. This is an expressive/onomatopoeic family in Turner’s analysis; do not claim that the word excludes sound-symbolic reshaping.'),
(['kərī́','kərī'],'3232','kuṭī','house','Turner explicitly gives Palestinian Domari kuri house/tent/room under kuṭī, alongside Asian kuri/guri tent. Herin §1.1 discusses centralization and elision in Aleppo kərī~krī, supporting the comparison rather than ghara4428.'),
(['bḗn','bēn-ṓr'],'9349','bhaginī','sister / your sister','Turner explicitly gives Palestinian Domari bēn under bhaginī. Herin §1.3 explicitly derives bēn-ōr your sister by adding the bound pronoun -ōr to bēn; this is ordinary possessive inflection of the same inherited noun.')]
a=[];lines=['# Aleppo Domari: nouns and know paradigm','','Standing autonomous authorization; source-defined inflections preserved.','','| # | Domari | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(forms,parent,head,gloss,ev) in enumerate(groups,1):
 hits=[r for r in rs if r['Form'] in forms and r['ID'] not in linked];assert len(hits)==len(forms),(forms,len(hits))
 for r in hits:a.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=parent,kind='reflex',citation=r['Source']+f';CDIAL[{parent}]',evidence=ev))
 lines.append(f'| {i} | **{" / ".join(forms)}** ‘{gloss}’ | **{head}**, CDIAL **{parent}** | {ev} |')
base='f_cbssr6z2uijwe';active=[];passive=[]
for r in rs:
 if r['ID'] in linked:continue
 if r['Form'].startswith('ǧān') and r['Gloss'] in ['to know','I know','know']:
  active.append(r)
  ev='Turner5193 explicitly gives Palestinian Domari ǰan- knows under jānāti. Herin §3.1 Tables7–9 presents the perfective ǧān-d-, imperfective ǧān- and subjunctive ǧān- paradigms and their agreement endings; these are ordinary inflections of one lexical verb. The remoteness -ā and contextualizing -e are explicitly analysed in §3.4 and §5. No passive or mixed know/go record is included.'
  a.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent='5193',kind='reflex',citation=r['Source']+';CDIAL[5193];herin2012domari[section 3.1, Tables 7–9]',evidence=ev))
 if r['ID'] in ['f_jjgk6hhvetzwo','f_kti6jpljiurns']:
  passive.append(r);a.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=base,kind='derived',citation=r['Source']+';herin2012domari[p. 35, section 3.3]',evidence='Herin §3.3 explicitly contrasts ǧān know with ǧānī be known and identifies passive -ī plus perfective extension -r in ǧānīre. Link both passive forms to the existing same-source lexical base ǧān- (f_cbssr6z2uijwe), whose historical ancestry is established in this pass; do not flatten the passive derivative into a direct Sanskrit reflex.'))
lines += [f'| 6 | **ǧān-**, its inflected paradigm and **ǧān-əm-ā** ‘know’ ({len(active)} records) | **jānāti**, CDIAL **5193** | Turner explicitly lists Palestinian Domari ǰan-. Herin Tables7–9 identifies perfective -d-, person endings and subjunctive/imperfective forms; §3.4 illustrates the remoteness form, and §5 the contextualizing form. |',f'| 7 | **ǧānī- / ǧānīre** ‘be known’ | Derived from Domari **ǧān-** (f_cbssr6z2uijwe) | Herin §3.3 explicitly identifies passive -ī and perfective extension -r; both records point to the existing active lexical base. |']
held=[dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},reason='Merged know/go glosses prevent a single unambiguous assignment; preserve source identity and leave unlinked rather than silently repairing the record.') for r in rs if r['ID']=='f_vsmv7fcwpgbey']
assert len(active)==22 and len(passive)==2,(len(active),len(passive));assert len(a)==31,len(a)
(P/'pass036-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n')
lines+=['',f'{len(a)} records / {len(a)} rows:29 reflex,2 derived. One mixed-gloss record held. Other possessive/case noun tokens and negative verb forms are outside this batch. Gold is deferred: Turner explicitly allows suvarṇa/sauvarṇa alternatives, and this source needs a closer Domari-specific check.','Evidence: homepage searches002–003; full dictionary articles/addenda domari-primary002–003.json; primary grammar domari-primary-prose.txt (existing source HTML cache). No new nodes, source reparsing, or rebuild.','Validation: pending.']
(P/'pass036-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(held),'held')
