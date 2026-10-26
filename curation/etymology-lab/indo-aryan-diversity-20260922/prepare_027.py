import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
rs=[{k:r[k] for k in ('ID','Language_ID','Form','Gloss','Source','Tags')} for r in csv.DictReader(open('cldf/forms.csv')) if 'rezaibaghbidi2003zargari[' in r['Source']]
assigned={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
('kʰām','sun','4445','gharma','Turner explicitly gives European Romani kham sun under gharma heat, alongside Romani gam sun. The aspirated voiceless initial and sun sense support this Romani family rather than a sūrya-based assignment.'),
('cʰon','moon','5301','jyotsnā','Turner section1 explicitly gives southeastern European Romani čhon moon, beside European čon and Welsh šōn. This selects jyotsnā rather than the separate *yotsnā subsection.'),
('pʰus','straw','9293-2','*bhusa','Turner section2 explicitly gives European and Armenian Romani phus straw under *bhusa. The addendum to9100 *phussa explicitly prefers *bhusa for the Romani word; the competing similar-looking defective-word entry is therefore not selected.'),
('bār','stone','11348','*varta','Turner section1 explicitly gives European Romani bar stone and discusses the reconstructed round-stone family. The uncertain deeper history and Kurdish comparison do not establish a recent donor for Zargari; section2 *vārta is the distinct Pashai/Shumashti branch.'),
('nāk','nose','6909','*nakka','Turner explicitly gives European Romani nak and explains the source-cited Sanskrit nakra/nakrā as Sanskritizations of Prakrit ṇakka. Use the *nakka parent while retaining the article’s original comparison in the source metadata; Hindi nāk is a comparison, not an asserted donor.'),
('kān','ear','2830','karṇa','The source explicitly compares Sanskrit karṇa and Hindi kān. Turner gives European Romani kan under karṇa, supporting the inherited Romani ear family without treating Hindi as an immediate donor.'),
('juv','louse','10512','yūkā','Turner explicitly gives European Romani ǰuv louse under yūkā. The source džuv is normalized to juv by its existing profile; phonetic [džuw] does not create another lexical stem.'),
('dād','father','6261','*dādda','Turner explicitly gives German Romani dād father under the expressive *dādda kinship family. The link identifies that family without claiming a unique reconstruction of an expressive nursery word.'),
('mitér','urine','10234','mūtra','Turner explicitly gives European Romani muter and Welsh mutār urine under mūtra. Zargari mitér matches the Romani consonantal structure and disyllabic form with a different first vowel; preserve the source i and final stress rather than silently replacing them with the comparandum.')]
a=[]
for f,g,pa,label,ev in groups:
 for r in rs:
  if r['ID'] not in assigned and r['Form']==f and r['Gloss']==g:a.append(dict(record=r,parent=pa,kind='reflex',citation=f'CDIAL[{pa} and addenda];'+r['Source'],evidence=ev+' Source glyph-decoded reading and grammar-example citation preserved.'))
h=[]
reasons={'dār':'Turner6651 *dvara includes a warning about Persian borrowing and no direct Romani dār witness;6459 instead gives Romani vudar. Resolve the Iranian versus inherited history before choosing a parent.','tʰāv':'Turner6203 tentatively compares Romani thav thread with darbha grass;6770 proposes *dhāgga crossed with dāman, with a further reconstruction revision in its addendum. The competing analyses need adjudication rather than an unqualified reflex link.','gād':'No sufficient direct dictionary match for shirt emerged from the homepage searches; source gives the form without etymology. Broader Romani/Iranian evidence is required.'}
for r in rs:
 if r['ID'] not in assigned and r['Form'] in reasons:h.append(dict(record=r,reason=reasons[r['Form']]))
assert len(a)==9,len(a)
(P/'pass027-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=2)+'\n')
lines=['# Zargari: additional inherited nouns','','9 records / 9 reflex rows across nine etyma; three unresolved records held. Already-linked rows excluded using compiled graph and accepted overlays.','','| # | Zargari | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(f,g,pa,label,ev) in enumerate(groups,1):
 label=label.replace('*',r'\*');lines.append(f'| {i} | **{f}** ‘{g}’ | **{label}**, CDIAL **{pa}** | {ev} |')
lines+=['','Holds: door dār needs Iranian/inherited-route discrimination; string thāv has competing darbha versus *dhāgga-with-dāman analyses; shirt gād lacks sufficient direct evidence. Earlier variant/loan holds remain.','', 'Evidence: source glyph-decoded audit20260825-rezai-baghbidi-zargari-audit.csv, relevant source-locator rows copied into pass027-source-audit.json. Source Hindi/Sanskrit comparisons are not converted into claims of immediate borrowing. Search003–005 and primary003–004 preserve discovery and full primary prose/addenda. No source reparsing, identity edits or rebuild.','', 'Validation and save status: pending.'];(P/'pass027-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(h),'held')
