import csv
import json
from pathlib import Path

from etymology_assignments import read_assignments

P = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
linked = {r['Form_ID'] for r in read_assignments() if r['Status'] == 'accepted' and r['Rank'] == '1'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
rows = [r for r in csv.DictReader(open('cldf/forms.csv')) if r['Source'].startswith('dhakal2011darai[') and r['ID'] not in linked]
rec = lambda r: {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')}
groups = [
    ('anike', 'to bring', '1174', 'ā́nayati', 'Dhakal p. 49 prints /anike/ “to bring”. Turner1174 section 1 gives Pali āneti “brings, fetches”, Bhojpuri ānal, Old Awadhi ānab and Hindi ānnā. The Darai -ike is the ordinary infinitive suffix, illustrated in the source morphology. The nonlong a is consistent with the source statement that vowel length is not distinctive.'),
    ('pʰerike', 'to change', '9078-2', '*phērayati', 'Dhakal pp. 45 and 51 print /pʰerike/ “to change” in minimal pairs. Turner9078 section 2 is transitive *phērayati; it specifically lists Nepali phernu “to change” and Old Awadhi pheraï “turns, changes”. This contrasts with the separate Darai /pʰorike/ “to break”.'),
    ('pʰorike', 'to break', '13857', 'sphōṭayati', 'Dhakal p. 45 prints /pʰorike/ “to break” opposite /pʰerike/. Turner13857 gives Prakrit phōḍēi and Maithili phorab, Awadhi phorab “to break”, with Nepali phornu. The r-bearing regional forms support the Darai form.'),
    ('tisrən', 'third', '6018', '*triḥsara', 'Dhakal pp. 53 and 67 print /tisrən/ “third” (the latter syllabified tis.rən). Turner6018 section 1 explicitly gives Tarai Nepali tisar and Awadhi tīsar “third”, alongside Hindi tīsrā; the Darai ending is retained without projecting an unattested ancestral suffix.'),
    ('pãʦəvə', 'fifth', '7669', 'pañcamá', 'Dhakal p. 46 prints /pãcəwə/ “fifth” in the nasal-vowel list; the CLDF form represents the affricate and approximant as pãʦəvə. Turner7669 section 1 gives Old Awadhi pāṁcavaṁ and Lakhimpuri pãcawā̃ alongside Nepali pā̃caũ. This is the ordinal family, distinct from Darai pãʦ “five” already linked to CDIAL7655.'),
]
accepted = []
audit = []
review = ['# Darai: bring, change, break and ordinals', '', 'Seven source-backed reflex records across five lexical families.', '', '| # | Darai form | Proposed etymology | Evidence |', '|---|---|---|---|']
source_dir = Path('data/other/forms/raw_data/dhakal_darai_2011/decoded')
i = 0
for form, gloss, parent, label, evidence in groups:
    hits = [r for r in rows if r['Form'] == form and r['Gloss'] == gloss]
    assert len(hits) in (1, 2), (form, gloss, len(hits))
    for r in hits:
        i += 1
        page = int(r['Source'].split('[p. ')[1].split(']')[0])
        source_text = (source_dir / f'p{page:03}.txt').read_text()
        assert gloss in source_text, (form, page)
        audit.append(dict(id=r['ID'], form=form, gloss=gloss, source=r['Source'], page=page, primaryText=source_text))
        accepted.append(dict(record=rec(r), parent=parent, kind='reflex', citation=r['Source'] + f';CDIAL[{parent}]', evidence=evidence))
        review.append(f'| {i} | **{form}** ‘{gloss}’ ({r["Source"]}) | **{label}**, CDIAL **{parent}** | {evidence} |')

holds = []
for form, reason in [
    ('aike', 'The “to come” verb may belong to the regional ā- family, but no precise Turner branch has yet been established.'),
    ('tapike', 'The source only glosses “to warm”; Turner5684 comparanda often mean “warm oneself”. Check transitivity before assigning.'),
    ('dudʰaili', 'The milk-giving cow has an additional suffix; CDIAL6399 dugdhin is semantically close but the immediate Darai derivation needs checking.'),
]:
    for r in rows:
        if r['Form'] == form:
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 7 and len(holds) == 3, (len(accepted), len(holds))
(P / 'pass080-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass080-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Seven affected records and seven reflex rows. Held: *aike* come, *tapike* warm, *dudʰaili* milk-giving cow for branch or morphology uncertainty.', 'Primary source: Dhakal 2011 printed pp. 45–46, 49, 51, 53, 67, checked in source PDF images and decoded text. Discovery: darai-search019; complete CDIAL articles darai-primary016. No source, identity or build changes.', 'Validation: pending.']
(P / 'pass080-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
