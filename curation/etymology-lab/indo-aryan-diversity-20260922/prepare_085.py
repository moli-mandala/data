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
    ('bʰəike', 'to become', '9416', 'bhávati',
     'Dhakal p. 48 prints /bʰəike/ “to become” beside a separately negated form. Turner9416 gives Pali bhavati, Prakrit bhavaï, Ḍoḍi bhōṇō and other regional verbs “become”; Darai -ike is the regular infinitive, and its bhəi- stem matches the Middle Indo-Aryan bhavai family.'),
    ('ʣiuno', 'living', '5244', 'jīvantá',
     'Dhakal p. 54 prints /dziuno/ “living.” Turner5244 is the living participial family, with Kumaoni jyūno, Nepali jiũdo and Kotgarhi j̈iundɔ “living, alive.” These adjective comparanda select 5244 rather than noun jīva “life” (5239) or verbal jīvati (5241).'),
    ('manus', 'man', '9827', 'mánuṣa',
     'Dhakal p. 54 gives /manus/ “man.” Turner9827 lists Prakrit manusa “man” and the inherited regional mánuṣa family; the Darai final s with final-vowel loss matches this branch better than the separate short mánus head 9825, whose cited NIA forms are manua/manū.'),
    ('ʣəri', 'root', '5086', 'jáṭā',
     'Dhakal p. 56 prints /dzəri/ “root.” Turner5086 section 1 lists Nepali jari, Bihari jar, Maithili jaṛi and Oriya jaṛa with the same meaning. This identifies the regional root family, while Turner leaves its remoter Dravidian versus Munda history unsettled.'),
    ('aʣu', 'today', '242', 'adyá',
     'Dhakal pp. 65 and 69 print /adzu/ “today.” Turner242 specifically lists Prakrit ajja, Apabhraṃśa ajju, Sindhi aju and Nepali āju, matching the Darai voiced affricate and final u. This is the adyá branch, not adyāpi “even now” (243).'),
]
accepted, audit = [], []
review = ['# Darai: become, living, man, root and today', '',
          'Six source-backed reflex records across five etyma.', '',
          '| # | Darai form | Proposed etymology | Evidence |', '|---|---|---|---|']
source_dir = Path('data/other/forms/raw_data/dhakal_darai_2011/decoded')
i = 0
for form, gloss, parent, label, evidence in groups:
    hits = [r for r in rows if r['Form'] == form and r['Gloss'] == gloss]
    assert len(hits) == (2 if form == 'aʣu' else 1), (form, gloss, len(hits))
    for r in hits:
        i += 1
        page = int(r['Source'].split('[p. ')[1].split(']')[0])
        source_text = (source_dir / f'p{page:03}.txt').read_text()
        assert gloss in source_text, (form, page)
        audit.append(dict(id=r['ID'], form=form, gloss=gloss, source=r['Source'], page=page,
                          sourceText=source_text))
        accepted.append(dict(record=rec(r), parent=parent, kind='reflex',
                             citation=r['Source'] + f';CDIAL[{parent}]', evidence=evidence))
        review.append(f'| {i} | **{form}** ‘{gloss}’ ({r["Source"]}) | **{label}**, CDIAL **{parent}** | {evidence} |')

holds = []
for form, reason in [
    ('nabʰəi', 'The source prints a negative finite predicate “does not become”; segment and classify the na- negator before representing this multi-morpheme response.'),
    ('pʰokso', 'Turner9090 is a plausible expressive lung family, but does not explain the Darai -kso cluster or establish its transmission route.'),
]:
    for r in rows:
        if r['Form'] == form:
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 6 and len(holds) == 2, (len(accepted), len(holds))
(P / 'pass085-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass085-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Six affected records and six reflex rows. Held: negated *nabʰəi* and *pʰokso* lung. The prior pass081 ambiguities remain excluded. No source, identity or build changes.',
           'Primary source: Dhakal 2011 printed pp. 48, 54, 56, 65 and 69, checked in source PDF images and decoded text. Discovery: darai-search023–024; complete CDIAL articles darai-primary021–023.',
           'Validation: pending.']
(P / 'pass085-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
