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
    ('bəl', 'strength', '9161', 'bala', 'Dhakal p. 44 explicitly contrasts central-vowel /bəl/ strength with /bal/ light. Turner9161 gives Pali/Prakrit bala strength and Nepali, Maithili and Bihari bal; choose only the strength homonym and preserve the source vowel distinction.'),
    ('ʦʰõṭo', 'short', '5071', '*chōṭṭa', 'Dhakal p. 46 prints /cʰõṭo/ short and describes its nasal vowel. Turner5071 gives Kumaoni choṭo, Kashmiri ċhoṭu short/small and Nepali choṭo small under *chōṭṭa. The source nasalization remains as printed; the data do not determine a contact-free path.'),
    ('naŋṭəi', 'naked', '6926', 'nagna', 'Dhakal p. 61 prints /naŋ.ṭəi/ naked. Turner6926 gives Maithili nā̃gaṭ, Hindi nãgṭā and Bengali neṅṭā, and explicitly notes deformations in this word family. The Darai -əi ending is preserved without claiming a specific historical suffix.'),
    ('dubro', 'thin', '6438', 'durbala', 'Dhakal p. 68 prints /dub.ro/ thin. Turner6438 gives Hindi dūbrā and Awadhi dūbar beside Nepali dublo and Bhojpuri dū̆bar under durbala. The r-bearing Darai outcome has regional comparanda, so it is not inferred from Nepali l by itself.'),
    ('galu', 'cheek', '4089', 'galla', 'Dhakal p. 67 prints /ga.lu/ cheek. Turner4089 gives Prakrit galla, Old Awadhi gālu, Oriya gāla and Bihari/Maithili gāl. The exact vowel ending is visible in the Old Awadhi comparison; this is the cheek family, not gala throat.'),
]
accepted = []
audit = []
review = ['# Darai: descriptors and cheek', '', 'Five distinct source-backed reflex records, preserving the source vowel and consonant contrasts.', '', '| # | Darai form | Proposed etymology | Evidence |', '|---|---|---|---|']
source_dir = Path('data/other/forms/raw_data/dhakal_darai_2011/decoded')
for i, (form, gloss, parent, label, evidence) in enumerate(groups, 1):
    hits = [r for r in rows if r['Form'] == form and r['Gloss'] == gloss]
    assert len(hits) == 1, (form, gloss, len(hits))
    r = hits[0]
    page = int(r['Source'].split('[p. ')[1].split(']')[0])
    source_text = (source_dir / f'p{page:03}.txt').read_text()
    assert gloss in source_text, (form, page)
    audit.append(dict(id=r['ID'], form=form, gloss=gloss, source=r['Source'], page=page, primaryText=source_text))
    accepted.append(dict(record=rec(r), parent=parent, kind='reflex', citation=r['Source'] + f';CDIAL[{parent}]', evidence=evidence))
    review.append(f'| {i} | **{form}** ‘{gloss}’ | **{label}**, CDIAL **{parent}** | {evidence} |')

holds = []
for form, reason in [
    ('ramrə', 'Turner10636 places Nepali rāmro good in a -ḍ extension, but Darai ramrə might be a Nepali contact form; the source does not distinguish inheritance and borrowing.'),
    ('bəḍkə', 'Lakhimpuri baṛkā big is associated with inferred CDIAL11225-2 *vaḍrakk-; the Darai -kə extension needs its own morphological comparison before choosing this node over 11225.'),
    ('səmmə', 'Turner13173 samá has level/flat comparanda but also explicitly labels Pashai sam a Pashto loan; Darai doubled -mm- and its immediate history require local comparison.'),
]:
    for r in rows:
        if r['Form'] == form: holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 5 and len(holds) == 3
(P / 'pass076-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass076-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Five affected records and five reflex rows. Held: *ramrə* good, *bəḍkə* big and *səmmə* flat for contact or extension history.', 'Primary source: Dhakal 2011 printed pp. 44, 46, 61, 67–68, visually checked in source PDF images and decoded text. Discovery: homepage darai-search017–018; complete CDIAL articles darai-primary012–015. No source, identity or build changes.', 'Validation: pending.']
(P / 'pass076-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
