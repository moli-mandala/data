"""Reproduce both seeded PDF-to-draft audits and whole-source accounting."""
import csv, json, random, unicodedata
from pathlib import Path
from segments import Tokenizer

ROOT=Path(__file__).resolve().parent

def expected_sample_value(record, field):
    # Historical samples remain immutable. Later source-description review adds
    # only explicit factual restrictions and grammar to their original readings.
    amendments={r['entry_key']:r for r in json.loads((ROOT/'description-review-20260926.json').read_text())['records']}
    amendment=amendments.get(record['entry_key'],{})
    if field=='notes':
        return ' '.join([record[field]]+amendment.get('notes',[])).strip()
    if field=='tags':
        return ' '.join(dict.fromkeys(record[field].split()+amendment.get('tags',[])))
    return record[field]

def verify():
    sample=json.loads((ROOT/'output-sample-2026092205.json').read_text())
    keys=random.Random(sample['seed']).sample(sample['population_keys'],20)
    assert keys==[r['entry_key'] for r in sample['records']]
    with (ROOT/'review-draft.csv').open() as stream:
        draft=list(csv.reader(stream))
    rows={r[10]:r for r in draft}
    raw=[json.loads(s) for s in (ROOT/'positioned-candidates.jsonl').read_text().splitlines()]
    native=[json.loads(s) for s in (ROOT/'native-recovery-review.jsonl').read_text().splitlines()]
    assert len(draft)==len(rows)==len(raw)==len(native)==2451
    assert all(len(r)==15 for r in draft)
    assert set(rows)=={r['entry_key'] for r in raw}=={r['entry_key'] for r in native}
    assert all(r[0]=='Mahali' and r[4] for r in draft)
    tokenizer=Tokenizer(str(ROOT/'cfel-mahali.txt'))
    for record in sample['records']:
        row=rows[record['entry_key']]
        display=unicodedata.normalize('NFC',tokenizer(row[2],column='IPA').replace(' ','').replace('#',' '))
        values={'source_ipa':row[2],'display':display,'gloss':row[3],
                'native':row[4],'tags':row[14],'citation':row[7],'notes':row[6]}
        assert all(expected_sample_value(record,k)==v for k,v in values.items()),record['entry_key']
        assert record['status']=='visually audited' and record['material_errors']==0
    whole=json.loads((ROOT/'output-sample-20260925.json').read_text())
    assert whole['population']==len(raw) and whole['sample']==20
    assert whole['material_errors']==0
    assert whole['pdf_sha256']==json.loads((ROOT/'source-manifest.json').read_text())['pdf_sha256']
    keys=random.Random(whole['seed']).sample([r['entry_key'] for r in raw],20)
    assert keys==[r['entry_key'] for r in whole['records']]
    by_raw={r['entry_key']:r for r in raw}
    for record in whole['records']:
        key=record['entry_key']; row=rows[key]; source=by_raw[key]
        display=unicodedata.normalize('NFC',tokenizer(row[2],column='IPA').replace(' ','').replace('#',' '))
        values={'source_ipa':row[2],'display':display,'gloss':row[3],
                'native':row[4],'tags':row[14],'citation':row[7],'notes':row[6]}
        assert all(expected_sample_value(record,k)==v for k,v in values.items()),key
        assert all(record[k]==source[k] for k in ('pdf_page','printed_page','page_item','raw_lexical_block')),key
        assert record['status']=='visually audited against printed PDF' and record['material_errors']==0
    return sample,whole

if __name__=='__main__':
    subset,whole=verify()
    print(f"2451 rows accounted for; frozen subset {len(subset['records'])}/190 and fresh whole-source {whole['sample']}/2451 audits reproduce; 0 material errors")
