"""Stage the whole Bilaspuri proposal, retaining the installed glossary exactly.

This has no install mode. Independent metadata/output review is still required.
"""
import csv,hashlib,json,unicodedata
from pathlib import Path

P=Path(__file__).resolve().parent
D=P.parents[4]
SOURCE='bailey1920bilaspuri'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths={
        'legacy-before-whole-recovery.csv':D/'data/other/forms/20260925-bailey-bilaspuri.csv',
        'legacy-before-whole-recovery-audit.jsonl':P/'audit.jsonl',
        'legacy-before-whole-recovery-profile.txt':D/'conversion/bailey-bilaspuri-1920.txt',
    }
    for name,source in paths.items():
        target=P/name
        if not target.exists():target.write_bytes(source.read_bytes())
    legacy=list(csv.reader((P/'legacy-before-whole-recovery.csv').open()))
    assert len(legacy)==288
    legacy_audit=[json.loads(s)for s in(P/'legacy-before-whole-recovery-audit.jsonl').read_text().splitlines()]
    assert len(legacy_audit)==237
    expanded_path=P/'whole-expansion-draft-20260927.jsonl'
    metadata_summary=json.loads((P/'whole-grammar-metadata-summary-20260927.json').read_text())
    assert metadata_summary['input_sha256']==sha(expanded_path),'Metadata is stale'
    units=[json.loads(s)for s in expanded_path.read_text().splitlines()]
    metadata=[json.loads(s)for s in(P/'whole-grammar-metadata-proposal-20260927.jsonl').read_text().splitlines()]
    keyed={r['candidate_key']:r for r in metadata}
    assert len(keyed)==len(metadata)
    rows=[r[:]for r in legacy];audit=list(legacy_audit)
    for unit in units:
        entry_keys=[]
        for candidate in unit['candidates']:
            m=keyed[candidate['candidate_key']]
            tags=m['tags'][:]
            notes=m['notes'][:]
            uncertainty=m.get('typed_uncertainty',[])
            if uncertainty:
                tags+=['uncertain']
                notes+=['Source uncertainty: '+u for u in uncertainty]
            assert ';'not in m['candidate_key']
            locator=f"p. {unit['printed_page']}, {unit['source_unit_key'].split(':',1)[1]}"
            parent=m.get('variant_of_key','')
            if parent:tags+=['alternate']
            rows.append(['bil','',unicodedata.normalize('NFC',m['form']),m['gloss'],'','',' '.join(notes),f'{SOURCE}[{locator}]','','',m['candidate_key'],parent,'','',' '.join(dict.fromkeys(tags))])
            entry_keys.append(m['candidate_key'])
        audit.append({**unit,'exported_entry_keys':entry_keys,'proposal_status':'not independently accepted; source/metadata/output review pending'})
    assert rows[:288]==legacy
    assert len({r[10]for r in rows})==len(rows)
    keys={r[10]for r in rows}
    assert all(not r[11]or (r[11]in keys and r[11]!=r[10])for r in rows)
    with(P/'whole-proposed.csv').open('w',newline='')as f:csv.writer(f).writerows(rows)
    (P/'whole-proposed-audit.jsonl').write_text(''.join(json.dumps(u,ensure_ascii=False)+'\n'for u in audit))
    # Preserve all existing mappings; extend composed and decomposed grapheme
    # spellings under that policy. No inferred phonemic transcription.
    mapping=dict(csv.reader((P/'legacy-before-whole-recovery-profile.txt').open(),delimiter='\t'))
    mapping.pop('Grapheme')
    clusters=set()
    for row in rows:
        current=''
        for ch in row[2]:
            if unicodedata.combining(ch)and current:current+=ch
            else:
                if current:clusters.add(current)
                current=ch
        if current:clusters.add(current)
    for cluster in clusters:
        normalized=unicodedata.normalize('NFC',cluster)
        output=mapping.get(normalized,normalized.lower().replace('w','v').replace('ṅ','ŋ'))
        if normalized==' ':output='#'
        for spelling in {normalized,unicodedata.normalize('NFD',normalized)}:
            if spelling in mapping:assert mapping[spelling]==output,(spelling,mapping[spelling],output)
            mapping[spelling]=output
    (P/'whole-proposed-profile.txt').write_text('Grapheme\tIPA\n'+''.join(k+'\t'+v+'\n'for k,v in sorted(mapping.items())))
    names=[*paths,'whole-expansion-draft-20260927.jsonl','whole-grammar-metadata-proposal-20260927.jsonl','whole-proposed.csv','whole-proposed-audit.jsonl','whole-proposed-profile.txt']
    report=dict(status='whole-source draft proposal; not installed or accepted',rows=len(rows),audit_units=len(audit),legacy_rows_preserved=288,new_candidate_rows=len(rows)-288,source_units=len(units),input_sha256={n:sha(P/n)for n in names},
                remaining=['complete independent original reading','candidate grammar/gloss and variant interpretation review','profile policy and NFC/NFD coverage checks','independent20-unit output audit','actual-parser checks and canonical installation','full compiled/build/browser gates deferred'])
    (P/'whole-proposal-progress-20260927.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k!='input_sha256'}))


if __name__=='__main__':main()
