"""Stage both complete paired sources, retaining every installed glossary row.

No install option. Root independent source/output audit precedes installation.
"""
import csv
import hashlib
import json
import unicodedata
from pathlib import Path

P=Path(__file__).resolve().parent
D=P.parents[4]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def extend_profile(path,forms):
    mapping=dict(csv.reader(path.open(),delimiter='\t'))
    assert mapping.pop('Grapheme')=='IPA'
    clusters=set()
    for form in forms:
        cluster=''
        for ch in form:
            if unicodedata.combining(ch) and cluster:cluster+=ch
            else:
                if cluster:clusters.add(cluster)
                cluster=ch
        if cluster:clusters.add(cluster)
    for cluster in clusters:
        nfc=unicodedata.normalize('NFC',cluster)
        output=mapping.get(nfc,nfc.lower().replace('w','v').replace('ṅ','ŋ'))
        if nfc==' ':output='#'
        for spelling in {nfc,unicodedata.normalize('NFD',nfc)}:
            if spelling in mapping:assert mapping[spelling]==output,(spelling,mapping[spelling],output)
            mapping[spelling]=output
    return 'Grapheme\tIPA\n'+''.join(k+'\t'+v+'\n' for k,v in sorted(mapping.items()))


def main():
    expanded=P/'whole-paired-expansion-draft-20260927.jsonl'
    units=[json.loads(x) for x in expanded.read_text().splitlines()]
    assert len(units)==395
    reports=[]
    for name,lang,count in [('rampur','ramp',288),('baghi','ba',296)]:
        package=P.parent/f'bailey_{name}_1920';source=f'bailey1920{name}'
        legacy_path=package/'legacy-before-whole-recovery.csv'
        legacy=list(csv.reader(legacy_path.open()))
        assert len(legacy)==count
        canonical=D/f'data/other/forms/20260925-bailey-{name}.csv'
        assert list(csv.reader(canonical.open()))==legacy,'Canonical changed after snapshot; reconcile explicitly before staging.'
        profile=package/'legacy-before-whole-recovery-profile.txt'
        original_profile=D/f'conversion/bailey-{name}-1920.txt'
        if not profile.exists():profile.write_bytes(original_profile.read_bytes())
        assert profile.read_bytes()==original_profile.read_bytes(),'Legacy profile changed; no silent override.'
        rows=[r[:] for r in legacy]
        audit=[json.loads(x) for x in (package/'legacy-before-whole-recovery-audit.jsonl').read_text().splitlines()]
        corrections=json.loads((P/'whole-legacy-corrections-20260927-pass2.json').read_text())
        own_corrections=[c for c in corrections['changes'] if c['package']==name]
        for change in own_corrections:
            index=next(i for i,r in enumerate(rows) if r[10]==change['entry_key'])
            assert rows[index]==change['before'],change['entry_key']
            rows[index]=change['after'][:]
            cell=change['entry_key'].split(':answer')[0]
            item=next(a for a in audit if a.get('source_cell_key')==cell)
            item.setdefault('whole_source_legacy_corrections',[]).append(change)
        corrected_legacy=[r[:] for r in rows]
        if name=='rampur':
            recovery=json.loads((P/'whole-glossary-hold-recovery-20260927.json').read_text())
            matches=[(i,a) for i,a in enumerate(audit) if a.get('source_cell_key')==recovery['source_cell_key']]
            assert len(matches)==1 and matches[0][1]['status']=='hold_typography'
            i,old=matches[0]
            key=recovery['source_cell_key']
            rows.append(['ramp','',recovery['form'],recovery['gloss'],'','',recovery['notes'],'bailey1920rampur[p. 144, vocabulary item 19, Rampur column]','','',key,'','','',' '.join(recovery['tags'])])
            audit[i]={**old,'status':'ingested','entry_keys':[key],'visual_reading':recovery['form'],'uncertainty':'source-typography interpretation','reason':recovery['notes'],'historical_hold':old,'whole_source_recovery':recovery}
        for unit in units:
            keys=[];emissions=[]
            for c in unit['candidates']:
                if c['lang']!=lang and not (name=='rampur' and c['lang']=='kc'):continue
                tags=c['tags'][:];notes=c['notes'][:]
                if c['typed_uncertainty']:
                    tags+=['uncertain'];notes+=['Source uncertainty: '+u for u in c['typed_uncertainty']]
                if c.get('variant_of_key'):tags+=['alternate']
                if c['expansion']:
                    notes+=['Printed abbreviated form: '+c['raw_cell']+'. '+ ' '.join(c['expansion'])]
                locator=f"p. {unit['printed_page']}, {unit['source_unit_key'].split(':',1)[1]}"
                if unit['source_kind']=='sentence' and unit['source_review']['item']=='12' and c['lang']=='ramp':locator='pp. 141–142, sentence 12 Rampuri'
                assert ';' not in locator
                row=[c['lang'],'',unicodedata.normalize('NFC',c['form']),c['gloss'],'','',' '.join(dict.fromkeys(n for n in notes if n)),f'{source}[{locator}]','','',c['candidate_key'],c.get('variant_of_key',''),'','',' '.join(dict.fromkeys(tags))]
                rows.append(row);keys.append(c['candidate_key']);emissions.append(dict(entry_key=c['candidate_key'],form=row[2],gloss=row[3],tags=row[14],notes=row[6],source=row[7]))
            audit.append({**unit,'exported_entry_keys':keys,'exported_rows':emissions,'package_disposition':'target rows staged' if keys else 'shared audit evidence; no target rows for this package','proposal_status':'complete-scope draft; independent actual-output audit pending'})
        assert rows[:count]==corrected_legacy
        assert [r[10] for r in rows[:count]]==[r[10] for r in legacy]
        keys={r[10] for r in rows}
        assert len(keys)==len(rows)
        assert all(not r[11] or (r[11] in keys and r[11]!=r[10]) for r in rows)
        with (package/'whole-proposed.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
        (package/'whole-proposed-audit.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in audit))
        (package/'whole-proposed-profile.txt').write_text(extend_profile(profile,[r[2] for r in rows]))
        report=dict(package=package.name,status='complete-scope proposal; not installed',rows=len(rows),legacy_rows_preserved=count,added_rows=len(rows)-count,recovered_glossary_holds=int(name=='rampur'),audit_units=len(audit),generic_kochi_rows=sum(r[0]=='kc' for r in rows),uncertain_rows=sum('uncertain' in r[14].split() for r in rows),variant_rows=sum(bool(r[11]) for r in rows),input_expansion_sha256=sha(expanded),files={n:sha(package/n) for n in ['whole-proposed.csv','whole-proposed-audit.jsonl','whole-proposed-profile.txt','legacy-before-whole-recovery.csv','legacy-before-whole-recovery-profile.txt']},required_yaml={'defaults.transcription':'inputform','preserve_hyphens':True},remaining=['owner expansion/metadata review','focused regressions and actual parser checks','independent complete-output audit','metadata and source-stage installation','full DB/CLDF/build/browser gates explicitly deferred'])
        (package/'whole-proposal-progress-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');reports.append(report)
    print(json.dumps([{k:v for k,v in r.items() if k!='files'} for r in reports],ensure_ascii=False))


if __name__=='__main__':main()
