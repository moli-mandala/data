"""Stage the complete Fawcett article inventory; never install canonical inputs."""
import csv,hashlib,json,unicodedata
from pathlib import Path
P=Path(__file__).resolve().parent
SOURCE='fawcett1888saora'
def read(name):return [json.loads(x) for x in (P/name).read_text().splitlines() if x]
def build():
    units=read('full-reviewed-inventory.jsonl');pages=read('article-page-census-first-reading-20260927.jsonl')
    assert sorted(x['printed_page'] for x in pages)==list(range(206,273))
    assert len(units)==len({u['source_unit_key'] for u in units})==164
    rows=[];audit=[]
    for u in units:
        a={**u,'entry_keys':[]}
        if u['scope']!='Saora':
            a['disposition']='audit-only-explicit-control-or-documentary-evidence';audit.append(a);continue
        assert u['forms'] and u['gloss']
        for i,form in enumerate(u['forms'],1):
            key=u['source_unit_key']+(f':alternate:{i}' if i>1 else '')
            tags=list(u['tags'])
            if ' ' in form:tags+=['multiword-expression']
            if i>1:tags+=['alternate']
            notes=u['notes']
            locator=f"p. {u['printed_page']}, {u['section']}, {u['source_unit_key'].rsplit(':',1)[1]}"
            assert ';' not in locator
            rows.append(['so','',unicodedata.normalize('NFC',form),u['gloss'],'','',notes,f'{SOURCE}[{locator}]','','',key,u['source_unit_key'] if i>1 else '','','',' '.join(dict.fromkeys(tags))])
            a['entry_keys'].append(key)
        a['disposition']='emitted-target-attestation';audit.append(a)
    with (P/'proposal.csv').open('w',newline='') as f:csv.writer(f,lineterminator='\n').writerows(rows)
    (P/'proposal-audit.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False,sort_keys=True)+'\n'for a in audit))
    graphemes=set()
    for r in rows:
        cluster=''
        for ch in r[2]:
            if unicodedata.combining(ch) and cluster:cluster+=ch
            else:
                if cluster:graphemes.add(cluster)
                cluster=ch
        if cluster:graphemes.add(cluster)
    # Literal historical notation, not an asserted IPA reconstruction. House w→v
    # only; vowel marks and printed internal syllable commas remain distinguishable.
    mappings={}
    for c in graphemes:
        value='#' if c==' ' else unicodedata.normalize('NFC',c.lower().replace('w','v'))
        for variant in {unicodedata.normalize('NFC',c),unicodedata.normalize('NFD',c)}:
            assert variant not in mappings or mappings[variant]==value
            mappings[variant]=value
    (P/'proposal-profile.txt').write_text('Grapheme\tIPA\n'+''.join(c+'\t'+mappings[c]+'\n'for c in sorted(mappings)))
    names=['full-reviewed-inventory.jsonl','proposal.csv','proposal-audit.jsonl','proposal-profile.txt','article-page-census-first-reading-20260927.jsonl','owner-reconciliation-20260927.json','kinship-owner-reconciliation-20260927.json','20260927-fawcett-saora.yaml','reference-proposal.bib','root-numerals-second-reading-20260927.json','root-kinship-targeted-review-20260927.json']
    report=dict(status='whole-article proposal staged; independent final audit and canonical installation pending',article_pages=67,units=len(audit),rows=len(rows),target_units=sum(bool(a['entry_keys'])for a in audit),audit_only_units=sum(not a['entry_keys']for a in audit),variant_edges=sum(bool(r[11])for r in rows),native='No native script printed; no Native or Phonemic supplied.',reuse='No automatic same-form merge: scholarly notes, speaker labels and contexts remain distinct.',unresolved_english_ditto_units=3,trailing_possessive_gloss_units=1,hashes={n:hashlib.sha256((P/n).read_bytes()).hexdigest()for n in names})
    (P/'proposal-summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':build()
