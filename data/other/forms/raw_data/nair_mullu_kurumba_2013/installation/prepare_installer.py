"""Standalone disabled transaction, based on the reviewed four-target engine."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
template=DATA/'data/other/forms/raw_data/nair_mannan_2013/installation/install_source.py'
assert hashlib.sha256(template.read_bytes()).hexdigest()=='4f0deead86e0ad073d33a005382e8dc5fa76b933fbbff718f25534fdf895db4b'
assert not (HERE/'root-acceptance.json').exists()
text=template.read_text().replace('Mannan','Mullu Kurumba').replace('nair_mannan_2013','nair_mullu_kurumba_2013').replace('nair-mannan','nair-mullu-kurumba').replace('nair2013mannan','nair2013mullukurumba').replace('joseph1980devikolam','robertsathyajoseph1982mullukurumba')
text=text.replace('154','135').replace('140','65').replace(".mannan-", ".mullu-")
text=text.replace("    assert set(approval['assets']) == set(TARGETS)", "    assert approval.get('shared_baseline_captured') is True, 'Shared baseline awaits root-reserved write window'\n    assert approval.get('audit_acceptance_ready') is True, 'Audit acceptance awaits root source-stage decision'\n    assert set(approval['assets']) == set(TARGETS)")
start,end=text.index('def validate_assets('),text.index('\ndef shared_membership():')
text=text[:start]+'''def validate_assets(approval, planned):
    rows=list(csv.reader(io.StringIO(planned['csv'].decode())))
    assert len(rows)==approval['rows']==135
    assert all(len(r)==15 and r[0] in {'Mullu Kurumba','Malayalam'} and r[2] and r[10] for r in rows)
    assert sum(r[0]=='Mullu Kurumba' for r in rows)==110
    assert sum(r[0]=='Malayalam' for r in rows)==25
    keys={r[10] for r in rows}
    assert len(keys)==135
    for row in rows:
        assert not any(row[i] for i in (1,4,5,8,12,13))
        assert not row[11] or row[11] in keys
        assert unicodedata.is_normalized('NFC',row[2])
        assert all(unicodedata.category(c)!='Co' and c!='\\ufffd' for c in row[2])
        assert row[7].startswith('nair2013mullukurumba[p. ')
    assert sum(bool(r[11]) for r in rows)==11
    audit=list(map(json.loads,planned['audit'].decode().splitlines()))
    assert len(audit)==len({a['audit_key'] for a in audit})==approval['audit_units']==65
    references=[k for a in audit for k in a['entry_keys']]
    assert len(references)==141 and len(set(references))==135 and set(references)==keys
    assert all(a['accepted_for_installation'] is True and 'source_record' in a for a in audit)
    witnesses=[w for a in audit for w in a['source_record']['witness_dispositions']]
    assert len(witnesses)==164 and len({w['witness_key'] for w in witnesses})==164
    assert sum(w['disposition']=='exact_repeated_attestation_reuse' for w in witnesses)==6
    assert sum(not w['entry_keys'] for w in witnesses)==23
    assert [a['source_record'] for a in audit]==list(map(json.loads,(PACKAGE/'proposal-audit.jsonl').read_text().splitlines()))
    settings=yaml.safe_load(planned['yaml'])
    assert settings['file']==STEM+'.csv'
    assert settings['defaults']['transcription']=={'convert':False,'preserve_hyphens':True}
    assert settings['defaults']['identity']['dedupe_by_entry_key'] is True
    assert settings['defaults']['forms']['split_alternates'] is False
    assert settings['defaults']['gloss']['grammar_tags'] is False
    assert settings['defaults']['importer']['commands']==[['data/other/forms/raw_data/nair_mullu_kurumba_2013/installation/install_source.py','--install']]
    refs=parse_string((HERE/'staged-references.bib').read_text(),'bibtex').entries
    assert set(settings['sources'])==set(refs)=={'nair2013mullukurumba','robertsathyajoseph1982mullukurumba'}
    for key,source in settings['sources'].items():
        assert source['reference']=={'editor':'OpenAI Codex','ocr':False,'etymology_provenance':'source' if key=='nair2013mullukurumba' else 'none'}
    meta=SourceMeta([HERE/(STEM+'.yaml')])
    assert set(meta.sources)==set(refs)
    installed_bib=parse_string(planned['bib'].decode(),'bibtex').entries
    assert all(installed_bib[k]==v for k,v in refs.items())

''' + text[end:]
(HERE/'install_source.py').write_text(text)
print('Standalone Mullu transaction prepared; no canonical writes')
