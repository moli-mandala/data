"""Actual canonical parser and metadata checks; no database generation."""
import csv,hashlib,io,json,os,sys,unicodedata
from pathlib import Path
import yaml
P=Path(__file__).resolve().parent;D=P.parents[4];W=P/'whole-source-recovery';F=W/'uncertainty-addendum'
CSV=D/'data/other/forms/20260925-bailey-kagani.csv';PROFILE=D/'conversion/bailey-kagani-1920.txt'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    from install_whole_source import verify_approval,bib_entry
    verify_approval()
    for src,dest in [(F/'proposal.csv',CSV),(F/'proposal-audit.jsonl',P/'audit.jsonl'),(F/'profile-proposal.txt',PROFILE)]:assert sha(src)==sha(dest)
    rows=list(csv.reader(CSV.open()));assert len(rows)==1306
    os.chdir(D);sys.path.insert(0,str(D));import make_cldf
    errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(CSV),errors,name=CSV.stem)
    assert not errors.getvalue(),errors.getvalue()
    assert len(parsed)==stats['converted']==1306
    assert [r.entry_key for r in parsed]==[r[10] for r in rows]
    for r,raw in zip(parsed,rows):
        assert (r.old_form,r.notes,r.source,r.tags)==(raw[2],raw[6],raw[7],raw[14]),r.entry_key
        display=unicodedata.normalize('NFC',raw[2]).lower().replace('w','v').replace('ṅ','ŋ')
        assert unicodedata.normalize('NFC',r.form)==display,r.entry_key
    y=yaml.safe_load(CSV.with_suffix('.yaml').read_text())
    assert y['defaults']['transcription']=={'profile':'bailey-kagani-1920','input':'form','preserve_hyphens':True}
    assert y['defaults']['identity']=={'legacy_ids':'stem','append_order':100}
    source=y['sources']['bailey1920kagani'];assert source['identity']['dedupe_by_entry_key'] is True
    assert source['forms']['split_alternates'] is False and source['reference']['ocr'] is True
    assert y['defaults']['importer']['commands']==[['data/other/forms/raw_data/bailey_kagani_1920/install_whole_source.py','--install']]
    keys={r[10] for r in rows};assert all(not r[11] or r[11] in keys for r in rows)
    baseline=json.loads((W/'legacy-baseline-20260927.json').read_text());assert [r[10] for r in rows[:329]]==baseline['entry_keys']
    entry=bib_entry((D/'cldf/sources.bib').read_text())[2];assert '1306 proposed forms' in entry and '1217 audited units' in entry
    report={'status':'passed actual canonical source-stage parser/profile/metadata/keys','rows':1306,'audit_units':1217,'preserved_legacy_keys':329,'parser_errors':0,'no_monkeypatch':True,'canonical_sha256':{str(q.relative_to(D)):sha(q) for q in [CSV,CSV.with_suffix('.yaml'),PROFILE,P/'audit.jsonl']},'deferred':['Full DB/CLDF build','Full suite','Compiled durable-ID verification','Browser QA']}
    (P/'whole-installed-validation-20260927.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
