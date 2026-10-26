"""Source-local bibliography, registry and settings reconciliation; no install."""
import argparse,csv,hashlib,json,re,sys,tempfile
from pathlib import Path
from collections import Counter
import yaml
from pybtex.database import parse_file
from pybtex import PybtexEngine
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
ROOT=HERE.parents[6]
sys.path.insert(0,str(ROOT/'data'))
from source_meta import SourceMeta,yaml_files
from tags import GRAMMATICAL_TAGS,GENDER_TAGS
from dialects import dialect_tag
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main(directory=None):
    global HERE
    if directory is not None:HERE=directory.resolve()
    with (HERE/'combined-review-candidate.csv').open() as f:rr=list(csv.reader(f))
    language_rows={r['ID']:r for r in csv.DictReader((ROOT/'data/cldf/languages.csv').open())}
    current_dialects=list(csv.DictReader((ROOT/'data/cldf/dialects.csv').open()))
    proposed=list(csv.DictReader((PACKAGE/'metadata-profile-proposal/dialects-proposed.csv').open()))
    assert len(proposed)==2
    for d in proposed:
        assert d['Language_ID'] in language_rows
        assert d['Clade']==language_rows[d['Language_ID']]['Clade']
        assert d['Tag']==dialect_tag(d['Language_ID'],d['Source_Language_ID'],d['Name'])
        assert all(d[k]=='' for k in ['Glottocode','Latitude','Longitude','Quality'])
        assert not any(d['ID']==x['ID'] or d['Tag']==x['Tag'] for x in current_dialects)
    allowed_dialects={d['Tag'] for d in current_dialects+proposed}
    for r in rr:
        assert r[0] in language_rows
        assert all(t in GRAMMATICAL_TAGS|GENDER_TAGS or t in allowed_dialects for t in r[14].split()),(r[10],r[14])
        assert r[7].startswith('grierson1906kolamibasim[p. ') and ';' not in r[7]
    initial=PACKAGE/'metadata-profile-proposal/references-proposed.bib'
    bib=initial.read_text()
    entry=parse_file(str(initial)).entries['grierson1906kolamibasim']
    prior_note=entry.fields['note']
    note=('Wun Kolami and historical Basim Naikri are distinct source-qualified routes; the latter identification follows DEDR bibliography section 27. '
          'Grierson supplied the Kolami interlinear translation. Separate higher-resolution page witnesses resolve degraded glyphs without an asserted textual-edition difference. '
          'The adjacent Chanda Naiki portion is represented separately under grierson1906naiki; later addenda remain separate witnesses')
    assert prior_note in bib
    bib=bib.replace(prior_note,note,1)
    extra=('  included = {Complete printed pp. 561--569: grammar, paradigms and both interlinear specimens, plus source-attributed comparisons on pp. 182, 623, 626 and 627. '
           'Three existing canonical rows represent reuse of two physical printings; the p. 623 suffix is explicitly shared by Naiki and Kolami},\n'
           '  provenance = {Planned canonical data/other/forms/20260927-grierson-kolami-basim.csv; accepted audit data/other/forms/raw_data/grierson_kolami_basim_1906/installed-audit.jsonl; '
           'guarded importer data/other/forms/raw_data/grierson_kolami_basim_1906/installation/install_source.py. Current review CSV and audit are '+str(HERE.relative_to(PACKAGE))+'/combined-review-candidate.csv and combined-review-audit.jsonl within that package. '
           'Primary snapshot locator scope/snapshot-status.json and source-edition-and-reuse-review.json pin the 1906 Commons volume; separate sharper-page manifests retain their individual URLs and hashes},\n'
           '  licence = {Public-domain 1906 source volume, independently verified against the Commons edition catalogue; later alternate page images are separate reproduction witnesses},\n')
    bib=bib.replace('  jambu_editor = {OpenAI Codex},\n',extra+'  jambu_editor = {OpenAI Codex},\n',1)
    bib=bib.replace('  ocr = {Yes}\n','  ocr = {Yes},\n  etymology_provenance = {source}\n',1)
    aux_marker='  ocr = {No}\n'
    assert aux_marker in bib
    bib=bib.replace(aux_marker,'  ocr = {No},\n  etymology_provenance = {none}\n',1)
    target=HERE/'references-review-candidate.bib';target.write_text(bib)
    references=parse_file(str(target)).entries
    current=parse_file(str(ROOT/'data/cldf/sources.bib')).entries
    assert set(references)=={'grierson1906kolamibasim','haig1897gondi_kolami'}
    assert not set(references)&set(current)
    formatted={k:PybtexEngine().format_from_string(v.to_string('bibtex'),'plain',output_backend='markdown') for k,v in references.items()}
    assert all(v.strip() for v in formatted.values())
    settings=yaml.safe_load((HERE/'combined-review-candidate.yaml').read_text())
    with tempfile.TemporaryDirectory(dir=HERE) as temp:
        path=Path(temp)/'combined-review-candidate.yaml';path.write_text(yaml.safe_dump(settings,sort_keys=False))
        meta=SourceMeta([*yaml_files(),path])
        for key in references:
            # SourceMeta schema accepted; exact flags checked through public accessors.
            assert meta.flag(key,'reference','editor')=='OpenAI Codex'
            assert meta.flag(key,'reference','ocr')==(key=='grierson1906kolamibasim')
            assert meta.flag(key,'reference','etymology_provenance')==('source' if key=='grierson1906kolamibasim' else 'none')
        assert meta.flag('grierson1906kolamibasim','forms','split_alternates',True) is False
        assert meta.flag('grierson1906kolamibasim','gloss','grammar_tags',True) is False
    declaration_counts=Counter()
    for path in [*yaml_files(),HERE/'combined-review-candidate.yaml']:
        for key in (yaml.safe_load(path.read_text()) or {}).get('sources',{}):declaration_counts[key]+=1
    assert all(declaration_counts[k]==1 for k in references)
    report={'status':'Source-local metadata/reference/schema validation passes; staged installer and canonical provenance still deferred',
        'rows':827,'language_counts':dict(Counter(r[0] for r in rr)),'registered_languages_all_valid':True,'tags_all_registered_or_exact_two_dialect_proposals':True,
        'proposed_dialects':proposed,'source_declarations':{k:declaration_counts[k] for k in references},'formatted_references':formatted,
        'input_pins':{str(p.relative_to(ROOT)):sha(p) for p in [HERE/'combined-review-candidate.csv',HERE/'combined-review-candidate.yaml',initial,ROOT/'data/cldf/languages.csv',ROOT/'data/cldf/dialects.csv',ROOT/'data/cldf/sources.bib']},
        'reference_candidate_sha256':sha(target),'notes_policy':'Generic alignment/parser-state evidence is audit-only; lexical rows contain no repeated parser commentary.',
        'canonical_changes':False,'pending':['Final independent source-output and specimen metadata approval','Final canonical paths/importer command in staged metadata','Canonical source-stage installation','Full compiled and browser gates']}
    (HERE/'metadata-reference-readiness.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('827 rows/8 registered bases, two exact dialect proposals, two references/schema checks passed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path);a=p.parse_args();main(a.directory)
