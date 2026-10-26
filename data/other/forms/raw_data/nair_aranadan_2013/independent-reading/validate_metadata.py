from pathlib import Path
import sys,json,hashlib,shutil
D=Path(__file__).resolve().parent;P=D.parent;ROOT=D.parents[5];sys.path.insert(0,str(ROOT))
import source_meta,pybtex
from pybtex.database import parse_file
settings=P/'root-review/proposed-settings.yaml';bib=P/'root-review/proposed-references.bib'
F=D/'metadata-fixture';F.mkdir(exist_ok=True);fixture=F/'20260927-nair-aranadan.yaml';shutil.copyfile(settings,fixture)
canonical=source_meta.load(); original_paths=source_meta.yaml_files();proposed=source_meta.SourceMeta([fixture])
assert not (set(proposed.sources)&set(canonical.sources)), 'Proposed citation already registered; review collision'
source_meta.yaml_files=lambda:[fixture];source_meta.load.cache_clear();meta=source_meta.load()
for language in ('Aranadan','Malayalam','Kannada'):
 assert meta.transcription('nair2013aranadan','20260927-nair-aranadan.csv',language)==(None,False)
 assert meta.transcription_rule('nair2013aranadan','20260927-nair-aranadan.csv',language)['preserve_hyphens'] is True
for key,provenance in [('nair2013aranadan','source'),('usha1992aranadanmar','none')]:
 assert meta.flag(key,'reference','editor')=='OpenAI Codex'
 assert meta.flag(key,'reference','ocr') is False
 assert meta.flag(key,'reference','etymology_provenance')==provenance
assert meta.file_flag('20260927-nair-aranadan.csv','identity','dedupe_by_entry_key') is True
assert meta.file_flag('20260927-nair-aranadan.csv','forms','split_alternates') is False
try:source_meta.importer_commands(meta,'20260927-nair-aranadan');command='declared; not executed'
except source_meta.SourceMetaError:command='pending actual installer creation; known unfulfilled preinstallation gate'
B=parse_file(bib);existing=parse_file(ROOT/'cldf/sources.bib');assert not (set(B.entries)&set(existing.entries))
formatted={}
for k,e in B.entries.items():
 assert e.fields['ocr']=='No';assert e.fields['jambu_editor']=='OpenAI Codex'
 formatted[k]=pybtex.PybtexEngine().format_from_string(e.to_string('bibtex'),'plain',output_backend='markdown')
 assert formatted[k].strip()
assert B.entries['nair2013aranadan'].fields['pages']=='9--11'
assert B.entries['nair2013aranadan'].fields['year']=='2013'
assert 'Original article not independently consulted' in B.entries['usha1992aranadanmar'].fields['note']
report={'status':'passed isolated real SourceMeta and production-style formatter','hashes':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (settings,bib)},'fixture':str(fixture.relative_to(P)),'checks':['Actual source_meta.load isolated via yaml_files override, declared filename preserved','All3language routes return profileNone convertFalse with preserve_hyphensTrue','ImmutableEntryKey dedupe and no alternate splitting','Both citations actual editor OpenAI Codex, OCR false, source/none etymology enum','No canonical YAML source-key or Bib collisions','Both records formatted through production Pybtex plain/markdown engine without exception'],'formatted':formatted,'original_attribution_review':'Nair author/title/journal13(7)July2013/pp9–11 match original page headers/footers and previously read source. Usha is reported bibliography only; unknown23-8 edition/volume interpretation is not invented. NoOCR is accurate for nativePDF glyph recovery plus visual reading.','literal_preservation_limit':'convert:false is verified metadata routing, not a claim that every downstream generic parser punctuation operation is bypassed; actual final output field tests remain required. No distinct IPA layer claimed.','importer_gate':command,'canonical_writes':False}
(D/'metadata-formatter-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(report['status'])
