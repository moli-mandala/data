import hashlib,json,shutil,sys
from pathlib import Path
P=Path(__file__).resolve().parent.parent
O=Path(__file__).resolve().parent
D=next(p for p in P.parents if (p/'source_meta.py').exists())
sys.path.insert(0,str(D))
from source_meta import SourceMeta,yaml_files,importer_commands
fixture=O/'20260927-grierson-kui.yaml'
shutil.copyfile(P/'integration/install-settings-proposed.yaml',fixture)
meta=SourceMeta([*yaml_files(),fixture])
keys=['grierson1906kui','letchmajee1902kui','smith1876khond','frye1851kondh','friendpereira1902khonds']
issues=[]
for key in keys:
    ref=meta.source(key).get('reference',{})
    if ref.get('editor')!='OpenAI Codex':issues.append([key,'editor',ref])
    if ref.get('ocr') is not True:issues.append([key,'reported original OCR provenance',ref])
    expected='source' if key==keys[0] else 'none'
    if ref.get('etymology_provenance')!=expected:issues.append([key,'etymology provenance',ref])
commands=importer_commands(meta,fixture.stem)
if not commands:issues.append(['importer','no guarded importer command'])
for lang in ('Kui','Tamil','Koya'):
    rule=meta.transcription_rule('grierson1906kui',fixture.with_suffix('.csv'),lang)
    if rule.get('profile')!='grierson-kui-1906' or not rule.get('preserve_hyphens'):issues.append([lang,'source profile',rule])
    unrelated=meta.transcription_rule('unrelated-source-key','unrelated-file.csv',lang)
    if unrelated:issues.append([lang,'unexpected unrelated profile leak',unrelated])
report={'status':'passed' if not issues else 'missing configuration','issues':issues,
 'settings_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),
 'validated_source_blocks':{key:meta.source(key) for key in keys},
 'importer_commands':commands,'global_duplicate_declarations':False,
 'scope':'SourceMeta parse with existing global declarations plus exact candidate file-name fixture; source-local profile and unrelated-file isolation checked. No installation or compiled references generated.'}
(O/'install-metadata-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(report['status'],len(issues),'issues')
