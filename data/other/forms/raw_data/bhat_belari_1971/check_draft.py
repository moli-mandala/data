"""Validate one source through the row parser; never execute a data build."""
import argparse,csv,io,json,random,sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[4]))
import make_cldf,source_meta
from segments import Tokenizer
from import_source import build,STEM

def check(output,seed):
    directory=output/'other/forms';directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{STEM}.csv';rows,audit=build()
    with path.open('w',newline='') as f:csv.writer(f).writerows(rows)
    meta=source_meta.SourceMeta([ROOT/f'{STEM}.yaml']);errors=io.StringIO()
    with patch.object(source_meta,'load',return_value=meta),patch.dict(make_cldf.convertors,{'bhat-belari':Tokenizer(str(ROOT/'bhat-belari.txt'))}):
        parsed,stats=make_cldf.parse_file(str(path),errors,name=STEM)
    assert not errors.getvalue(),errors.getvalue()
    raw={r[10]:r for r in rows}
    assert len(parsed)==len(raw)==192
    assert {r.entry_key for r in parsed}==set(raw)
    for row in parsed:
        assert row.old_form==raw[row.entry_key][2]
        assert row.gloss==raw[row.entry_key][3]
        assert row.notes==raw[row.entry_key][6]
        assert row.etymology==raw[row.entry_key][9]
        assert row.variant_of_key==raw[row.entry_key][11]
        assert row.tags==raw[row.entry_key][14]
    selected=random.Random(seed).sample(sorted(parsed,key=lambda r:r.entry_key),20)
    report={'seed':seed,'stats':stats,'status':'parser assertions passed; source-output sample awaits acceptance review',
            'sample':[{'key':r.entry_key,'form':r.form,'original':r.old_form,'gloss':r.gloss,'tags':r.tags,'parent':r.variant_of_key,'source':r.source} for r in selected]}
    (output/f'output-sample-{seed}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    print(json.dumps(check(a.output,a.seed),ensure_ascii=False,indent=2))
