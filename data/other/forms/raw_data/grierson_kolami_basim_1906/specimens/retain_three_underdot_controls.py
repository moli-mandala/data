"""Supplement the frozen correction ledger with native occurrence evidence."""
import hashlib
import json
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[6]
SOURCE = ROOT / 'tmp/jambu-next-source/kolami-specimen-originals/alternate-leaf585-printed564.jpg'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert sha(SOURCE) == '46268f190aac6db45ddc1e4195d033e75edf04b87351ef5aa4a309819046f8a4'
    first = HERE/'p564-first-reading.jsonl'
    atoms = {a['source_unit_key']: a for s in first.read_text().splitlines() for a in json.loads(s)['target_atoms']}
    im = Image.open(SOURCE)
    rows = []
    for line, word, old, new, gloss, rectangle in [
        (6,4,'aṇchan','aḍchan','difficulty',(1770,2800,2180,3070)),
        (11,3,'anḍā,','aṇḍā,','is,',(1070,4120,1410,4390)),
        (15,3,'anḍān','aṇḍān','is',(1020,5150,1390,5430)),
    ]:
        key=f'grierson1906kolamibasim:p564:specimen75-line:{line}:word:{word}'
        assert atoms[key]['raw_source_form']==old
        assert atoms[key]['raw_interlinear_gloss']==gloss
        p=SOURCE.parent/f'owner-native-p564-underdot-l{line}w{word}.png'
        if not p.exists(): im.crop(rectangle).save(p)
        rows.append(dict(source_unit_key=key,field='raw_source_form',old=old,new=new,raw_interlinear_gloss=gloss,
                         source=str(SOURCE.relative_to(ROOT)),source_sha256=sha(SOURCE),rectangle=list(rectangle),
                         crop=str(p.relative_to(ROOT)),crop_sha256=sha(p),
                         disposition='Occurrence crop supplement; frozen 24-field overlay unchanged; root confirmation pending'))
    out=HERE/'three-underdot-native-evidence.json'
    out.write_text(json.dumps(dict(first_reading_sha256=sha(first),controls=rows,canonical_changes=False),ensure_ascii=False,indent=2)+'\n')
    print(out)

if __name__=='__main__': main()
