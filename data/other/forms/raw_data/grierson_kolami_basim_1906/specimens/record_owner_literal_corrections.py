"""Keyed owner adjudications; frozen first readings are never edited."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROPOSALS=[
    (6,4,'aṇchan','aḍchan','Root strong original correction; owner agrees detached underdot belongs to d, not n.'),
    (11,3,'anḍā,','aṇḍā,','Root strong original correction; owner agrees n underdot preserved separately from following ḍ.'),
    (15,3,'anḍān','aṇḍān','Root strong original correction; owner agrees n underdot preserved separately from following ḍ.'),
    (7,4,'actnā','ectnā','Owner isolated original crop shows initial e crossbar/open lower bowl; no analogical substitution.')]


def main():
    path=HERE/'p564-first-reading.jsonl'
    atoms={a['source_unit_key']:a for r in map(json.loads,path.read_text().splitlines()) for a in r['target_atoms']}
    crop=json.loads((HERE/'detail-crop-locators.json').read_text())[0]
    corrections=[]
    for line,word,old,new,evidence in PROPOSALS:
        key=f'grierson1906kolamibasim:p564:specimen75-line:{line}:word:{word}'
        assert atoms[key]['raw_source_form']==old
        corrections.append(dict(source_unit_key=key,field='raw_source_form',old=old,new=new,
            review='owner original-adjudicated proposal; root full-page acceptance remains pending',evidence=evidence,
            original_evidence=crop if line==7 else dict(source_image='tmp/jambu-next-source/kolami-specimen-originals/printed-564-2800.png',physical_line=line,physical_word=word)))
    report=dict(status='Four keyed owner corrections; no silent first-reading replacement',input_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest()},
        corrections=corrections,material_form_corrections=4,complete_page_review=False,
        remaining='Root fine-mark/punctuation review and complete owner second reading; no final output candidate yet.',canonical_changes=False)
    (HERE/'owner-literal-corrections-p564.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
