"""Conservation/hashes for the four immutable specimen first-reading pages."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    records=[];pins={};counts={}
    for page,expected in [(564,17),(565,18),(568,17),(569,12)]:
        path=HERE/f'p{page}-first-reading.jsonl'
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        assert len(rows)==expected
        assert [r['target_atoms'][0]['physical_line'] for r in rows]==list(range(1,expected+1))
        for r in rows:assert [a['physical_word'] for a in r['target_atoms']]==list(range(1,len(r['target_atoms'])+1))
        records.extend(rows);pins[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        counts[page]=dict(lines=expected,physical_atoms=sum(len(r['target_atoms']) for r in rows))
    atoms=[a for r in records for a in r['target_atoms']]
    assert len(records)==64 and len({a['source_unit_key'] for a in atoms})==len(atoms)
    assert all(a['raw_source_form'] and a['raw_interlinear_gloss'] for a in atoms)
    report=dict(status='All64lines complete manual first reading; not final literal acceptance',physical_lines=64,
        physical_atoms=len(atoms),counts_by_page=counts,inputs_sha256=pins,
        explicit_word_continuations=[dict(first_key='grierson1906kolamibasim:p568:specimen76-line:11:word:8',
            second_key='grierson1906kolamibasim:p568:specimen76-line:12:word:1',raw_first='Bāp-',raw_second='phōkān',
            disposition='Both physical records retained; joined atomic/export proposal still requires reviewed derivative.')],
        scope='Specimen75 Wun/Kolami pp564–565 and specimen76 Basim/Naikri pp568–569',
        pending=['Complete owner second read with keyed fine-mark/punctuation corrections','Independent full literal review',
            'Source-bounded expressions and two-page punctuation conservation','Atomic continuation derivative','Grammar/metadata/identity integration and complete outside closure',
            'Fresh output audit and actual parser gates'],canonical_changes=False)
    (HERE/'first-reading-handoff.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
