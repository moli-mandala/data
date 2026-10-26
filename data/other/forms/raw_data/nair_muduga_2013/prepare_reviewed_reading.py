"""Apply exact owner lowline corrections and complete quotation accounting."""
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent

def main():
    first=BASE/'first-reading.jsonl'
    assert hashlib.sha256(first.read_bytes()).hexdigest()=='7830d8e6d3d7d88d4a9bf58b2d65eba4bad85191ee293c9653362d4a196f421b'
    units=[json.loads(x) for x in first.read_text().splitlines()]
    corrections=json.loads((BASE/'owner-second-reading/corrections.json').read_text())['corrections']
    bykey={w['witness_key']:w for u in units for w in u['witnesses']}
    for c in corrections:
        w=bykey[c['witness_key']]
        assert w[c['field']]==c['old'],c['witness_key']
        w[c['field']]=c['new']
    extra=json.loads((BASE/'owner-second-reading/documentary-supplement.json').read_text())
    assert len(extra)==6
    assert not {u['source_unit_key'] for u in extra}&{u['source_unit_key'] for u in units}
    units+=extra
    assert len(units)==128 and sum(len(u['witnesses']) for u in units)==229
    (BASE/'reviewed-first-reading.jsonl').write_text(''.join(json.dumps(u,ensure_ascii=False)+'\n' for u in units))
    print(hashlib.sha256((BASE/'reviewed-first-reading.jsonl').read_bytes()).hexdigest())

if __name__=='__main__':main()
