"""Keyed sharper-witness decisions; immutable first readings stay unchanged."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    first = HERE / 'p564-first-reading.jsonl'
    assert sha(first) == '9231c09556f8e658f823363bfbf408bc57651ec053677f8a02e75e19ce1974b7'
    atoms = [a for line in first.read_text().splitlines() for a in json.loads(line)['target_atoms']]
    by = {a['source_unit_key']: a for a in atoms}
    crops = {r['name']: r for r in json.loads((HERE/'sharp-p564-glyph-control-locators.json').read_text())}
    corrections = []
    for line, word, old, gloss, accepted, crop, history in [
        (2, 5, 'vāṭū', 'share', 'vāṭā', 'share', None),
        (4, 5, 'sedeṇ.', 'went.', 'seden.', 'went', 'Initial ledger wrongly keyed word6 Attīn; rejected before applying any derivative.'),
        (7, 4, 'actnā', 'going', 'setnā', 'going', 'Earlier low-resolution ectnā proposal is superseded by sharp initial s+e.'),
        (7, 1, 'muḷuk', 'country(-of)', 'muluk', 'country', None),
        (8, 8, 'āṅse', 'husks', 'sōse', 'husks', None),
        (9, 7, 'vāṭlīn.', 'it-appeared.', 'vāṭīltīn.', 'appeared', 'Intermediate vāṭiltīn. proposal superseded after both flat i macrons were independently confirmed.'),
        (13, 2, 'In-din-tānāt', 'This-day-from', 'In-dīn-tānāt', 'day', None),
        (16, 1, 'laīltnā', 'running', 'tūltnā', 'running', None),
    ]:
        key=f'grierson1906kolamibasim:p564:specimen75-line:{line}:word:{word}'
        assert by[key]['raw_source_form'] == old, (key, by[key])
        assert by[key]['raw_interlinear_gloss'] == gloss, (key, by[key])
        corrections.append(dict(source_unit_key=key, field='raw_source_form', old=by[key]['raw_source_form'],
                                new=accepted, decision='Root and owner independently accepted sharper occurrence reading',
                                evidence=crops[crop], superseded_proposal=history,
                                source_witness_caveat='Sharp alternate witness resolves source shape; no demonstrated editorial edition change.'))
    rejected = []
    for line, word, hypothesis, crop in [(11,5,'kārūt','hunger'),(12,1,'sērātūn','will-go'),(12,3,'yēnātūn,','will-say'),
                                         (8,1,'kāyēṅg','feed'),(10,1,'siyetēn.','gave-not')]:
        key=f'grierson1906kolamibasim:p564:specimen75-line:{line}:word:{word}'
        rejected.append(dict(source_unit_key=key, retained=by[key]['raw_source_form'], rejected=hypothesis,
                             evidence=crops[crop], reason='Native crop does not print the hypothesized extra macron.'))
    report=dict(status='p564 all153 atoms visually compared; source-bar versus diaeresis running control retains provisional macron',
                first_reading_sha256=sha(first), reviewed_physical_atoms=153,
                corrections=corrections, rejected_preliminary_hypotheses=rejected,
                source_manifest_sha256=sha(HERE.parent/'specimen-conservation-review/sharper-selected-originals-manifest.json'),
                glyph_crop_manifest_sha256=sha(HERE/'sharp-p564-glyph-control-locators.json'),
                top_mark_control=dict(source_unit_key='grierson1906kolamibasim:p564:specimen75-line:16:word:1',
                                      provisional='tūltnā',rejected_hypothesis='tültnā',
                                      reason='Root pixel-preserving nearest inspection shows contiguous upper bar; apparent second body is u stem, not separate same-level dot. No phonemic claim.',
                                      evidence=crops['running']),
                pending=['Reviewed derivative and fresh final combined candidate audit'], canonical_changes=False)
    (HERE/'sharp-p564-review-decisions.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

if __name__ == '__main__': main()
