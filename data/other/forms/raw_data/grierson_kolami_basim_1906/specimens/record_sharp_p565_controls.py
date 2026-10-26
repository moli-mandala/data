"""Record occurrence-specific controls; no accepted derivative is produced."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    first = HERE / 'p565-first-reading.jsonl'
    assert sha(first) == 'be018c7f7826a0f24fdcd7f2de67230a213665f3dc5bc0eaf3b8a30726ce0323'
    atoms = [a for s in first.read_text().splitlines() for a in json.loads(s)['target_atoms']]
    by = {(a['physical_line'], a['physical_word']): a for a in atoms}
    crops = {r['name']: r for r in json.loads((HERE/'sharp-remaining-glyph-control-locators.json').read_text())}
    decisions = []
    specs = [
        (1,6,'māsaruṅg','servants-to','māsarūṅg','servants-said','retain'),
        (1,7,'iṭten,','said,','iṭṭen,','servants-said','retain'),
        (4,3,'bhulīlta-āṇḍeṇ,','lost-was,','bhulīlta-āṇḍen,','lost-was','proposed_correction'),
        (7,1,'ellā-merāt','house-near','ellā-mērāt','house-came-v2','retain'),
        (7,5,'endākād','dancing','ēndākād','big-son','retain'),
        (8,9,'iṭten,','said,','iṭṭen,','said8-v2','retain'),
        (9,6,'khuśālīnāḍ','safety-in','khūśālīnāḍ','father-happy-v2','retain'),
        (15,4,'laī','big','lāī','big15-v2','retain'),
        (16,8,'an','my','ān','my16-v2','retain'),
        (17,4,'kāken','to-make','kākēn','thine-make-v2','retain'),
        (6,4,'vaḍlāk','big','vāḍlāk','son-enclosure','retain'),
        (6,6,'vegāṭ','field-in','vēgāṭ','son-enclosure','retain'),
        (9,5,'bānu','father-to','bānū','father9','retain'),
        (10,2,'laī','much','lāī','big10-v3','retain'),
    ]
    for line, word, old, gloss, hypothesis, name, decision in specs:
        row = by[line,word]
        assert row['raw_source_form'] == old, row
        assert row['raw_interlinear_gloss'] == gloss, row
        decisions.append(dict(source_unit_key=row['source_unit_key'], old=old,
                              raw_interlinear_gloss=gloss, field='raw_source_form',
                              proposed_value=hypothesis,
                              reviewed_value=hypothesis if decision == 'proposed_correction' else old,
                              disposition=decision, evidence=crops[name],
                              reason='Native occurrence crop; not an analogical or phonemic remapping.',
                              accepted_for_installation=False))
    report = dict(status='all161 atoms visually compared with native page; one literal correction proposed, no final acceptance',
                  first_reading_sha256=sha(first), physical_atom_count=len(atoms),
                  crop_manifest_sha256=sha(HERE/'sharp-remaining-glyph-control-locators.json'),
                  decisions=decisions,
                  full_body_crop_manifest_sha256=sha(HERE/'sharper-owner-crop-locators.json'),
                  top_mark_review='Every source i/u was visually compared at native scale; broad apparent extra bars in these controls were rejected occurrence by occurrence. No phonemic normalization.',
                  retained_uncertainty=[dict(source_unit_key='grierson1906kolamibasim:p565:specimen75-line:7:word:6',
                                            type='damaged-print',reason='Printed medial cluster/possible lower mark remains degraded; retain vintēr. without inferred retroflex correction.')],
                  canonical_changes=False)
    (HERE/'sharp-p565-narrow-controls.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

if __name__ == '__main__': main()
