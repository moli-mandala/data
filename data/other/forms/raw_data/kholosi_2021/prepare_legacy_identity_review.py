"""Keyed concordance decisions; never overwrite a historical etymon assignment."""
import csv
import hashlib
import json
from pathlib import Path

P=Path(__file__).resolve().parent
baseline=P/'legacy-baseline/20220913-kholosi.csv'
old=list(csv.reader(baseline.open()))
search=json.loads((P/'legacy-concordance-candidates.json').read_text())
units={r['source_unit_key']:r for r in map(json.loads,(P/'witness-metadata-reviewed-proposed.jsonl').open())}
# These decisions are intentionally more conservative than spelling/IPA search.
# Different meanings, combined senses, and unexplained vowel differences do not
# justify rewriting the historical source witness.
retain={
 4:'Historical adjectival dark versus published nominal darkness; keep the analyses distinct.',
 21:'Two historical ancestry assignments cover a combined very/much/many analysis; do not allocate either to a newly separated POS sense.',
 22:'Two historical ancestry assignments cover a combined very/much/many analysis; do not allocate either to a newly separated POS sense.',
 26:'Historical cāllo [t͡ʃɔllo] differs from published čāllā [t͡ʃɔllɔ] in final vowel; common etymon alone does not license identity replacement.',
 32:'Historical to scare versus published to be afraid; causative/experiencer meaning difference is not silently corrected.',
 52:'No equivalent whole head located in the pinned dictionary or structured lexicon.',
 70:'Historical len to take differs from tentative repository len? to put?; preserve both source states.',
 98:'Historical combined new/moon meaning cannot be assigned arbitrarily to one of two published homonyms.',
 113:'No equivalent whole head located; inflected sentence material does not license reconstructing the legacy infinitive.',
 119:'No equivalent whole head located in the pinned dictionary or structured lexicon.',
 127:'Historical bare pūnī is explicitly an analysis of compound puni jelji; keep that historical analysis and the complete printed compound distinct.',
 148:'Historical sāndo spelling differs from printed sānda; preserve both witnesses despite matching supplied IPA.',
 156:'Historical sovoyī [soʋoji] differs from published sāvāyi [sɔʋɔji]; no analogical harmonization.',
 162:'Historical son versus published boy; do not erase the earlier kinship sense.',
 163:'Historical daughter versus published girl; do not erase the earlier kinship sense.',
}
for n in (138,139,157,158,187,188):
 retain[n]='This literal witness has multiple historical ancestry assignments. Preserve each historical analysis and emit a separate neutral source witness so source variants/paradigms do not arbitrarily choose one ancestry.'
chosen={35:'kholosi:pdf:p2:c2:u11',55:'kholosi:pdf:p4:c1:u7',56:'kholosi:pdf:p4:c1:u13',
 59:'kholosi:github:18b43f982efa:lexicon.csv:row110',64:'kholosi:pdf:p4:c2:u1',
 108:'kholosi:pdf:p6:c2:u4',117:'kholosi:pdf:p6:c2:u27',133:'kholosi:pdf:p7:c2:u3',
 147:'kholosi:pdf:p8:c1:u7',154:'kholosi:pdf:p8:c1:u19',179:'kholosi:pdf:p9:c2:u16',180:'kholosi:pdf:p9:c2:u17'}
expected_gloss={35:'sun',55:'who, which',56:'spider',59:'ear',64:'to',108:'potato',
 117:'old',133:'father',147:'rice',154:'thin',179:'mouth',180:'path'}
out=[]
for n,(raw,found) in enumerate(zip(old,search),1):
 assert raw[10]==found['legacy_entry_key']
 record=dict(entry_key=raw[10],historical_fields=raw,etymon_assignment=raw[1],
             assignment_disposition='Preserve the existing Jambu editorial assignment exactly; source claims are separately attributed.',
             source_claim_ledger='source-etymology-claims.jsonl')
 if n in retain:
  record.update(disposition='retain historical witness separately',reason=retain[n],equivalent_new_source_units=[])
 else:
  key=chosen.get(n)
  if key is None:
   assert len(found['candidates'])==1,(n,found)
   key=found['candidates'][0]['source_unit_key']
  unit=units[key]
  assert len(unit['candidates'])==1,(n,key)
  candidate=unit['candidates'][0]
  if n in expected_gloss:
   assert expected_gloss[n] in candidate['gloss'],(n,key,candidate['gloss'])
  equivalent=[key]
  if n in (117,133):
   equivalent += [x['source_unit_key'] for x in found['candidates'] if x['source_unit_key']!=key]
  record.update(disposition='reuse legacy key for reviewed equivalent lexical witness',
    reason='Reviewed spelling/IPA, lexical meaning and homonym scope together; search normalization is not an output transcription rule.',
    equivalent_new_source_units=equivalent,proposed_source_entry_key=candidate['entry_key'],
    before=dict(form=raw[2],gloss=raw[3],phonemic=raw[5]),
    after=dict(form=candidate['form'],gloss=candidate['gloss'],phonemic=candidate.get('phonetic','')))
 out.append(record)
assert len(out)==198 and len({r['entry_key'] for r in out})==198
(P/'legacy-identity-review.json').write_text(json.dumps(dict(
 status='source-local identity proposal; independent integrated-output review required',
 baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
 proposal_input_sha256=hashlib.sha256((P/'witness-metadata-reviewed-proposed.jsonl').read_bytes()).hexdigest(),
 records=out),ensure_ascii=False,indent=2)+'\n')
print('legacy keys',len(out),'historical retained',len(retain),'equivalent key reuses',len(out)-len(retain))
