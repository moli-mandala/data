from pathlib import Path
import hashlib,json

P=Path(__file__).resolve().parent.parent
ROOT=P.parents[5]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
old=P/'grammar-review/sharp-reviewed-revision/reviewed-child-witnesses.jsonl'
new=P/'grammar-review/sharp-reviewed-revision2/reviewed-child-witnesses.jsonl'
assert sha(old)=='3b0621fd4ed10d80758ec9f448bd4e49a466be25fd89e45ed412a568223d1b13'
assert sha(new)=='c17d355aa870971bbb2433a7f9115fa5ce6deeac3ddc34911451d42534393814'
aa={r['witness_key']:r for r in rows(old)};bb={r['witness_key']:r for r in rows(new)}
assert aa.keys()==bb.keys() and len(bb)==224
fields=['form','gloss','printed_label','language_label','source_unit_key','source_record','physical_parent_key','proposed_disposition']
assert all(aa[k][f]==bb[k][f] for k in aa for f in fields)
assert sum('resolved_reading_cautions' in r for r in bb.values())==14
assert sum(bool(r['typed_uncertainties']) for r in bb.values())==7
for r in bb.values():
    assert r['current_native_witness']['printed_page']==r['printed_page']
    assert sha(ROOT/r['current_native_witness']['path'])==r['current_native_witness']['sha256']
    assert '3.m. accepted' in r['owner_original_review']['source_labels']
tagfile=new.parent/'grammar-tag-proposal.jsonl';graphfile=new.parent/'grammar-graph-proposal.jsonl'
tt=rows(tagfile);gg=rows(graphfile)
assert sum('uncertain' in r['proposed_tags'] for r in tt)==16
assert sum(bool(r['variant_of_witness_key']) for r in gg)==15
assert sum(bool(r['ordered_parents']) for r in gg)==2
qualkey='grierson1906kolamibasim:p567:conjunctive-participle:1:independent:w5'
conjkey='grierson1906kolamibasim:p567:conjunctive-participle:1:independent:w4'
copkey='grierson1906kolamibasim:p566:present-copula-comparison:1:independent:w1'
graphby={r['witness_key']:r for r in gg}
findings=[
 {'key':qualkey,'field':'qualification','old':bb[qualkey]['qualification'],
  'new':'The compact upper mark on final u remains ambiguous between ū, ü and plain u; source segments are retained without phonological interpretation.',
  'reason':'Current typed hold is only final-u mark encoding, a closed editorial source-preservation decision. Earlier middle-quantity/narrow-review wording is historical, not an outstanding new review task.'},
 {'key':conjkey,'field':'source_claim','old':graphby[conjkey]['source_claim'],
  'new':'The source lists sāhādūn and saddan together as having-gone examples in its conjunctive-participle paragraph.',
  'reason':'Literal endpoints were corrected against the sharp original, but source_claim still names the rejected vāttān/vaddan readings. Keep old claim only in history; retain the already approved local Variant pointer.'},
 {'key':copkey,'field':'scope_caveat','old':graphby[copkey]['scope_caveat'],
  'new':'Pointer targets the actual third-person singular neuter row āṇḍād. Retain the source spelling and quantity difference; do not link to masculine āṇḍān.',
  'reason':'The low-resolution 2.m. misreading is historical; sharp original explicitly reads 3.m. Current graph commentary must not call it a disputed source label.'}
]
report={'reviewer':'/root/asur_finish','status':'revision2_literal_history_separation_passes_display_commentary_repairs_required',
 'pins':{str(p.relative_to(P)):sha(p) for p in [old,new,tagfile,graphfile]},
 'counts':{'children':224,'invariant_fields_per_child':len(fields),'historical_caution_rows':14,'current_literal_hold_rows':7,'current_uncertain_tag_rows':16,'variants':15,'components':2},
 'checks':['Every old/new key and literal/gloss/label/raw record/disposition invariant matches.','All 224 current native image paths/folios and hashes match retained originals.','Old 2.m. policy is historical; current owner policy reflects 3.m.','Six upper-u encoding holds and one numeral-mark hold remain explicit; no forced phonology.'],
 'keyed_display_findings':findings,'canonical_changes':False,
 'limits':'Independent metadata-history continuity review; does not replace earlier full 224 original reading or final combined-output sample.'}
out=P/'grammar-native-review/sharp-revision2-history-review.json'
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'sha256':sha(out),'counts':report['counts']}))
