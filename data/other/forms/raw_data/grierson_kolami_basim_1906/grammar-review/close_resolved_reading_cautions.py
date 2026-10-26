"""Separate historical reading cautions from current source uncertainty."""
import copy,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE/'sharp-reviewed-revision'
OUT=HERE/'sharp-reviewed-revision2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=OLD/'reviewed-child-witnesses.jsonl'
    assert sha(source)=='3b0621fd4ed10d80758ec9f448bd4e49a466be25fd89e45ed412a568223d1b13'
    raw=[json.loads(s) for s in source.read_text().splitlines()]
    proposals=HERE.parent/'grammar-native-review/full224-sharp-review-proposal.jsonl'
    assert sha(proposals)=='47801d590eba805cff65682759806a159ad25f97dab04e79100b4ed8ec5820a1'
    native={r['source_unit_key']:r['original_page'] for r in (json.loads(s) for s in proposals.read_text().splitlines())}
    result=[];decisions=[]
    for before in raw:
        r=copy.deepcopy(before)
        historical=[u for u in r['typed_uncertainties'] if u['type']=='transcription']
        live=[u for u in r['typed_uncertainties'] if u['type']!='transcription']
        assert all(u['type'] in ['source_grapheme_quantity','source-mark-shape'] for u in live)
        if historical:
            r['resolved_reading_cautions']={'historical_uncertainties':historical,'historical_qualification':r['qualification'],
              'resolution':'Sharper native whole-child comparison resolves the prior body/segment/quantity question; current uncertainty, if any, is separately scoped.',
              'review_evidence':'grammar-native-review/full224-sharp-review-proposal.jsonl and sharp-reviewed-revision/owner-corroboration.json'}
            r['qualification']=''
            decisions.append({'source_unit_key':r['source_unit_key'],'old_current_uncertainties':before['typed_uncertainties'],
                              'removed_to_history':historical})
        for u in live:
            if u['type']=='source_grapheme_quantity':
                u['detail']='Source u body and surrounding segments reviewed; compact upper mark remains ambiguous between the listed literal alternatives. No phonological interpretation.'
        r['typed_uncertainties']=live
        r['historical_owner_source_labels']=r['owner_original_review']['source_labels']
        r['owner_original_review']['source_labels']='Sharper native p5663.m. accepted; historical2.m. reading retained only in source_record/history.'
        r['current_native_witness']=native[r['source_unit_key']]
        r['historical_image_provenance']={'original_image':r['original_image'],'original_image_sha256':r['original_image_sha256'],
             'status':'Earlier degraded witness retained as history; current literal comparison uses separately pinned current_native_witness.'}
        result.append(r)
    assert len(result)==224
    assert sum(bool(r['typed_uncertainties']) for r in result)==7
    assert all(a['form']==b['form'] and a['gloss']==b['gloss'] and a['printed_label']==b['printed_label'] and a['source_record']==b['source_record'] for a,b in zip(raw,result))
    OUT.mkdir(exist_ok=True)
    target=OUT/'reviewed-child-witnesses.jsonl'
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result))
    report={'input_path':str(source),'input_sha256':sha(source),'output_sha256':sha(target),
      'counts':{'children':224,'current_uncertain_children':7,'six_upper_u_cases':6,'numeral_mark_cases':1,'historical_caution_rows':len(decisions)},
      'all_forms_glosses_labels_keys_raw_history_unchanged':True,'decisions':decisions,
      'canonical_changes':False,'accepted_for_installation':False}
    (OUT/'resolved-caution-continuity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'counts':report['counts'],'output_sha256':sha(target)}))
if __name__=='__main__':main()
