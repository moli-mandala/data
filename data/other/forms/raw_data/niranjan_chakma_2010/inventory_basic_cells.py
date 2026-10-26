"""Align all 99 printed basic-wordlist cells, retaining unaccepted OCR fields."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
# English anchors read directly from original PDF108–109. These are physical
# rows, not a presumed 100-item standardized list.
ANCHORS={
 (108,'left'): 'I You We This That Who What Not All Many One Two Big Long Small Woman Man Person Fish Bird Dog Louse Tree'.split(),
 (108,'right'): 'Seed Leaf Root Bark Skin Flesh Blood Bone Grease Egg Horn Tail Feather Hair Head Ear Eye Nose Mouth Tooth Tongue Claw Knee Hand'.split(),
 (109,'left'): 'Belly Neck Breasts Heart Liver Drink Eat Bite See Hear Know Sleep Die Kill Swim Fly Walk Come Lie Sit Stand Give Say Sun Moon Star'.split(),
 (109,'right'): 'Water Rain Stone Sand Earth Cloud Smoke Fire Ash Burn Path Mountain Red Green Yellow White Black Night Hot Cold Full New Good Round Dry Name'.split(),
}


def centre(line):
    return (line['bbox'][1]+line['bbox'][3])/2


def main():
    source=HERE/'basic-column-raw-evidence.json'
    evidence=json.loads(source.read_text())
    assert set(evidence['recipe']['columns'])=={'108','109'}, 'Requires corrected page-specific crop geometry'
    columns={(r['pdf_page'],r['column']):r for r in evidence['columns']}
    output=[]
    for (page,side),labels in ANCHORS.items():
        minimum=290 if (page,side)==(108,'left') else 220 if page==108 else 100
        english=[r for r in columns[page,side+'_english']['raw_lines'] if minimum<centre(r)<2120]
        english.sort(key=centre)
        assert len(english)==len(labels),(page,side,len(english),len(labels))
        for index,(label,anchor) in enumerate(zip(labels,english),1):
            midpoint=centre(anchor)
            distances=[abs(midpoint-centre(other)) for other in english if other is not anchor]
            radius=min(distances)*.45
            fields={}
            for language in ['chakma','bengali']:
                selected=[r for r in columns[page,side+'_'+language]['raw_lines'] if abs(centre(r)-midpoint)<radius]
                fields[language]=dict(raw_lines=selected,raw=' '.join(r['raw'] for r in selected),
                    status='OCR reading unaccepted; original-glyph review required')
            if page==108 and label in ['Bird','Ear']:
                fields['bengali']['original_blank_review']='Original image shows blank Bengali gloss cell; do not fill from lexical expectation.'
            notes=[]
            if label=='Smoke':
                notes.append('Printed English Smoke conflicts with Chakma খানা and Bengali পানকরা; retain all evidence, semantic interpretation unresolved.')
            if label=='Lie':
                notes.append('Bengali control is মিথ্যা; do not assume English Lie means recline.')
            if label=='Bark':
                notes.append('Printed Bengali control appears ছাগল; review source error versus glyph reading rather than silently correcting to tree bark.')
            output.append(dict(source_unit_key=f'niranjan2010chakma:p{page-2}:basic:{side}:row{index}',
                pdf_page=page,printed_page=page-2,block=side,row=index,
                english_original=label,english_ocr=anchor,source_fields=fields,
                structural_status='Owner original-page row census; independent review pending',
                lexical_status='Unaccepted preparation; no installed row',source_caveats=notes))
    assert len(output)==len({r['source_unit_key'] for r in output})==99
    path=HERE/'basic-cell-inventory-proposal.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output))
    status=dict(stage='Complete basic-section physical inventory, not lexical acceptance or whole-book completion',
        physical_cells=99,by_block={f'{p}-{s}':len(v) for (p,s),v in ANCHORS.items()},
        missing_chakma_ocr=[r['source_unit_key'] for r in output if not r['source_fields']['chakma']['raw']],
        printed_blank_bengali=['p106 Bird','p106 Ear'],
        raw_evidence_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        inventory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),canonical_rows_changed=0,
        next='Original-glyph transcription of all99 target cells and controls, independent audit and integration with entire book')
    (HERE/'basic-cell-inventory-status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status))


if __name__=='__main__':main()
