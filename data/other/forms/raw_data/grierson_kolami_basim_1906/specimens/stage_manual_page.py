"""Serialize explicit manually read pairs; never OCR, split alternates or infer forms."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'data/source_meta.py').exists())


def stage(page,specimen,lines,scope,notes=None):
    image=ROOT/f'tmp/jambu-next-source/kolami-specimen-originals/printed-{page}-2800.png'
    source_hash=hashlib.sha256(image.read_bytes()).hexdigest()
    census={r['source_unit_key']:r for r in map(json.loads,(HERE.parent/'scope/physical-unit-census.jsonl').read_text().splitlines())}
    rows=[]
    for line,text in enumerate(lines.strip().splitlines(),1):
        parent=f'grierson1906kolamibasim:p{page}:specimen{specimen}-line:{line}'
        assert parent in census
        atoms=[]
        for word,pair in enumerate(text.split('|'),1):
            form,gloss=pair.split('=',1)
            atoms.append(dict(source_unit_key=f'{parent}:word:{word}',printed_page=page,physical_page_onebased=page+20,
                physical_line=line,physical_word=word,raw_source_form=form,raw_interlinear_gloss=gloss,
                source_scope=f'Specimen{specimen} / {scope}',review='manual first reading; independent review pending'))
        rows.append(dict(source_unit_key=parent,printed_page=page,physical_page_onebased=page+20,specimen=specimen,
            source_scope=scope,source_image=str(image.relative_to(ROOT)),source_image_sha256=source_hash,
            target_atoms=atoms,raw_target_line=' '.join(a['raw_source_form'] for a in atoms),
            raw_aligned_english_line=' '.join(a['raw_interlinear_gloss'] for a in atoms),
            source_punctuation_note=(notes or {}).get(line,''),
            typed_uncertainties=[dict(type='transcription',reason='Fine marks need independent original-controlled review; no analogy-based harmonization.')],
            review='complete page manual original-first reading; not accepted or exported'))
    expected={564:17,565:18,568:17,569:12}[page]
    assert len(rows)==expected
    output=HERE/f'p{page}-first-reading.jsonl'
    text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    if output.exists():assert output.read_text()==text,'Frozen first reading must not be overwritten'
    else:output.write_text(text)
    report=dict(page=page,physical_lines=len(rows),physical_atoms=sum(len(r['target_atoms']) for r in rows),
        output_sha256=hashlib.sha256(text.encode()).hexdigest(),status='First-reading evidence, no canonical proposal',
        final_expression_spans='Pending source-boundary review; physical lines are not automatically sentence boundaries.',canonical_changes=False)
    (HERE/f'p{page}-first-reading-status.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    return rows
