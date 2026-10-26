"""Prepare complete physical specimen evidence; never writes canonical data."""
from pathlib import Path
import csv, hashlib, json, re, unicodedata
from collections import Counter

P = Path(__file__).resolve().parent
WORKSPACE = P.parents[6]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name, rows):
    (P/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
page_specimen={464:40,465:41,466:41,467:41,468:42,470:43,471:43}
lects={40:'grierson1906_kui_gumsur_bod',41:'grierson1906_kui_khondmals',42:'grierson1906_kui_khondmals',43:'grierson1906_kui_kalahandi'}
notes={40:'Printed credit: Major J. McD. Smith, 1876. Grierson supplies interlinear glosses; the chapter explicitly qualifies the certainty of some translation. Provisional ñ preserves the nasal-letter reading; Smith specimen upper marks vary from straight to slanted/curved, unlike the clearer tilde in later specimens. No phonological normalization is implied.',
41:'Printed heading: District Khondmals; Specimen I.',42:'Printed heading: District Khondmals; Specimen II.',
43:'Printed heading: State Kalahandi. The approved historical source route is Kui/Kalahandi, with conflicting modern language identification (Glottolog Kuvi coding) retained; no geographic coordinates inferred.'}
uncertainties={
 (470,'10',8):{'type':'source-glyph-identity','reading':'ēohē','alternatives':['ēchē'],'reason':'The middle round letter appears closed in the alternate scan, unlike clear c in nearby puchēñju; o/c identity remains provisional.', 'crop_fraction':[0,0.60,1,0.74]},
 (471,'11',3):{'type':'damaged-medial-cluster','reading':'sibki-mānēñju','alternatives':['sipki-mānēñju','sikki-mānēñju'],'reason':'After si the tall ascender and right bowl support b over p or k; the following glyph is severely broken and compatible with k. Parent independently supports provisional sibki, without resolving damaged type.', 'crop_fraction':[0.10,0.43,0.40,0.56],'retained_crop':'tmp/kui-specimens/p471-giving-detail.png'},
 (471,'16',7):{'type':'source-upper-mark-shape','reading':'mũtēñju','alternatives':['mŭtēñju'],'reason':'The upper u mark is uneven and wavy in this occurrence; tilde is provisional rather than a phonological claim or a harmonization to earlier plain mutēñju.', 'crop_fraction':[0,0.67,1,1]},
}
atoms=[]; lines=[]; inputs={}
for page,spec in page_specimen.items():
    path=P/f'p{page}-first-reading.tsv';inputs[path.name]=sha(path)
    for row in csv.DictReader(path.open(),delimiter='\t'):
        forms=row['forms'].split('|');glosses=row['glosses'].split('|')
        assert len(forms)==len(glosses),(page,row['line'],forms,glosses)
        line_key=f'grierson1906kui:p{page}:specimen{spec}:line{row["line"]}'
        keys=[]
        for index,(form,gloss) in enumerate(zip(forms,glosses),1):
            key=f'{line_key}:word{index}';keys.append(key)
            u=[]
            if spec==40 and 'ñ' in form.lower():
                u.append({'type':'nasal-letter-upper-mark','reason':'Upper mark is straight/slanted/curved in Smith specimen; ñ provisional, pending independent source-specific adjudication. Volume general notation does not uniquely settle this mark.'})
            if (page,row['line'],index) in uncertainties:
                u.append(uncertainties[(page,row['line'],index)])
            source_note=notes[spec]
            if spec==43 and form.strip('.,;‘’“”').lower()=='āmu':
                source_note += ' Preserve this occurrence’s printed pronoun gloss: the specimen glosses āmu as both we and I in different contexts; no uniform person/number interpretation has been imposed.'
            atoms.append(dict(source_unit_key=key,specimen=spec,printed_page=page,physical_line=row['line'],word=index,
              raw_source_form=unicodedata.normalize('NFC',form),raw_gloss=gloss,parent_line=line_key,
              language_id='Kui',source_dialect=lects[spec],source_note=source_note,
              source_image=f'tmp/kui-specimens/alternate-n{page+21}.jpg',source_image_sha256=None,
              typography_uncertainty=u,review_status='first_reading_pending_independent_review',
              unit_type='title_atom' if row['line']=='title' else 'interlinear_atom'))
        lines.append(dict(source_unit_key=line_key,specimen=spec,printed_page=page,physical_line=row['line'],
                          raw_source_form=' '.join(forms),raw_interlinear_gloss=' '.join(glosses),atom_keys=keys))

# Physical line-break continuations are preserved separately, not silently lost.
joins=[]
for first,second in [('grierson1906kui:p464:specimen40:line5:word7','grierson1906kui:p464:specimen40:line6:word1'),
                     ('grierson1906kui:p464:specimen40:line7:word8','grierson1906kui:p464:specimen40:line8:word1')]:
    lookup={x['source_unit_key']:x for x in atoms};a,b=lookup[first],lookup[second]
    assert a['raw_source_form'].endswith('-')
    joined=a['raw_source_form']+b['raw_source_form']
    joins.append(dict(source_unit_key=first+':continuation',atom_keys=[first,second],raw_form_parts=[a['raw_source_form'],b['raw_source_form']],
       combined_form=joined,combined_gloss=a['raw_gloss']+b['raw_gloss'],reason='Explicit printed word continued at physical line break.',review_status='source_boundary_proposal'))

# Every non-title atom participates in one contiguous source-punctuated span.
# This is literal textual evidence, not an inferred syntactic analysis or free translation.
spans=[]
for spec in [40,41,42,43]:
    buffer=[]
    seq=[x for x in atoms if x['specimen']==spec and x['unit_type']=='interlinear_atom']
    for atom in seq:
        buffer.append(atom)
        if re.search(r'[.?][’”\'\"]*$',atom['raw_source_form']):
            spans.append(dict(source_unit_key=f'grierson1906kui:specimen{spec}:expression{sum(x["specimen"]==spec for x in spans)+1}',specimen=spec,
                atom_keys=[a['source_unit_key'] for a in buffer],printed_pages=sorted({a['printed_page'] for a in buffer}),
                raw_source_form=' '.join(a['raw_source_form'] for a in buffer),raw_interlinear_gloss=' '.join(a['raw_gloss'] for a in buffer),
                boundary_basis='Contiguous printed interlinear sequence ending in source full stop/question mark; syntax and readable translation not inferred.',review_status='expression_boundary_proposal'))
            buffer=[]
    assert not buffer,(spec,[a['source_unit_key'] for a in buffer])

manifest=json.loads((P/'alternate-source-manifest.json').read_text())
image_hashes={name:sha(WORKSPACE/name) for name in {a['source_image'] for a in atoms}}
for atom in atoms:
    atom['source_image_sha256']=image_hashes[atom['source_image']]
assert len({a['source_unit_key'] for a in atoms})==len(atoms)
assert [k for s in spans for k in s['atom_keys']]==[a['source_unit_key'] for a in atoms if a['unit_type']=='interlinear_atom']
write('interlinear-first-reading.jsonl',atoms);write('physical-lines-first-reading.jsonl',lines);write('expression-spans-first-reading.jsonl',spans)
(P/'explicit-continuations-first-reading.json').write_text(json.dumps(joins,ensure_ascii=False,indent=2)+'\n')
status=dict(status='complete physical first-reading inventory; independent review and metadata/export remain pending',
            input_sha256=inputs,atom_count=len(atoms),interlinear_atom_count=sum(a['unit_type']=='interlinear_atom' for a in atoms),
            title_atom_count=sum(a['unit_type']=='title_atom' for a in atoms),physical_line_count=len(lines),expression_span_count=len(spans),
            per_page=dict(Counter(a['printed_page'] for a in atoms)),per_specimen=dict(Counter(a['specimen'] for a in atoms)),
            uncertain_atoms=sum(bool(a['typography_uncertainty']) for a in atoms),explicit_continuations=len(joins),
            free_translations={'40':'free-translation-40-p465.txt','42':'free-translation-42-p469.txt'},
            excluded_context={'465':'Surrounding linguistic commentary owned by grammar/prose worker.','469':'After specimen42 free translation, Chinna Kimedi/Kalahandi commentary owned by grammar/prose worker.','470':'Introductory commentary above specimen43 owned by grammar/prose worker.'},
            guards={'all_form_gloss_counts_equal':True,'unique_physical_keys':True,'every_non_title_atom_in_one_expression_span':True,'canonical_files_written':False},
            pending=['Independent complete literal/gloss review','Whole-expression boundary/gloss review','Source-supported grammatical metadata','Integrated profile/parser/graph/reference/identity checks','Canonical installation and deferred full build/browser gates'])
(P/'first-reading-status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in status.items() if k not in ['input_sha256','pending']},ensure_ascii=False,indent=2))
