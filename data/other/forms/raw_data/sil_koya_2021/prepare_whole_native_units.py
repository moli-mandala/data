"""Extract review-only narrative units from native PDF layout; no installation."""
import collections, hashlib, json, re
from pathlib import Path
import fitz

PACKAGE=Path(__file__).resolve().parent
WORKSPACE=PACKAGE.parents[5]
PDF=WORKSPACE/'tmp/pdfs/koya/JLSR2021_029.pdf'
EXPECTED='a6541e0d2397849ce7c36961b3849f3b2c1f1c267036cfa1a3f6025796e14e7d'
SECTIONS=[('B.1','Utnoor Gondi',22,24),('B.2','Bhamani Gondi',25,26),
          ('B.3','Bhamani Madia',27,28),('B.4','Malakanagiri Koya',29,30),
          ('B.5','Podia Koya',31,32),('B.6','Chintoor Koya',33,34),
          ('B.7','Jaganathapuram Koya',35,36)]

def extract():
    assert hashlib.sha256(PDF.read_bytes()).hexdigest()==EXPECTED
    units=[];controls=[];counts=collections.Counter()
    with fitz.open(PDF) as document:
        for section,lect,first,last in SECTIONS:
            current=None
            def finish():
                nonlocal current
                if current is None:return
                current['source_form']=' '.join(x['text'] for x in current['target_lines'])
                current['translation_raw']=' '.join(x['text'] for x in current['translation_lines']).removeprefix('• ').strip()
                assert current['target_lines'] and current['translation_lines'],current
                firstline=current['target_lines'][0]
                question=re.match(r'^Q\.?\s*(\d+)\.?\s*',current['source_form'])
                current['question_number']=int(question[1]) if question else None
                current['kind']='test-question' if question else 'narrative-unit'
                current['form_without_question_label']=current['source_form'][question.end():] if question else current['source_form']
                current['status']='native-layout extraction; original-page and translation-scope review pending'
                units.append(current);current=None
            for pdfpage in range(first,last+1):
                page=document[pdfpage-1]
                lines=[]
                for block in page.get_text('dict')['blocks']:
                    for line in block.get('lines',[]):
                        text=''.join(s['text'] for s in line['spans']).strip()
                        if text:lines.append({'pdf_page':pdfpage,'printed_page':pdfpage-5,'bbox':[round(v,3) for v in line['bbox']],'text':text})
                lines.sort(key=lambda x:(round(x['bbox'][1],1),x['bbox'][0]))
                for line in lines:
                    text=line['text'];x,y=line['bbox'][:2]
                    if text.isdigit() or text.startswith('Appendix B:') or re.fullmatch(r'B\.\d',text) or text.startswith('Text Test:'):
                        controls.append(dict(section=section,**line,disposition='page-number-or-section-heading'));continue
                    if x<90:
                        if current and current['translation_lines']:finish()
                        if current is None:
                            counts[pdfpage]+=1
                            current={'source_unit_key':f'devagnanavaram2021:whole:{section}:p{pdfpage-5}:{counts[pdfpage]}',
                                     'section':section,'lect':lect,'target_lines':[],'translation_lines':[]}
                        current['target_lines'].append(line)
                    else:
                        assert current is not None,(section,line)
                        current['translation_lines'].append(line)
            finish()
    return units,controls

if __name__=='__main__':
    units,controls=extract()
    destination=PACKAGE/'whole-source-recovery';destination.mkdir(exist_ok=True)
    for name,records in [('narrative-native-first-reading.jsonl',units),('narrative-layout-controls.jsonl',controls)]:
        (destination/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    summary={'status':'review-only native extraction; no canonical changes','source_sha256':EXPECTED,
             'narrative_and_question_units':len(units),'layout_controls':len(controls),
             'section_counts':dict(collections.Counter(u['section'] for u in units)),
             'question_counts':dict(collections.Counter(u['section'] for u in units if u['kind']=='test-question')),
             'required_review':['Original page review of all target units','Translation/free-answer boundaries','Source orthography profile distinct from wordlist IPA','Whole-report comparative/prose/control census','Independent output audit']}
    (destination/'narrative-native-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))
