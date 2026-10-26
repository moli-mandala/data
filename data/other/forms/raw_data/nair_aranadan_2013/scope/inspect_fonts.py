from pathlib import Path
import json,hashlib
import fitz
H=Path(__file__).resolve().parent
W=next(p for p in H.parents if (p/'data/tags.py').exists())
doc=fitz.open(W/'tmp/pdfs/tribal-languages-kerala-nair2013.pdf')
fonts=[];special=[]
for i in (8,9,10):
    for font in doc[i].get_fonts(full=True):
        x=font[0]
        name,ext,typ,content=doc.extract_font(x)
        fonts.append(dict(pdf_page=i+1,xref=x,basefont=font[3],type=font[2],encoding=font[5],
                          to_unicode=doc.xref_get_key(x,'ToUnicode'),embedded_bytes=len(content),
                          embedded_sha256=hashlib.sha256(content).hexdigest() if content else None))
    for block in doc[i].get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            for s in line['spans']:
                if any(0xe000<=ord(c)<=0xf8ff for c in s['text']):
                    special.append(dict(pdf_page=i+1,raw_text=s['text'],font=s['font'],bbox=s['bbox'],size=s['size']))
report=dict(status='font census only; no glyph mappings accepted',fonts=fonts,private_use_spans=special,
    recommendation='Use embedded native outlines and positioned spans before OCR. Read underlines, nasalization marks and legacy PUA visually; do not replace by phonological assumption. Ordinary source double vowels must remain literal until source-specific display policy.',
    ocr_used=False)
(H/'native-font-census.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'font_page_records':len(fonts),'private_use_spans':len(special)}))
