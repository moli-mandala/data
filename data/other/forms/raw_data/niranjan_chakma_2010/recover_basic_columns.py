"""Raw column OCR for both final basic-wordlist pages; no installed output."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import pypdfium2 as pdfium
from extract import read_lines
from recover_remaining_ocr import PDF_SHA

HERE=Path(__file__).resolve().parent
COLUMNS={'left_english':(.04,.215,'eng'),'left_chakma':(.22,.36,'ben'),
         'left_bengali':(.375,.505,'ben'),'right_english':(.53,.69,'eng'),
         'right_chakma':(.695,.835,'ben'),'right_bengali':(.845,.99,'ben')}
PAGE_COLUMNS={108:COLUMNS,109:{
    'left_english':(.035,.165,'eng'),'left_chakma':(.17,.325,'ben'),
    'left_bengali':(.35,.49,'ben'),'right_english':(.51,.665,'eng'),
    'right_chakma':(.645,.795,'ben'),'right_bengali':(.82,.99,'ben')}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--tessdata',type=Path,required=True)
    args=parser.parse_args()
    assert hashlib.sha256(args.pdf.read_bytes()).hexdigest()==PDF_SHA
    args.cache.mkdir(parents=True,exist_ok=True)
    document=pdfium.PdfDocument(args.pdf)
    output=[]
    for number in [108,109]:
        page=document[number-1];bitmap=page.render(scale=4)
        image=bitmap.to_pil().copy();bitmap.close();page.close()
        for name,(left,right,language) in PAGE_COLUMNS[number].items():
            x0,x1=int(image.width*left),int(image.width*right)
            path=args.cache/f'p{number}-{name}.tsv'
            if not path.exists():
                crop=args.cache/'current-column.png'
                image.crop((x0,0,x1,image.height)).save(crop)
                temporary=path.with_suffix('.tmp')
                with temporary.open('w') as stream:
                    subprocess.run(['tesseract',str(crop),'stdout','--tessdata-dir',str(args.tessdata),
                                    '-l',language,'--psm','6','-c','tessedit_create_tsv=1'],
                                   stdout=stream,check=True,env=dict(os.environ,OMP_THREAD_LIMIT='1'))
                temporary.replace(path);crop.unlink()
            output.append(dict(pdf_page=number,column=name,crop_x=[x0,x1],
                full_image_size=list(image.size),raw_tsv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                raw_lines=read_lines(path.read_text(),x0),status='unreviewed column OCR'))
            print(f'{number} {name} recovered',flush=True)
    recipe=dict(pdf_sha256=PDF_SHA,pages=[108,109],scale=4,psm=6,workers=1,columns=PAGE_COLUMNS,
                models={lang:hashlib.sha256((args.tessdata/f'{lang}.traineddata').read_bytes()).hexdigest() for lang in ['ben','eng']})
    (HERE/'basic-column-raw-evidence.json').write_text(json.dumps(dict(recipe=recipe,columns=output),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
