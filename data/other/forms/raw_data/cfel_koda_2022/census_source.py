"""Count Koda entry bullets directly from outlines, without rasterization.

This is physical scope evidence, not a lexical transcription. Each edition has
its own page/item locators; do not transfer a page count or IPA between editions.
"""
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def census_xps(path):
    pages=[]
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith('.fpage'):
                continue
            page=int(re.search(r'(\d+)\.fpage$',name)[1])
            if page<10:
                continue
            root=ET.fromstring(archive.read(name)); bullets=[]
            for element in root:
                transform=element.attrib.get('RenderTransform','').split(',')
                # Printed bullets are the only paths in this left-margin band.
                # Include both inline curves and StaticResource PathGeometry.
                if element.tag.endswith('}Path') and len(transform)==6 and 125<float(transform[4])<140:
                    bullets.append(float(transform[5]))
            pages.append({'xps_page':page,'printed_page':page-3,'entries':len(bullets),'bullet_y':bullets})
    pages.sort(key=lambda r:r['xps_page'])
    assert len(pages)==357 and sum(r['entries'] for r in pages)==2450
    assert [r['entries'] for r in pages[:10]]==[5]+[6]*9
    return {'edition':'Koda-Bangla-Hindi-English, 2022','source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'lexical_entries':2450,'page_census':pages,'status':'Physical scope verified; lexical readings and full independent audit remain pending.'}


def census_pdf(path):
    import pdfplumber
    pages=[]
    with pdfplumber.open(path) as pdf:
        assert len(pdf.pages)==371
        for index in range(9,len(pdf.pages)):
            page=pdf.pages[index]
            bullets=[curve for curve in page.curves if 97<curve['x0']<100 and 3.4<curve['width']<4.2 and 3.4<curve['height']<4.2]
            pages.append({'pdf_page':index+1,'printed_page':index-2,'entries':len(bullets),'bullet_tops':[round(curve['top'],3) for curve in bullets]})
            page.close()
    return {'edition':'English-Hindi-Bangla-Koda, 2022','source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'lexical_entries':sum(r['entries'] for r in pages),'page_census':pages,'status':'Physical bullet inventory; lexical readings and independent full audit remain pending.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    report=census_xps(args.source) if args.source.suffix.lower()=='.xps' else census_pdf(args.source)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
