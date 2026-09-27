"""Recover malformed Cambria ToUnicode spaces from embedded TrueType cmap."""
import argparse
import io
import json
import struct
from pathlib import Path
from unittest.mock import patch
import pdfplumber
from pdfminer.pdftypes import resolve1
from pdfminer.pdffont import PDFCIDFont
from extract_scaffold import extract

def font_mapping(path):
    with pdfplumber.open(path) as pdf:
        fonts=resolve1(pdf.pages[5].page_obj.resources['Font'])
        for obj in fonts.values():
            f=resolve1(obj)
            if 'Cambria' in str(f) and 'DescendantFonts' in f:
                child=resolve1(resolve1(f['DescendantFonts'])[0])
                assert str(child['CIDToGIDMap'])=="/'Identity'"
                data=resolve1(resolve1(child['FontDescriptor'])['FontFile2']).get_data()
                break
        else:raise ValueError('Expected embedded Cambria CID font not found')
    tables={}
    for i in range(struct.unpack_from('>H',data,4)[0]):
        tag,_,off,length=struct.unpack_from('>4sIII',data,12+16*i);tables[tag]=(off,length)
    off,_=tables[b'cmap'];mapped={}
    def add(gid,code):
        if gid:mapped.setdefault(gid,set()).add(chr(code))
    for i in range(struct.unpack_from('>H',data,off+2)[0]):
        platform,encoding,rel=struct.unpack_from('>HHI',data,off+4+8*i);sub=off+rel
        if platform not in (0,3):continue
        fmt=struct.unpack_from('>H',data,sub)[0]
        if fmt==4:
            count=struct.unpack_from('>H',data,sub+6)[0]//2;end0=sub+14;start0=end0+2*count+2;delta0=start0+2*count;range0=delta0+2*count
            for k in range(count):
                end=struct.unpack_from('>H',data,end0+k*2)[0];start=struct.unpack_from('>H',data,start0+k*2)[0];delta=struct.unpack_from('>h',data,delta0+k*2)[0];ro=struct.unpack_from('>H',data,range0+k*2)[0]
                for code in range(start,end+1):
                    gid=(code+delta)&65535 if not ro else struct.unpack_from('>H',data,range0+k*2+ro+(code-start)*2)[0]
                    if ro and gid:gid=(gid+delta)&65535
                    add(gid,code)
        elif fmt==12:
            for k in range(struct.unpack_from('>I',data,sub+12)[0]):
                start,end,gid=struct.unpack_from('>III',data,sub+16+k*12)
                for code in range(start,end+1):add(gid+code-start,code)
    return mapped

def recover(path):
    mapped=font_mapping(path);original=PDFCIDFont.to_unichr;repairs={};unknown=set()
    def decode(font,cid):
        text=original(font,cid)
        if 'Cambria' in str(font.fontname) and text==' ' and cid!=3:
            choices=mapped.get(cid,set())
            if len(choices)==1:
                value=next(iter(choices));repairs[f'{cid:04X}']=value;return value
            unknown.add(cid);return '\ue000'
        return text
    with patch.object(PDFCIDFont,'to_unichr',decode):
        rows=extract(path)
    for row in rows:
        row['status']='font-recovered-pending-visual-review'
        row['warning']='Embedded-font cmap repairs malformed ToUnicode spaces; U+E000 marks unmapped glyph. All readings still require visual review.'
    return rows,{'mapping_basis':'Embedded TrueType Unicode cmap, CIDToGIDMap Identity; repair only false U+0020 mappings, excluding actual space CID0003','repairs':repairs,'unmapped_cids':[f'{v:04X}' for v in sorted(unknown)],'unmapped_items':[r['item'] for r in rows if '\ue000' in r['raw_text']]}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pdf',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows,report=recover(a.pdf);a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'font-recovered-scaffold.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (a.output/'font-recovery.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
