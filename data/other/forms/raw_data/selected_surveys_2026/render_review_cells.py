"""Re-render a saved Orissa targeted-review index without changing review decisions.

Usage: python render_review_cells.py orissa-nasal-empty-review.json --output /tmp/orissa-review
Accepts any of the saved annotation/residual/symbol indices (item + column).
The cached original PDF is required; table geometry comes from the pinned extraction.
"""
import argparse,json
from pathlib import Path
from PIL import Image,ImageDraw
from pypdf import PdfReader
import orissa
R=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('index');p.add_argument('--output',type=Path,default=Path('/tmp/orissa-review'));a=p.parse_args()
 records={(r['item'],r['column']):r for r in orissa.records()[0]}
 wanted=[records[r['item'],r['column']] for r in json.loads((R/a.index).read_text())]
 pdf=PdfReader(R/'orissa-official.pdf');images={}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 for start in range(0,len(wanted),20):
  canvas=Image.new('RGB',(1500,1800),'#eeeeee');draw=ImageDraw.Draw(canvas)
  for n,r in enumerate(wanted[start:start+20]):
   pg=r['pdf_page']
   if pg not in images:images[pg]=pdf.pages[pg-1].images[0].image.convert('RGB').rotate(r['rotation_degrees'],resample=Image.Resampling.BICUBIC,fillcolor='white')
   crop=images[pg].crop(r['bbox']);x=n%2*750;y=n//2*180
   if crop.height>150:crop=crop.resize((round(crop.width*150/crop.height),150))
   draw.text((x+5,y+2),f"{r['item']}:{r['column']} {r['eng']}",fill='black');canvas.paste(crop,(x+5,y+23))
  canvas.save(str(a.output)+f'-{start//20}.png')
if __name__=='__main__':main()
