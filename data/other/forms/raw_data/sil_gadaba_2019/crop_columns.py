import pdfplumber
from PIL import Image
PDF='tmp/pdfs/sil-surveys/silesr2019_005.pdf'; OUT='%s/gadaba'%'/private/tmp/claude-501/-Users-aryamanarora-Documents-Code-jambu-all/b54f657d-6dcd-4484-bc40-af4daeb09487/scratchpad'
COLS=[(0.100,0.400),(0.365,0.668),(0.632,0.960)]
with pdfplumber.open(PDF) as pdf:
    for n in range(18,36):
        im=pdf.pages[n].to_image(resolution=336).original
        W,H=im.size
        for i,(a,b) in enumerate(COLS):
            c=im.crop((int(W*a),int(H*0.05),int(W*b),int(H*0.95)))
            g=c.convert('L'); bb=g.point(lambda v:255 if v<190 else 0).getbbox()
            if bb: c=c.crop((0,max(0,bb[1]-15),c.size[0],min(c.size[1],bb[3]+15)))
            c.save(f"{OUT}/p{n:02d}c{i+1}.png")
        print(n,end=' ',flush=True)
print()
