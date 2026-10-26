"""Render original embedded CIDs independently; writes only a bounded evidence PNG.
The original PDF is opened in memory and never overwritten. No replacement font
or inferred Unicode glyph is used to draw the inspected marks.
"""
from pathlib import Path
import fitz
P=Path(__file__).resolve().parent
D=fitz.open(P.parents[5]/'tmp/pdfs/darai-survey/dhakal-2011.pdf')
page=D.new_page(width=650,height=270)
helv=page.insert_font(fontname='helv')
items=[('F041',1740,'0023'),('F060',1719,'0042'),('F032',1779,'0014'),('F03E',1869,'0020'),('F0C3',1913,'00a1')]
fonts=' '.join(f'/F{i} {ref} 0 R' for i,(_,ref,_) in enumerate(items))
D.xref_set_key(page.xref,'Resources',f'<< /Font << {fonts} /Label {helv} 0 R >> >>')
parts=[]
for i,(label,ref,cid) in enumerate(items):
 x=75+125*i
 parts += [f'BT /Label 10 Tf 1 0 0 1 {x-30} 30 Tm ({label}; original xref {ref}) Tj ET',f'BT /F{i} 80 Tf 1 0 0 1 {x} 120 Tm <{cid}> Tj ET',f'0.8 0.8 0.8 RG {x-35} 120 m {x+35} 120 l S']
xref=D.get_new_xref();D.update_object(xref,'<<>>');D.update_stream(xref,'\n'.join(parts).encode());page.set_contents(xref)
page.get_pixmap(matrix=fitz.Matrix(2,2)).save(str(P/'embedded-mark-isolation-extra.png'))
