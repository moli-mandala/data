"""Extract full shortlisted CDIAL articles, including addenda, without reparsing data."""
import json,pickle,re,sys
from pathlib import Path
from bs4 import BeautifulSoup
P=Path(__file__).resolve().parent; ROOT=P.parents[2]
ids=set(sys.argv[2:]);out={i:[] for i in ids}
for page,html in enumerate(pickle.load((ROOT/'data/cdial/cdial.pickle').open('rb')),1):
    for chunk in html.split('<number>')[1:]:
        number=chunk.split('</number>',1)[0].strip()
        if number not in ids:continue
        raw='<number>'+chunk.split('</hw>',1)[0]
        text=BeautifulSoup(raw,'html.parser').get_text(' ',strip=True)
        out[number].append({'page':page,'url':f'https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page={page}','html':raw,'text':text})
(P/(sys.argv[1]+'.json')).write_text(json.dumps(out,ensure_ascii=False,indent=1)+'\n')
for i in sys.argv[2:]:
    print('\nENTRY',i)
    for x in out[i]:print('PAGE',x['page'],x['text'])
