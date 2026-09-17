from research_helpers import Batch
b=Batch(23)
def a(l,g,w,p,e,**kw):
 rs=[r for r in b.inv[l] if r['ID'] not in b.used and r['Gloss']==g and r['Form'] in w.split('|')]
 if rs:b.add(l,g,'|'.join(dict.fromkeys(r['Form'] for r in rs)),p,e,tier='qualified',**kw)
for l,w in [('Malvi','cakku|sakku|cāku'),('Nimadi','cakku|cākku'),('Bagheli','cekku|cakku|caku|cakū|cāku|cākū')]:
 a(l,'knife',w,'f_xqxbqhxafwpsu','Platts ćāqū identifies the Persian-origin clasp-knife/penknife word; the existing Hindi caku control (Kannauji survey p.62) supplies the proposed immediate Hindustani donor. Gemination, Bagheli e and Malvi s require local review; this anchor does not establish a particular donor village or exclude regional mediation.',kind='borrowed',citation='platts1884[s.v. ćāqū];kannauji[p. 62]',source_url='https://www.rekhta.org/urdudictionary?keyword=chaaquu')
for l,w in [('Malvi','daphor|daphoriyā|dappor|daphori|daphore|dappār|dapərā|dapper|dupher|dāphorā|daporiyā|duphar'),('Nimadi','duphār|duphar|duphāri|dupār|duperi'),('Bagheli','dopahar|dopəheṛi|dopəher|ḍupeher|ḍupeherə|ḍupeheri|ḍupeheṛiya')]:
 a(l,'noon',w,'f_jg7fkl55nyhjm','Platts do-pahar explicitly means noon and includes do-paharī/do-pahrī and du-pahriyā. This supports the compound family, but Platts cites dvi-prahara whereas the installed Arora node is labelled *dva-prahara; the first-member reconstruction, local contraction/aspiration and Bagheli retroflex initials remain for review, as does inheritance versus regional diffusion.',citation='platts1884[s.v. do, do-pahar compounds];arora',source_url='https://www.rekhta.org/urdudictionary?keyword=ro')
for l,w in [('Malvi','aŋguṭi'),('Bagheli','eŋguṭi|aŋguṭi')]:
 a(l,'ring',w,'138-2x','CDIAL 138.2 *aṅguṣṭhiya specifically gives Bihari ãguṭhī, Awadhi/Hindi ãgūṭhī and Gujarati ãguṭhī for a ring. The dedicated -iya branch is selected rather than aṅguṣṭha ‘thumb’; loss of aspiration and Bagheli initial e remain qualified, and regional mediation is possible.',locator='138.2')
b.save()
