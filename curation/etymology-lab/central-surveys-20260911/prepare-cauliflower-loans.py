from research_helpers import Batch
b=Batch(38)
for lang,words in [('Malvi','phulgobi|phulgobhi'),('Nimadi','phulgobi|ɸulkobi'),('Bagheli','phulgobi')]:
 b.add(lang,'cauliflower',words,'f_wl2o55yco7mxk','Borrowing proposed from the existing whole Hindi phulgobi ‘cauliflower’ (Kannauji survey p. 73). The Hindi–English table in Central Bank of India’s Cent Saral Bhasha, PDF p. 11, independently matches फूलगोबी with cauliflower; aspiration/frication and '+('Nimadi k for g' if lang=='Nimadi' else 'vowel length')+' remain qualified. The Hindi compound donor is unlinked and requires ancestry review before acceptance; reversed gobiphul and bare gobi responses are excluded.',tier='qualified',kind='borrowed',citation='kannauji[p. 73]',source_url='https://www.centralbankofindia.co.in/sites/default/files/%E0%A4%95%E0%A4%A8%E0%A5%8D%E0%A4%A8%E0%A4%A1.pdf#page=11')
 b.proposals[lang][-1]['acceptanceDependency']={'type':'unlinked-donor','parentId':'f_wl2o55yco7mxk'}
b.save()
