from research_helpers import Batch
b=Batch(24)
def a(l,g,w,p,e,**kw):
 rs=[r for r in b.inv[l] if r['ID'] not in b.used and r['Gloss']==g and r['Form'] in w.split('|')]
 if rs:b.add(l,g,'|'.join(dict.fromkeys(r['Form'] for r in rs)),p,e,tier='qualified',**kw)
a('Nimadi','mortar','khāyṇo','3796','CDIAL 3796 khaṇḍana gives Gujarati khā̃yṇī/khā̃yṇiyɔ ‘mortar’ beside khā̃ḍṇũ ‘pounding in a mortar’. The instrument sense and yṇ sequence support the match; absent written nasalization and the masculine -o ending need review, and Gujarati or regional mediation remains possible.',locator='3796, Gujarati instrument nouns')
a('Bagheli','millet','jinhora|joṇṇeri|joneri|junheri|juneri|juneṛi|joṇḍari','10434','CDIAL 10434 yavanāla gives Bihari janer(ā), jonhrī, jõdhrī, jinorā and Hindi junhār, jundrī, jõḍrī for millet/maize. These support the regional nasal and rhotic variants; the survey’s broad millet gloss is preserved because the article covers different grain species.')
for l,w in [('Malvi','bandigobi'),('Nimadi','bandgobi'),('Bagheli','beṇḍagobi|beṇḍgobi|beṇḍhe gobi|baṇḍegobi|baṇḍgobi')]:
 a(l,'cabbage',w,'f_ody2xkuwe2bqg','The CSTT English–Hindi–Dogri agriculture glossary explicitly gives Hindi बंदगोभी ‘cabbage’, matching the existing Hindi donor bandgobʰī. Whole-word borrowing is proposed; epenthetic vowels, retroflexion and aspiration differences remain qualified, and regional mediation cannot be excluded.',kind='borrowed',citation='cstt-agriculture-eng-hin-dogri[s.v. cabbage]',source_url='https://cstt.education.gov.in/sites/default/files/fundamental-glossary-agriculture-eng-hin-dogri.pdf')
a('Bagheli','ring','celli|celle|chella','f_grji7azl5s42k','Platts ćhallā explicitly denotes a plain finger/toe ring and offers competing cakkala/chakra-based derivations. The installed *chhallā node is used only as a provisional family head: no secure remote reconstruction is claimed, and source deaspiration, e-vowels and feminine/oblique endings require review; Hindi mediation is possible.',citation='platts1884[s.v. ćhallā, ring];arora',source_url='https://www.rekhta.org/urdudictionary?keyword=%E0%A4%9B%E0%A4%B2%E0%A5%8D%E0%A4%B2%E0%A4%BE')
b.save()
