from research_helpers import Batch
b=Batch(28)
def a(l,g,w,p,e,**kw):
 rs=[r for r in b.inv[l] if r['ID'] not in b.used and r['Gloss']==g and r['Form'] in w.split('|')]
 if rs:b.add(l,g,'|'.join(dict.fromkeys(r['Form'] for r in rs)),p,e,tier='qualified',**kw)
for l,w in [('Malvi','kharab|kharāb'),('Nimadi','kharāb|khərāb'),('Bagheli','khereb|kherab|kherah')]:
 a(l,'bad',w,'f_dbu2w7q6m7sh2','Platts ḵẖarāb explicitly includes bad, spoiled and worthless. The installed Hindi xaraab donor (Liljegren LX002408) supplies the immediate Hindustani stage; local x→kh adaptation, vowels and Bagheli final h need review, without skipping to the Arabic ancestor.',kind='borrowed',citation='platts1884[s.v. ḵẖarāb];liljegren[entry LX002408]',source_url='https://www.rekhta.org/urdudictionary?keyword=kharaab')
for l in ['Malvi','Bagheli']:
 a(l,'bad','bekar','f_nc5i5pyxldlxg','Platts lists be-kār under be- as useless/worthless, supporting the survey bad sense. The existing Hindi बेकार bekaar head comes from a provisional editorial inventory, so this proposal relies on the independent dictionary entry for meaning; whole-word Hindustani borrowing and regional mediation remain for review.',kind='borrowed',citation='platts1884[s.v. be-, be-kār];nihali-provisional2026',source_url='https://www.rekhta.org/urdudictionary?keyword=be')
for l,w in [('Malvi','bura|buro|burā'),('Nimadi','buro')]:
 a(l,'bad',w,'9289','CDIAL 9289.1 *bura gives Sindhi buro, Hindi burā, Gujarati būrũ and Marathi burā bad/wicked. The variable defective/bad family is distinct from section 2 *bōra; local vowel quantity and gender endings are retained without claiming a secure remote origin.',locator='9289.1')
a('Bagheli','twenty','kori','3503','CDIAL 3503 *kōḍi means a score/twenty, with Hindi koṛī, Nepali kori and eastern cognates. Turner discusses Austroasiatic origin and subsequent diffusion; the proposal preserves uncertain transmission and the survey’s nonretroflex r rather than treating this as the viṃśati family.')
for l,w in [('Malvi','sagḷo|hagaḷa|sagaḷa|hagaḷe|sagḷā'),('Nimadi','sagəḷo')]:
 a(l,'all',w,'13066','CDIAL 13066 sakala supplies the all/whole family; Molesworth’s complete sagaḷā entry explicitly connects Marathi sagaḷā all/entire with Sanskrit sakala, supplying the retained g and ḷ comparison absent from Turner’s short article. Malvi s/h and local vowels remain qualified, as does learned or regional reinforcement.',citation='CDIAL[13066];Molesworth MD[s.v. sagaḷā]',source_url='https://www.wisdomlib.org/definition/sagala')
b.save()
