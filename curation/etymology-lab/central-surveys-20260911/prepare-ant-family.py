from research_helpers import Batch
b=Batch(14)
for l,w in [('Malvi','ciṭi|ciṭyā|ciḍi|cĩṭi'),('Nimadi','ciṭ̃i'),('Bagheli','cihuṭi|ciuṭi|cihiṭi|cimṭi|ciṭi|ciūṭi|cīṭi|cīṭiya|ciṭuəua')]:
 b.add(l,'ant',w,'4822','CDIAL 4822 connects Hindi cimṭā/cyũṭā ‘pincers’ with the *cimb pinch family; Platts s.v. cimṭī explicitly includes ant and cross-refers to cīṅṭī/cyūṅṭī, whose entries give the small-ant formations. This supports a qualified semantic family, not a uniquely established remote reconstruction; nasal loss, Bagheli internal h and extended endings still need local review.',tier='qualified',citation='CDIAL[4822];platts1884[s.v. cimṭī, cīṅṭī, cyūṅṭī]',source_url='https://www.rekhta.org/urdudictionary?keyword=chimtii')
b.add('Malvi','body','tan','5656','CDIAL 5656 gives Pali tanu and Prakrit taṇū ‘body’, with Hindi tan. The bare Malvi form fits this body-word family; direct local inheritance versus wider Hindi circulation cannot be distinguished from this elicitation alone.',tier='qualified')
b.save()
