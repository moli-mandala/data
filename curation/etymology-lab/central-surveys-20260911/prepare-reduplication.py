from research_helpers import Batch
b=Batch(30)
for l,word,parent,base,num,note in [('Nimadi','ālagalag','f_xmyt2e5ltrhie','alag',176,'The alag anchor is from Sonipura-Balai, while these repeated forms occur at other survey sites; it represents the proposed survey-level base, not evidence that one village borrowed from another.'),('Bagheli','elegeleg','f_swnisrqkeocbe','eleg',177,'The merged eleg and elegeleg records overlap in source sites P and b, although each carries a different first locality tag; source locators, not that first tag alone, establish the overlap.')]:
 e=f'Proposed full reduplication of {base} ‘different/separate’, with CDIAL 700 alagna supporting the base family. Repetition is an analysis of the survey form, not a separately reconstructed ancient compound. {note} Acceptance depends on {l} proposal {num}; local vowel length and the precise distributive nuance remain qualified.'
 b.add(l,'different',word,parent,e,tier='qualified',kind='derived',citation='CDIAL[700]')
 x=b.proposals[l][-1];x['acceptanceDependencies']=[{'type':'pending-proposal','survey':l,'proposal':num,'parentId':parent}]
 for row,record in zip(x['assignments'],x['records']):row['Source']=record['Source']+';CDIAL[700]'
b.save()
