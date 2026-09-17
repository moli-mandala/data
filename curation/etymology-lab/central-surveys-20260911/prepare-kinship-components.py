import json
from research_helpers import Batch,P
b=Batch(18);anchors=json.loads((P/'component-anchors.json').read_text())
def c(l,g,words,first,second,note=''):
 aa=[anchors[l][first],anchors[l][second]]
 evidence=f"Segment as {first} ‘small/big’ + {second} ‘sibling’, with ordinary gender/number stem variants. The survey’s age-relative kinship meaning supports this composition. {note} The anchors are existing same-survey lexical records, not asserted historical donor villages. Acceptance depends on {l} proposals "+', '.join(str(a['proposal']) for a in aa)+'.'
 citation=';'.join(dict.fromkeys(a['citation'] for a in aa))
 b.add(l,g,words,aa[0]['id'],evidence,tier='qualified',kind='component',citation=citation)
 x=b.proposals[l][-1];x['components']=[{**a,'position':i} for i,a in enumerate(aa,1)];x['parentForm']=' + '.join(a['form'] for a in aa)
 x['acceptanceDependencies']=[{'type':'pending-proposal','survey':l,'proposal':a['proposal'],'parentId':a['id']} for a in aa]
 x['assignments']=[{'Form_ID':r['ID'],'Etymon_ID':a['id'],'Kind':'component','Rank':'1','Status':'accepted','Source':r['Source']+';'+citation,'Notes':evidence,'Pos':str(i)} for r in x['records'] for i,a in enumerate(aa,1)]
c('Malvi','younger brother','choṭobhai|choṭābhāi','choṭo','bhai')
c('Malvi','younger brother','nānobhai|nanabhai','nano','bhai','The nano anchor also has child/short/small senses; its broader merged semantics remain qualified.')
c('Malvi','older sister','baḍibēn','baḍo','ben')
c('Malvi','older sister','moṭibēn','moṭo','ben')
c('Malvi','younger sister','choṭiben|choṭibēn|coṭibəhin','choṭo','ben','The bahin versus ben alternation belongs to the independently proposed bhaginī family.')
c('Malvi','younger sister','nānibēn|naniben','nano','ben')
c('Nimadi','older sister','baḍibæṇ|baḍibayeṇ|baḍibahin','baḍo','beiṇ','The second member varies from contracted bæṇ to bahin; this is compatible with the bhaginī family but merits local review.')
c('Nimadi','older sister','moṭibayiṇ|moṭibeiṇ|moṭibāiṇ','moṭo','beiṇ')
c('Nimadi','younger sister','choṭibæṇ|choṭibayiṇ|choṭibayeṇ|choṭibeyin|coṭibəhin','choṭo','beiṇ','The anchor is elicited as older sister, whereas the compound is younger sister; the hypothesis treats age as supplied by the adjective, which needs semantic review.')
c('Nimadi','younger sister','nanibeiṇ|nānibāiṇ','nāno','beiṇ','The same older-versus-younger sense qualification applies to the sister anchor.')
c('Bagheli','older brother','beḍa bhay|beḍebhay','beḍa','bhay')
c('Bagheli','older brother','berka bhay','berka','bhay')
c('Bagheli','younger brother','choṭebhay|coṭka bhay|coṭa bhay','choṭa','bhay','The -ka and -e adjective variants are preserved, not silently equated with the exact anchor spelling.')
c('Bagheli','older sister','beri behen|bəḍibəhin','beḍa','behin','The adjective’s r/ḍ alternation and feminine ending require review.')
c('Bagheli','older sister','berki behin','berka','behin')
c('Bagheli','younger sister','coṭki behin|coṭi behini','choṭa','behin','The sister anchor is elicited as older sister; treating the age distinction as supplied by the modifier remains qualified. The final -i is an ordinary feminine/extended form.')
b.save()
