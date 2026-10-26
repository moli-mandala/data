"""Research-only decisions for the ten unlinked Arsenault–Abraham Muduga forms."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
records=json.loads((P/'survey-inventory.json').read_text())
proposal={
 'f_7lhvgflruimrm':('d2484','DEDR 2484 explicitly lists Tamil cāṟu “juice, pepper water”, Malayalam cāṟu “broth, soup”, Kannada sāṟu “sauce, pepper water”, and Merolu cāru “a kind of curry”. Muduga cāṟɯ has the same liquid-food sense; the article family applies despite its displayed reconstruction “to go reach”.'),
 'f_nwvd4eiozsefk':('d3111','DEDR 3111 lists Tamil/Malayalam taviṭu and Kannada/Telugu tavuḍu “bran”, matching Muduga tavuḍɯ in consonants and sense. The paper gives this form as a vowel-conditioning example without proposing a different origin.'),
}
reasons={
 'f_lelxtpayi7ns2':'The paper identifies a Sanskrit-origin loan and compares Tamil vīṇai “lute”, but does not establish the immediate donor of Muduga bɯ̄ṇe “harmonium”; a remote Sanskrit link would overstate the evidence.',
 'f_4xl3hexw3hpuu':'The authors mark the proposed Kuwi mirka comparison under DEDR 4764 uncertain and expressly doubt DEDR 4616; neither is secure enough for a direct reflex edge.',
 'f_qaamur662enpi':'The paper compares Tamil montan, a cultivar of plantain, but does not establish inheritance or an immediate donor; no matching DEDR family has been verified.',
 'f_5blfobusatlr4':'Pēl-tuṟukki is a compound insect name, and the two constituents have not been verified against primary lexical evidence. The whole phrase should not be assigned to a single inherited stem.',
 'f_yjgwbryrjlb3m':'Kaccɯ-ṟādɯ is a complex wash expression used for phonetic illustration. Its constituent meanings and derivational structure are not established.',
 'f_52e4pznnyuaam':'The paper cites DEDR 2885 for the dove comparison, but Jambu lacks d2885. Its cached DEDR audit has two distinct articles both labelled 2835: page 246 “to drip” and page 250 Kota cōr “dove”, apparently a source-number collision. Do not link this bird word to the conflated d2835 node before the DEDR source is repaired. The near-homophonous Muduga cōre “blood” belongs to DEDR 2883 and is unrelated.',
 'f_jdkj7nkl5syw2':'Bē-ṟa “she gets cooked” is an inflected clause used for a phonetic example. The stem and suffix structure need analysis before etymologising the entire response.',
 'f_yr6txyrhyqm2o':'Toṯṯi “beak” lacks verified cognates or a matching DEDR bird-part family; the formally similar DEDR 3484 toṭṭi means “trough”, not beak.',
}
assert set(proposal)|set(reasons)=={r['ID'] for r in records}
accepted=[];held=[]
for i,r in enumerate(records,1):
 if r['ID'] in proposal:
  parent,evidence=proposal[r['ID']]
  accepted.append(dict(record=r,parent=parent,kind='reflex',citation=r['Source'].split(';dedr[')[0]+f';dedr[entry {parent[1:]}]',evidence=evidence,status='pending-review',surveyIndex=i))
 else:held.append(dict(record=r,reason=reasons[r['ID']],status='research-hold',surveyIndex=i))
(P/'proposed-decisions.json').write_text(json.dumps(dict(status='pending-review',accepted=accepted,held=held),ensure_ascii=False,indent=2)+'\n')
(P/'research-holds.json').write_text(json.dumps(held,ensure_ascii=False,indent=2)+'\n')
print('accepted',len(accepted),'held',len(held))
