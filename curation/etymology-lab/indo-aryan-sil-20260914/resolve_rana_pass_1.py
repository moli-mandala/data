"""Apply the explicit scholarly decisions made after reading the candidate review and CDIAL prose.
Discovery normalization does not mutate any lexical content. This script writes research only.
"""
import json,csv,re,collections,datetime
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
qs=json.loads((P/'previous-families.json').read_text());cs=json.loads((P/'resolve-rana-candidates-1.json').read_text());ps=json.loads((P/'primary-parents.json').read_text())
DARDIC={'Kohistani','Shinaic','Chitrali','Kunar','Pashai'}
# Named or immediate close comparanda from full primary articles, allowing the listed
# transcription differences; other Dardic matches need further local/borrowing work.
dardic={4:{'Bshk','Tor','Mai','Chil','Gowro'},5:{'Tor','Mai','Chil','Gowro','bhatr'},6:{'Phal'},10:{'Bshk','Mai'},11:{'Mai','Tor','Gowro'},15:{'Tor','Mai'},16:{'Tor','Phal','Chil','Gowro','bhatr'},26:{'Bshk','Mai','Chil','Gowro'},27:{'Gaw'},33:{'Bshk','Kal'},44:{'Bshk'},51:{'Kal','Kho'},52:{'Bshk','Gaw','Tor'},53:{'Bshk','Tor','Mai','Chil','Gowro','bhatr'},55:{'Bshk'},63:{'Bshk','Tor','Mai','Kal','Gaw','Chil','Gowro','bhatr'},67:{'Bshk','Tor','Mai','Kal','Chil'},68:{'Tor','Mai'},69:{'Kal','Mai','Chil'},70:{'Bshk','Tor','Mai','Chil','Gowro','Gaw','bhatr'},73:{'Phal'},78:{'Kal'},90:{'Bshk','Tor','Gaw','Mai','Gowro'},97:{'Bshk','Gaw'},130:{'Bshk','Tor','Mai'}}
dardic[133]={'Kal','Tor'}
# Remove citations to previous-task locality matrices: their evidence is not automatically
# transferable to a new speech variety. Scope-specific concerns are held below.
def evidence(i):
 s=qs[i]['evidence'][0]
 if i in (41,67,72,83):
  s=next((x for x in qs[i]['evidence'] if 'source matrix' not in x and 'locality comparison' not in x and 'four-family' not in x),s)
 s=re.sub(r'(?:; |\. )(?:Bagheli|Malvi|Nimadi|the Bagheli|the source meaning therefore).*$', '.',s)
 s=re.sub(r' Bagheli[^.]*\.', '',s)
 s=re.sub(r' The Bagheli[^.]*\.', '',s)
 s=re.sub(r'; Nimadi[^.]*\.', '.',s)
 if i==99:s='CDIAL 5798.4 places Hindi, Lahnda, Punjabi and Marathi tārā, Gujarati tāro, and Bengali/Oriya tārā in tāraka; this is distinct from the tārā and tārakā branches used for several Dardic comparanda.'
 if i==72:s='CDIAL 12278 śata gives Prakrit saya/saa, Hindi and Old Marwari sau, Gujarati sɔ, Nepali sai and eastern sai/so-type continuations; medial-t loss and vowel contraction support this numeral.'
 if i==11:s='CDIAL 4368 grāma gives Prakrit gāma, Gujarati gām, Marwari gā̃v, Hindi gā̃u and eastern gā̃, alongside Torwali gām, Maiya gā̃ and Gowro gaõ. The nasal and v/u outcomes occur in this village family.'
 if i==4:s='CDIAL 11572 vāla lists Hindi bāl, Gujarati vāḷ and Bhojpuri bār, and explicitly Gawri/Torwali bāl and Maiya bāla. These are hair comparanda; no similarly spelled word in another sense is included.'
 if i==16:s='CDIAL 14024 hasta gives Middle Indo-Aryan hattha, Hindi hāth, Marathi/Nepali hāt and eastern hāt/hāta, with both hand and forearm senses; Torwali hatth and Palula hāt support the included Dardic hand/arm forms.'
 if i==133:s='CDIAL 2462.2 specifically assigns the retained-k one-numerals (Hindi/Nepali/Bengali ek, Punjabi ikk) to *ēkka, while acknowledging that the Middle Indo-Aryan geminate may be emphatic or a learned replacement. The link stops at this section, not a claim about how *ēkka itself originated.'
 if i==134:s='CDIAL 12803.3 gives Pali/Prakrit cha, Hindi cha, Punjabi che and Nepali cha under *kṣaṭ/*kṣvaṭ. This selects the affricate branch, distinct from the ṣ- and ṣv- branches.'
 if i==135:s='CDIAL 135.1 aṅguli/aṅguri gives Hindi uṅglī, Gujarati ā̃gḷī, Marathi ãgḷī and eastern āṅguli. The selected l-bearing finger words fit that branch; the Dardic *aṅgūḍi branch is excluded.'
 if i==136:s='CDIAL 4701 explicitly lists the historical -ḍ- extension: Hindi camṛā, Gujarati cāmḍũ/cāmḍī and Marathi cāmḍẽ. These skin words select Jambu’s existing *carmaḍa extension node, not unextended carman.'
 if i==137:s='CDIAL 8249 gives Prakrit puccha, Punjabi pucch, Kumaoni pūch and Oriya pucha for tail. The selected unextended forms are kept separate from the -ḍ- extension and the nasalized puñcha branch.'
 if i==138:s='CDIAL 9757 matsara “mosquito” lists Hindi/Nepali/Maithili macchar, Lahnda macchur and Gujarati machrũ. Its proposed deeper connection with humming is uncertain; the link asserts only this mosquito family.'
 if i==139:s='CDIAL 10816 rētra compares Hindi ret/retī, Old Marwari reta, Gujarati ret/retī and Punjabi ret “sand”. The dictionary’s limited evidence for the reconstructed -tr- is preserved; no stronger remote derivation is asserted.'
 if i==140:s='CDIAL 11580 vālukā compares Hindi/Bihari bālū, Gujarati vāḷu and Marathi vāḷū. Only the u-final sand family is selected, distinct from the -ikā replacement underlying eastern bāli.'
 if i==141:s='CDIAL 3120 kāṣṭha gives Prakrit kaṭṭha and widespread kāṭh “wood”, with Hindi/Kachchi kāṭhī explicitly “wood”. Firewood is wood used as fuel; the selected retroflex forms need no respelling.'
 if i==142:s='CDIAL 2333 *uppari gives Punjabi uppar, Hindi ūpar, Gujarati upar, and eastern upari/upara “above”. The link is to this adverbial formation, rather than a similarly shaped adjective.'
 if i==143:s='CDIAL 5071 *chōṭṭa compares Hindi choṭā, Punjabi choṭṭā and Nepali choṭo “small”. Expressive origin remains unspecified; the selected forms denote size rather than a kinship term.'
 if i==144:s='CDIAL 3104.2 kalya gives Prakrit kalla/kalhiṃ for tomorrow or yesterday, Hindi kal, Bengali kāl and Gujarati kāl. Both elicited day-relative senses are supported; the kālya branch is not selected without local evidence.'
 return s
accepted=[];held=[]
for x in cs:
 r=x['record'];inds=x['families'];parents={qs[i]['parent'] for i in inds};i=inds[0];q=qs[i];w=r['Form'];l=r['Language_ID'];cl=r['clade'];reason=None
 if len(parents)!=1:reason='Competing dictionary families after discovery normalization; examine the exact stem and subsection.'
 elif False: reason=None  # Locality-only flag resolved below.
 elif cl in DARDIC and l not in dardic.get(i,set()):reason='The general Indo-Aryan match does not settle the local Dardic reflex or immediate borrowing route; do not propagate the plains analysis.'
 elif i in (41,67,72,83) and w.startswith('h'):reason='Initial s/ś > h needs a survey-local correspondence check; the previous Malvi locality evidence does not prove this new lect.'
 elif i==79 and 'ā' not in w and 'aː' not in w:reason='Short-vowel cal- is not diagnostic between calati and calyati; long-vowel cāl- evidence cannot be transferred.'
 elif i==55 and not re.search(r'[iyī]',w) and l not in {'awan','srk'}:reason='Bare gā/gã does not distinguish gāvā from gāvī without the local paradigm; CDIAL gives separate branches.'
 elif i==85 and (cl=='Lahndic' or l in {'Goj','kul'}):reason='Northwestern macchī may belong to matsiya (9758.2); inspect local evidence before choosing matsya versus matsiya.'
 elif i==59 and (cl in {'Lahndic','W. Pahari'} or l=='Goj'):reason='The mango family is clear, but the northwestern distribution may involve a regional loan; immediate donor not established.'
 elif i==109 and (cl in {'Lahndic','W. Pahari','E. Pahari'} or l in {'Goj','Bote','Majhi'}):reason='Banana kel-/ker- may be a contact form here (CDIAL explicitly labels Nepali kero a Maithili loan); donor pathway requires review.'
 elif i==108 and cl=='Bihari' and not re.search('r|ɾ',w):reason='Retained lateral in Bihari phal may indicate Hindi/Bengali contact; CDIAL distinguishes inherited phar from phal loans.'
 elif i==123 and l=='jaun':reason='Jaunsari bhēs contrasts with CDIAL local mahiś and may reflect Hindi contact; do not infer inheritance from the family resemblance.'
 elif i==39 and w.strip() in {'naī','nai'}:reason='Bare feminine nai does not distinguish nava and naviya.'
 elif i==40 and w.strip() in {'naī','nai'}:reason='Bare feminine nai does not distinguish nava and naviya.'
 elif i==135 and not re.search('[iī]$',w):reason='Final-l finger noun may continue aṅgula rather than aṅguli; full paradigm needed before selecting the exact parent.'
 elif i==137 and re.search('[̃ũõãñŋn]',w):reason='Nasalized tail form may require the distinct puñcha branch; do not collapse it into plain puccha.'
 elif i==144 and cl=='W. Pahari':reason='Western Pahari tomorrow-forms may continue kālya (3104.1), rather than the kalya branch; exact local form needs checking.'
 elif i==95 and cl=='W. Pahari':reason='Western Pahari pestle forms require comparison with the separate muśala/muṣala branch.'
 elif i==50 and cl in {'Bihari','Eastern'}:reason='Onion/root sense and transmission of kanda require a local check outside the directly supported western onion family.'
 elif i==70 and cl=='Lahndic' and w=='das':reason='The s-final ten-form contrasts with local dā(h); contact or local phonology needs confirmation.'
 if reason:held.append(dict(**x,reason=reason));continue
 ev=evidence(i)
 if cl in DARDIC:ev+=' The local form was checked against the Dardic comparanda in the full CDIAL article; no borrowing arrow was applied to this selected comparison.'
 accepted.append(dict(record=r,family=i,parent=q['parent'],citation=';'.join(q['citations']),evidence=ev))
(P/'resolve-rana-decisions-1.json').write_text(json.dumps({'accepted':accepted,'held':held},ensure_ascii=False,indent=1))
print('accepted',len(accepted),'held',len(held),'families',len({x['parent'] for x in accepted}))
