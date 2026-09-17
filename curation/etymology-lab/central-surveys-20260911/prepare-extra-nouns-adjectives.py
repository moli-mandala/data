import sys
sys.path.insert(0,'data/curation/etymology-lab/central-surveys-20260911')
from research_helpers import Batch
b=Batch(12)
def a(l,g,w,p,e,t='straightforward',s=None):
 rows=[r for r in b.inv[l] if r['ID'] not in b.used and r['Form'] in w.split('|') and (r['Gloss']==g if s is None else set(r['Gloss'].split('; '))<=set(s))]
 if rows:b.add(l,g,'|'.join(dict.fromkeys(r['Form'] for r in rows)),p,e,tier=t,source_glosses=list(dict.fromkeys(r['Gloss'] for r in rows)))
for l,w in [('Malvi','āmbā|āmbo'),('Nimadi','āmbā|āmbo'),('Bagheli','eme|am|ama|amba')]:
 a(l,'mango',w,'1268','CDIAL 1268 gives Pali/Prakrit amba, Hindi ām/ā̃b, Old Awadhi āṁba and Gujarati ā̃b/ām. Retained mb and its simplification both belong to the mango family. '+('The e-vowel in eme needs local confirmation.' if l=='Bagheli' else 'Final vowels accord with the western forms.'),'qualified' if l=='Bagheli' else 'straightforward')
for l,w in [('Malvi','caval|savaḷə|sāvar|cavar|cāvəl|camal|cavaḷ'),('Nimadi','caval'),('Bagheli','cevur|caur|cauur|cauel|caual')]:
 a(l,'rice',w,'4749','CDIAL 4749 explicitly offers *cāmala OR *cāvala: Prakrit cāulā/cavala, Bhojpuri cāur, Hindi cāwal/cā̃war and Gujarati cāvaḷ. The remote origin is uncertain; both reconstructions remain alternatives. '+('Sibilant initials and l/r outcomes need dialectal review.' if l=='Malvi' else 'Contraction and regional vowel/liquid outcomes are compatible with this family.'),'qualified')
for l,w in [('Malvi','marca|maraca|marcā|marac|mircā|mirci|marəcā|maras|marsa|maraś|marcyā|maraci|mārcā|marəci|mariśā'),('Nimadi','marca|mirci|mircyā')]:
 a(l,'chilli',w,'9875-2','CDIAL 9875.2 *maricca gives Punjabi marc/mirc, Bhojpuri maricā ‘chillies’, Awadhi mircā and Gujarati marcī/marcũ ‘red pepper’. The chilli meaning is an attested later extension of the pepper family. Palatal versus sibilant outcomes and local transmission need review; this is not a claim that chilli was the ancient referent.','qualified')
a('Nimadi','chilli','miri|mirin','9875','CDIAL 9875 marīca gives Prakrit miria, Hindi mirī and Gujarati marī; Bastar Oriya miri already means red pepper. Loss of medial c fits this branch rather than *maricca. The chilli extension and final n of mirin remain qualified.','qualified')
for l,w in [('Malvi','kiḍi|kiḍiyu'),('Nimadi','kiḍi|kiḍa')]:
 a(l,'ant',w,'3193','CDIAL 3193 gives Prakrit kīḍa/kīḍī/kīḍiyā ‘worm, insect, ant’, Hindi kīṛī and Gujarati kīṛī ‘ant’. The feminine/extended forms fit directly; masculine kiḍa may represent a broader insect word used for the elicited ant.','qualified' if l=='Nimadi' else 'straightforward')
for l,w in [('Malvi','gilːa|gilo|gilːo|gilā'),('Nimadi','gilo|giḷo'),('Bagheli','gil|gila')]:
 a(l,'wet',w,'4386','CDIAL 4386 *grilla compares Prakrit gilla, Hindi gīlā/gillā and Old Marwari gīlau ‘wet’. Turner calls the proposed deeper *gr̥dla derivation very doubtful; the present proposal only identifies this attested wet-adjective family.','qualified')
for l,w in [('Malvi','alo|ālo|alːo|alā|ālːā|ālːo'),('Nimadi','ālo|aḷḷa|allo|alo')]:
 a(l,'wet',w,'1340-2','CDIAL 1340.2 specifically gives *ālla < *ārdla, Pali/Prakrit alla, Hindi ālā and Old Marwari ālo ‘wet’. It supplies the relevant l/geminate-l branch rather than an unexplained direct link to ārdra.')
for l,w in [('Malvi','baḍo|baḍa|bəḍā'),('Nimadi','baḍo|bar'),('Bagheli','beḍa|beḍḍa|beḍḍe|bereka|berka')]:
 a(l,'big',w,'11225','CDIAL 11225 compares Prakrit vaḍḍa, Hindi baṛā/baḍḍā, Marwari baṛo and Awadhi baṛkā. Its addenda derive vaḍra by extraction from formations such as evaḍa/kevaḍa, ultimately involving -vant plus -ḍa; this is not a simple direct descent from vṛddha. Regional vowel and liquid variation remains for triage.','qualified')
a('Malvi','red','rāto|rata|rātu','10539','CDIAL 10539 gives Pali/Prakrit ratta, Hindi rātā and Gujarati rātũ ‘red’. The kt > tt > t development and adjective endings support this colour family.')
a('Malvi','red','rāṭo','10539','The red meaning and rā-t stem compare Hindi rātā and Gujarati rātũ in CDIAL 10539, but the source’s retroflex ṭ needs local explanation and is not silently normalized.','qualified')
a('Bagheli','heavy','geri|geru|geṛu|garu','4209','CDIAL 4209 gives Pali/Prakrit garu/garua, Awadhi garū and Hindi garuā ‘heavy’. Bagheli e and the occasional retroflex liquid need confirmation; the garu family is proposed with those qualifications.','qualified')
a('Bagheli','whole; good','nikaha|nikeha','7150','CDIAL 7150 explicitly supplies MIA *nikka, Prakrit ṇikka ‘clear’, Maithili nīk/nikāh and Awadhi nīk ‘good’. The source combines whole and good; the cited article supports good, while whole and the extended ending require review. Turner separately notes Persian nēk influence on Hindi nekā, so a unique transmission path is not asserted.','qualified',s=['whole','good'])
b.save()
