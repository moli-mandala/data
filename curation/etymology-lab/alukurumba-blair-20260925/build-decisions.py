"""Conservative pending proposals for residual Blair et al. Alu Kurumba records."""
import csv
import gzip
import json
from pathlib import Path

p = Path(__file__).resolve().parent
records = json.loads((p / 'survey-inventory.json').read_text())
assert len(records) == 328
proposals = {}
holds = {}

def add(indices, number, evidence, components=None):
    for i in indices:
        assert i not in proposals
        proposals[i] = (number,evidence,components)

def hold(indices, reason):
    for i in indices:
        assert i not in holds
        holds[i] = reason

add([1],497,"DEDR 497 gives Tamil ilai, Kota el and Kannada ele 'leaf'; Alu Kurumba śele belongs to this western leaf family.")
add([2],4345,"DEDR 4345 gives Tamil/Malayalam pū and Kannada pūvu 'flower'; Alu Kurumba ū is a reduced local flower form also attested in the earlier Alu Kurumba survey.")
add([4],5401,"DEDR 5401 gives Tamil vittu and Kannada bittu 'seed'; Alu Kurumba bɨttɨ retains the seed noun.")
add([5],4711,"Marapātte has visible tree + bark components: DEDR 4711 gives Kannada mara 'tree', and DEDR 3876 gives Kannada paṭṭe 'tree bark'. The component order is preserved.",[4711,3876])
add([6],1514,"DEDR 1514 directly gives Kurub. ticci, Kannada kiccu and Tulu kicci 'fire'; Alu Kurumba kīśśu belongs to this fire family.")
add([9],4559,"DEDR 4559 gives Kota poṛt 'sun' and related sun/time forms; Alu Kurumba vottu agrees with western reflexes of this article.")
add([12],1499,"DEDR 1499 gives Kannada gāḷi and Kodagu ga·ḷi 'wind'; Alu Kurumba gāśī is the same wind noun with local consonant development.")
add([14,29],5474,"DEDR 5474 gives Kannada beṭṭa and Kodagu beṭṭa 'mountain, hill'; both Alu Kurumba battu elicitation records match.")
add([16],1355,"DEDR 1355 gives Kannada gadde 'paddyfield' and Kota gad; Alu Kurumba gaḍḍe is specifically a wet field.")
add([17],2891,"DEDR 2891 gives Tamil cōlai 'grove' and Kota te·l 'forest'; its Badaga sōle is explicitly a tree-covered jungle, matching Alu Kurumba sole 'forest'.")
add([20],5354,"DEDR 5354 gives Malayalam vātil 'doorway', Kannada bāgil 'gate, doorway', and Tulu bākilů 'door'; Alu Kurumba aḍɨlu preserves the door term with the same medial stop and final liquid.")
add([21],2207,"DEDR 2207(b) gives Kannada gōḍe and Tulu gōḍè 'wall'; Alu Kurumba goḍe is a close match to this wall subsection.")
add([22],5538,"DEDR 5538 gives Tamil vēli and Kannada bēli 'fence, hedge'; Alu Kurumba belɨ retains the fence noun.")
add([33],1118,"DEDR 1118 gives Tamil/Malayalam kaṭal and Kannada kaḍal 'sea'; Alu Kurumba kāḍal is a close match.")
add([34],399,"DEDR 399 gives Tamil āḷ 'man, adult' and Malayalam āḷ 'person'; Alu Kurumba ālu has the general person sense.")
add([36],4395,"DEDR 4395 groups Tamil peṇ, Kannada peṇṇu/heṇṇu and related western 'woman, female' forms; Alu Kurumba ɛṇṇugā retains that feminine base with a local ending.")
add([38],4410,"DEDR 4410 gives Tamil peyar and Kannada pesar(u) 'name'; Alu Kurumba yedzaru resembles the western name form.")
add([40],4682,"DEDR 4682 gives Kannada maṇḍe and Kota maṇḍ 'head'; Alu Kurumba maṇḍe is an exact match.")
add([41],3759,"DEDR 3759 gives Kannada netti and Kodagu netti 'forehead'; Alu Kurumba neṭṭɨ matches.")
add([43],5352,"DEDR 5352 gives Kannada bāyi and Kodagu bayi 'mouth'; Alu Kurumba bāyɨ is the same mouth noun.")
add([44],3986,"DEDR 3986(a) gives Tamil pal, Malayalam pallu and Kannada pal 'tooth'; Alu Kurumba hallu preserves the same tooth family with the western h- variant.")
add([45],3633,"DEDR 3633 gives Kannada nālige and Tulu nālā̆yi 'tongue'; Alu Kurumba nālaŋgā belongs to this tongue family.")
add([46],1366,"DEDR 1366 gives Kannada kattu 'neck, throat' and the closely related Badaga gattu; Alu Kurumba gaṭṭu retains the neck sense.")
add([51],3736,"DEDR 3736 gives Tamil neñcu 'heart, breast, bosom, chest', Kodagu neññi 'chest of body', and Konda ninjam 'chest'; Alu Kurumba neñdzu preserves the chest noun.")
add([52],4096,"DEDR 4096 gives Tamil/Malayalam pāl and Kannada pāl 'milk'; Alu Kurumba ālu preserves the milk noun.")
add([53,54],2023,"DEDR 2023 gives Tamil kai and Kannada kay/kayi 'hand, arm'; both Alu Kurumba kā' body-part records preserve that base.")
add([55],7,"DEDR 7 includes Kota aŋga·l 'instep', Kodagu aŋga·lï 'sole of foot', and Tulu aṅgārů 'sole of the foot'; Alu Kurumba aŋgalu is a close foot reflex within the full article.")
add([61],2426,"DEDR 2426 gives Kannada sāvu 'death, corpse' and Kodagu ca·vu 'corpse'; Alu Kurumba sāvu is the corresponding noun.")
add([62],393,"DEDR 393 gives Tamil āvi 'soul, breath' and Kota a·yv 'soul'; Alu Kurumba āvī preserves the spirit sense within this whole article.")
add([72],3302,"DEDR 3302 gives Kannada toḍe 'thigh'; Alu Kurumba ṭoḍa is the same body-part noun.")
add([78],1159,"The Alu Kurumba tears word kāŋŋīru transparently joins eye and water. DEDR 1159(b) itself lists Tamil, Malayalam and Kannada kaṇ ṇīr 'tears'; DEDR 3690 gives the independent nīr 'water' family. Both ordered constituents are retained.",[1159,3690])
add([82],2903,"DEDR 2903 gives Tamil narampu 'blood-vessel', Malayalam narampu 'nerve, pulse', Tulu narambu 'sinew, nerve', and Telugu naramu 'vein'; Alu Kurumba nerambu is a close vein reflex.")
add([85],546,"DEDR 546 gives Tamil īral, Malayalam īral and Kannada hīri 'liver'; Alu Kurumba yīralu matches this organ noun.")
add([86],1274,"DEDR 1274 gives Kannada karuḷ and Kodagu karï 'intestines, bowels'; Alu Kurumba karu preserves this organ noun.")
add([87],4886,"DEDR 4886 gives Tamil mūccu 'respiration, breath' and Kannada mūsu 'smell'; Alu Kurumba mūsu keeps the respiratory sense, also attested in Badaga.")
add([89],2115,"DEDR 2115 gives Kannada kombu and Tulu kombu 'horn'; Alu Kurumba kambu retains the horn noun.")
add([92],4449,"DEDR 4449 gives Tamil pēṉ and Kannada pēn 'louse'; Alu Kurumba yenu preserves the head-louse sense.")
add([93],99,"DEDR 99 gives Tamil aṭṭai and Kannada aṭṭe 'leech'; Alu Kurumba atta retains the land-leech noun.")
add([96],2901,"DEDR 2901 gives Tamil naḷḷi and Kannada naḷḷi 'crab'; Alu Kurumba nallī is a close match.")
add([97],4085,"DEDR 4085 gives Kannada hāvu and Tulu hāvu 'snake'; Alu Kurumba āvu retains this snake noun.")
add([98],833,"DEDR 833 gives Tamil/Malayalam eli and Kannada eli/ili 'rat'; Alu Kurumba yelī is the same rodent noun.")
add([99],3650,"DEDR 3650 gives Tamil nāy and Kannada nāyi 'dog'; Alu Kurumba nā' matches this family.")
add([102],4307,"DEDR 4307 gives Tamil puli and Tulu pili 'tiger'; Alu Kurumba vulī belongs to the same tiger family.")
add([103],1263,"DEDR 1263 gives Kannada karaḍi 'bear'; Alu Kurumba keraḍī is an especially close match.")
add([107],3470,"DEDR 3470 gives Tamil tēḷ, Tulu tēlů and Telugu tēlu 'scorpion'; Alu Kurumba ṭelu retains this arthropod noun.")
add([108],3591,"DEDR 3591 gives Malayalam natta and Telugu natta 'snail'; Alu Kurumba naṭṭa is an exact local match.")
add([109],4312,"DEDR 4312 gives Tamil pur̤u and Kannada pur̤u 'worm', and the South-Central reflex Gadaba puḍut 'insect'; Alu Kurumba puṭu belongs to this worm family.")
add([116],5370,"DEDR 5370 gives Tamil/Malayalam vavvāl 'bat'; Alu Kurumba vavvāl is exact.")
add([117],1224,"DEDR 1224(a) gives Kannada kappe and Tulu kappè 'frog'; Alu Kurumba kappa retains the frog noun.")
add([123],637,"DEDR 637 gives Tamil umi and Kannada ummi 'husk, chaff'; Alu Kurumba vumī has the same husk meaning.")
add([125],5422,"DEDR 5422 gives Kannada/Tulu billu 'bow'; Alu Kurumba bīllu is the same bow noun.")
add([126],178,"DEDR 178 gives Kannada and Tulu ambu 'arrow'; Alu Kurumba ambu is an exact match.")
add([127],5288,"DEDR 5288 gives Kannada bale and Kodagu bale 'net'; Alu Kurumba bale is exact.")
add([128],4088,"DEDR 4088 gives Tamil/Malayalam pāy 'mat'; Alu Kurumba pāyī is a matching mat noun.")
add([135],2306,"DEDR 2306 gives Tamil caṭṭi and Kannada caṭṭi 'earthen pot'; Alu Kurumba sattī preserves the pot noun.")
add([136],4032,"DEDR 4032 gives Malayalam paṟa and Kannada paṟe 'drum'; Alu Kurumba are resembles their reduced western drum forms.")
add([137],1613,"DEDR 1613 gives Tamil, Malayalam and Kannada kīri 'mongoose'; Alu Kurumba kīrī is exact.")
add([139],5161,"DEDR 5161 gives Tamil āṉai and Kannada/Kodagu āne 'elephant'; Alu Kurumba āne is exact.")
add([141],1917,"DEDR 1917 gives Kannada gūḷi and Kodagu gu·ḷi 'bull'; Alu Kurumba gūlī is a close match.")
add([142],4642,"DEDR 4642 gives Tamil mayil and Kannada maylu 'peacock'; Alu Kurumba mayīlu matches.")
add([143],1584,"DEDR 1584 gives Tamil kiḷi and Kannada giḷi 'parrot'; Alu Kurumba kīlī belongs to this bird-name family.")
add([145],1893,"DEDR 1893 directly records a Kurumba-dialect Kannada kūnu 'mushroom', beside Malayalam kūṇu; Alu Kurumba kūnu is exact.")
add([148],3268,"DEDR 3268 gives Kannada jēnu 'honey' and Tamil tēṉ; Alu Kurumba jenu retains the honey noun.")
add([150],4784,"DEDR 4784 gives Tamil māvu 'flour, meal' and related western forms; Alu Kurumba māvu is an exact food term.")
add([154],946,"DEDR 946 gives Tamil ōṭṭai 'hole' and related western forms; Alu Kurumba votta preserves the aperture noun.")
add([162],990,"DEDR 990 gives Kannada obba 'one man or person'; Alu Kurumba vabbā is the corresponding human classifier form for 'one person'.")
add([163],990,"DEDR 990 gives Kannada ondu 'one thing'; Alu Kurumba vaŋḍu is the corresponding nonhuman 'one' form.")
add([164,165],474,"DEDR 474 gives Malayalam raṇḍu and Kannada eraḍu 'two'; Alu Kurumba rādu occurs in both person and stone contexts.")
add([166],2826,"DEDR 2826 gives Kannada aidu and Tamil aintu 'five'; Alu Kurumba ā'ḍu is the numeral cognate.")
add([167],3918,"Annaŋḍu 'eleven' contains ten plus one: DEDR 3918 gives Kannada pann- in 11, while DEDR 990 gives Kannada ondu 'one'. Both ordered components are retained.",[3918,990])
add([168],3805,"DEDR 3805 gives Tamil/Malayalam pakal 'daytime'; Alu Kurumba agalu is the corresponding daylight noun.")
add([169],3656,"DEDR 3656 gives Tamil nāḷai and Malayalam nāḷa 'tomorrow'; Alu Kurumba nāŋu belongs to the day/tomorrow family.")
add([174],5020,"DEDR 5020 gives Tamil muṉ 'before' and Tulu mundè 'before, in front'; Alu Kurumba muŋḍādu preserves this temporal/spatial front root.")
add([176],1873,"DEDR 1873 gives Kannada kūsu 'infant, babe'; Alu Kurumba kūsu is an exact baby noun.")
add([177],4616,"DEDR 4616 gives Tamil makaṉ and Kannada maga 'son'; Alu Kurumba maŋź agrees with western child/son forms.")
add([178],131,"DEDR 131 gives Tamil aṇṇaṉ and Kannada aṇṇa 'elder brother'; Alu Kurumba aṇṇā is an exact kin term.")
add([179],3085,"DEDR 3085 gives Kannada tamma 'younger brother'; Alu Kurumba tammā is the corresponding kin term.")
add([180],23,"DEDR 23 gives Kannada akka and Kota akn 'elder sister'; Alu Kurumba akkā is exact.")
add([181],3015,"DEDR 3015 gives Kannada taṅgi and Irula taŋge 'younger sister'; Alu Kurumba taŋge is a close match.")
add([182],196,"DEDR 196(a) gives Kannada ayya 'father, grandfather' and Kota ayṇ 'father'; Alu Kurumba ayyā keeps the grandfather sense. The article notes only a questioned Sanskrit comparison.")
add([184,185],4715,"DEDR 4715 gives Kannada mom-maga 'grandson' and mom-magaḷu 'granddaughter'; Alu Kurumba mammakka/mammagā preserve these kinship formations.")
add([186,187],4813,"DEDR 4813 gives Tamil māmaṉ 'father-in-law' and māmi 'mother-in-law', with western cognates; Alu Kurumba māmā/māmɨ retain the pair.")
add([188],2410,"DEDR 2410 gives Kannada aḷiya and Tulu aḷiya 'son-in-law'; Alu Kurumba āṛṛyā belongs to this kin term.")
add([201],5160,"DEDR 5160 gives Tamil nāṉ and Kannada nān 'I'; Alu Kurumba na is the reduced first-person singular form.")
add([205],1,"DEDR 1 gives Tamil avaṉ and Kannada avan 'he'; Alu Kurumba ava retains the distal masculine pronoun.")
add([210],410,"DEDR 410(a) gives Tamil itu and Kannada idu 'this'; Alu Kurumba yīḍu preserves the proximal demonstrative.")
add([211,229],1,"DEDR 1 gives Tamil atu and Kannada adu 'that'; Alu Kurumba aḍu/aḍā are the two distal elicitation forms.")
add([212,213,214,232,233,234,235],5151,"DEDR 5151 groups the yār 'who', eṉṉa 'what', eṅkē 'where', and related interrogative stems; these Alu Kurumba question words retain that family with local endings.")
add([216],3748,"DEDR 3748 gives Kannada nettar(u) 'blood' and Gondi netral 'red'; Alu Kurumba neṭaru preserves the blood-to-red colour development.")
add([217],4635,"DEDR 4635 gives Tamil mañcaḷ and Tulu mañjalu 'turmeric, yellow'; Alu Kurumba maŋḍźā retains the colour term.")
add([220],5496,"DEDR 5496(a) gives Kannada biḷi and Tamil veḷ 'white'; Alu Kurumba belle preserves the white adjective.")
add([221],1942,"DEDR 1942 gives Tamil and Malayalam keṭṭa 'bad, spoiled'; Alu Kurumba kɔttaḍḍā retains the negative-quality stem with local adjectival morphology.")
add([231],1,"DEDR 1 gives the distal demonstrative base and locative 'there' forms, including Kannada alli; Alu Kurumba allī matches.")
add([230],410,"DEDR 410 gives Kannada illi 'in this place, here'; Alu Kurumba yɨllī preserves the proximal locative.")
add([238],8,"DEDR 8 gives Tamil akalam and Kannada agala 'breadth, wide'; Alu Kurumba agala is exact.")
add([240],5078,"DEDR 5078 gives Tamil mella/meḷḷa 'slowly, gently' and Kannada mella; Alu Kurumba mellā keeps the slow adverb/adjective sense.")
add([260],3569,"DEDR 3569 gives Tamil naku and Kannada nagu 'laugh'; Alu Kurumba nageḍaṭṭu contains the laugh base with local verbal morphology.")
add([259],2426,"DEDR 2426 gives Kannada sāy (sattu) 'to die' and Tamil cā (cett-) 'to die'; Alu Kurumba saṭṭuṭaḍḍu contains the attested sattu death stem with local verbal morphology.")
add([261],3582,"DEDR 3582 gives Tamil naṭa and Kannada naḍe 'walk'; Alu Kurumba nāḍe preserves this movement verb.")
add([267],916,"DEDR 916 gives Tamil ēṟu and Malayalam ēṟuka 'ascend'; Alu Kurumba eru has the matching direction and root.")
add([272,273],4797,"DEDR 4797 gives Kannada māḍu 'do, make'; both Alu Kurumba māḍu elicitation records are exact matches.")
add([275],1466,"DEDR 1466 gives Kota ka·rm 'hot taste of pepper or chilli' and Kannada kāra 'pungent'; Alu Kurumba kāra is specifically hot in the spicy sense.")
add([285],3323,"DEDR 3323 gives Tamil tuppu and Kannada tuppu 'spit'; Alu Kurumba ṭuppu matches this verb.")
add([288],347,"DEDR 347 gives Tamil āṭu and Kannada āḍu/āṭu 'dance, play'; Alu Kurumba āṭṭaḍu retains the dance verb with local verbal morphology.")
add([287],4065,"DEDR 4065 gives Tamil pāṭu and Kannada pāḍu 'sing'; Alu Kurumba āḍu is the local initial-p-loss form, also found in Badaga.")
add([300],1147,"DEDR 1147 gives Tamil kaṭṭu and Kannada kaṭṭu 'tie, bind'; Alu Kurumba kattu retains the verb.")
add([301],3549,"DEDR 3549 gives Kannada tōḍu 'dig'; Alu Kurumba ṭoŋḍu belongs to this earth-moving verb family.")
add([302],4797,"DEDR 4797 gives Kannada māḍu 'do, make, prepare'; the survey's māḍu response to 'cook' is the same preparation verb used in this elicitation context, not a separate cooking root.")
add([305],1372,"DEDR 1372 gives Tamil kaḷ and Malayalam kaḷkkuka 'steal'; Alu Kurumba kallu retains the steal root.")
add([308],5393,"DEDR 5393 gives Tamil viṭu 'dispatch, send away' and Tulu buḍupuni; Alu Kurumba buḍu retains the release/send verb.")
add([310],2598,"DEDR 2598 gives Tamil ī and Kannada ī 'give'; Alu Kurumba yīḍā preserves this transfer verb with local suffixing.")
add([311],2017,"DEDR 2017 gives Tamil kēḷ and Kannada kēḷu 'hear'; Alu Kurumba keɨ is a reduced local hearing verb.")
add([314],1443,"DEDR 1443 gives Tamil kāṇ 'see, consider' and kāṇkai 'knowledge'; Alu Kurumba kāŋḍo 'know' is a plausible knowledge extension of this visual root.")
add([325],3566,"DEDR 3566 gives Tamil tōṟṟu 'cause to appear, show' and the western tōr- family; Alu Kurumba ṭorcu matches this display verb.")
add([328],5401,"DEDR 5401 gives Tamil vitai and Kannada bittu 'sow, seed'; Alu Kurumba bɨṭṭu retains the sow verb.")

hold([10,11],"Tsaṇḍrā 'moon' and natcatrā 'star' resemble Sanskrit candra/nakṣatra regional vocabulary; the immediate donor is not established.")
hold([7],"Būḍī 'ash' matches bracketed Kannada būdi in DEDR 4316; that article flags bracketed forms as probably Sanskrit bhūti loans, and the immediate donor into this lect is unverified.")
hold([3],"Katte 'wood' resembles an exact Kannada form under DEDR 1675, but that article means woodworm decay, not wood; no secure parent follows.")
hold([28],"Hā 'cave' has a misleading Badaga hā match under DEDR 4110, whose article concerns empty or ruined places, not a cave noun.")
hold([134],"Kūḍe 'basket' has a same-form Sinhala search hit under DEDR 1883, but the primary article describes nests and enclosures rather than this basket type; immediate source unresolved.")
hold([218],"Ace 'green' resembles Kannada pace in DEDR 3821 but its missing initial consonant is not yet supported by a sound correspondence.")
hold([244,246],"Kunna 'thin' matches the small/young family in DEDR 1646, but neither the thin-sheet nor thin-rope sense is documented there; semantic extension needs evidence.")
hold([26],"Kāṇabɨllu 'rainbow' is compound-looking; its first constituent and full semantic construction need verification before ordered components are proposed.")
hold([27],"Būmɨalɨŋghā 'earthquake' is a multiword or multi-part elicitation response; the earth and motion elements require source segmentation.")
hold([35],"Manusā 'man' resembles Sanskrit manuṣya-type vocabulary; the immediate donor into this lect is unverified.")
hold([202,203],"The same nange form is explicitly elicited for inclusive and exclusive 'we'; DEDR 3647/5154 distinguish those paradigms, so a single reflex analysis requires source adjudication.")
hold([206,207,208,209],"The they/we/you dual response contains additional pronoun, numeral or all-quantifier material; its complete ordered segmentation is not established.")
hold([219],"Nīla 'blue' resembles Indo-Aryan nīla-type colour vocabulary; its immediate donor is unverified.")
hold([223,224],"Bārā 'heavy' and bārayīlla 'light' appear to pair weight vocabulary with a negator; direct inheritance or contact and the full negative construction remain unresolved.")
hold([256,257],"The broken/not-broken responses contain resultative and negative morphology; a simple root link would omit visible material.")
hold([262],"Paraŋḍānavogu 'fly' includes a fly-looking stem and a go auxiliary; the full compound predicate requires ordered analysis.")
hold([266],"Arugu is glossed 'descend' but has no secure matching descent-family comparandum in the checked full DEDR articles.")
hold([292,294,296,307,309,313,315],"The elicited predicate visibly contains an auxiliary, directional or other second element; complete ordered components have not been verified.")
hold([317],"Vɔḷɨṭṭu 'happy' may be a derived good/pleasure expression, but its segmentation and source sense are not secure.")

for i,r in enumerate(records,1):
    if i not in proposals and i not in holds:
        holds[i]=f"{r['Form']} ‘{r['Gloss']}’: no secure whole-form parent follows from the exact-form and global searches plus checked primary articles; retain for further sound, sense, or compound analysis."
assert not set(proposals)&set(holds)
assert set(proposals)|set(holds)==set(range(1,len(records)+1))

accepted=[];held=[]
for i,r in enumerate(records,1):
    if i in proposals:
        number,evidence,components=proposals[i]
        nums=components or [number]
        accepted.append(dict(record=r,parent=f'd{number}',kind='component' if components else 'reflex',
                             citation=r['Source']+';'+';'.join(f'dedr[entry {n}]' for n in nums),
                             evidence=evidence,status='pending-review',surveyIndex=i,
                             **({'components':[f'd{n}' for n in components]} if components else {})))
    else:held.append(dict(record=r,reason=holds[i],status='research-hold',surveyIndex=i))
parents={par for x in accepted for par in x.get('components',[x['parent']])};articles={}
with gzip.open(p.parents[2]/'data/dedr/entry-texts-audit.csv.gz','rt') as fh:
    for row in csv.DictReader(fh):
        if row['Entry_ID'] in parents:
            assert row['Entry_ID'] not in articles
            articles[row['Entry_ID']]={k:row[k] for k in ('Page','Citation','Raw_Article')}
            articles[row['Entry_ID']]['URL']='https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page='+row['Page']
assert set(articles)==parents,parents-set(articles)
(p/'proposed-decisions.json').write_text(json.dumps(dict(status='pending-review',accepted=accepted,held=held),ensure_ascii=False,indent=2)+'\n')
(p/'research-holds.json').write_text(json.dumps(held,ensure_ascii=False,indent=2)+'\n')
(p/'primary-evidence.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2)+'\n')
(p/'unexamined-inventory.json').write_text('[]\n')
print({'proposed':len(accepted),'rows':sum(len(x.get('components',[x['parent']])) for x in accepted),'held':len(held),'parents':len(parents)})
