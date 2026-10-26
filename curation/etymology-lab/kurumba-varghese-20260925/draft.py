"""Draft review-only Kurumba Varghese analyses from exact records and full DEDR articles."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
D = P.parents[2]
records = json.loads((P / 'survey-inventory.json').read_text())
assert len(records) == 150
search = []
for a, b in [(1,30),(31,60),(61,90),(91,120),(121,150)]:
    search += json.loads((P / f'discovery-{a:03}-{b:03}-results.json').read_text())
assert len(search) == 150 and all(q['count'] == len(q['rows']) for q in search)

# The IDs refer to whole DEDR articles; the displayed database head is not a
# demand that every reflex match that particular reconstruction or sense.
groups = [
('d5031',[3,4],'DEDR 5031 includes Tamil mūñci, Malayalam muññi/moññi and Kodagu mu·ḍï “face”; Kurumba mūñḍźi/muñḍźi are affricated face variants.'),
('d1448',[5,6],'DEDR 1448 has Tamil/Malayalam kātu and PālKu. ka·du “ear”; Kurumba kātɨ/kādɨ preserve the dental ear stem.'),
('d3633',[7,8],'DEDR 3633 contains southern nāḷ-/nālig- “tongue” forms; Kurumba nāḷaŋge keeps that stem with a local extension.'),
('d5409',[9],'DEDR 5409 compares Tamil/Malayalam viral and Kannada beraḷ “finger”; Kurumba bira l is the lateral finger form, with source-internal spacing retained.'),
('d5393',[11,12],'DEDR 5393 includes Tamil vīṭu “house, habitation” and Malayalam vīṭu “house”; Kurumba vīḍɨ/viḍɨ match the dwelling sense.'),
('d5440',[13,14],'DEDR 5440 gives Tamil viṟaku and Malayalam viṟuku “firewood”; Kurumba birā shares the fuel sense with initial b- and a reduced ending.'),
('d4867',[46],'DEDR 4867 includes Malayalam muḷaku “pepper” and Kodagu maḷu “chillies”; Kurumba mɔḷa retains the chilli stem.'),
('d4096',[54],'DEDR 4096 explicitly compares Tamil/Malayalam/Kannada pāl “milk” and Kodagu pa·lï; Kurumba pælɨ is the same milk stem.'),
('d3650',[55],'DEDR 3650 gives Tamil/Malayalam nāy and Kannada nāyi “dog”; Kurumba nāyɨ is the matching dog noun.'),
('d4616',[58],'DEDR 4616 gives Tamil makaṉ and Malayalam makan “son”; Kurumba magæ is a corresponding son form.'),
('d601',[76,77],'DEDR 601 gives Tamil uṇaṅku and Malayalam uṇaṅṅuka “dry”; Kurumba uṇandɨtɨ is an inflected dry response preserving the uṇa(n)- base.'),
('d5496',[86,87],'DEDR 5496 includes Tamil veḷḷai and Malayalam veḷḷa “white”; Kurumba vɛḷḷɛ retains the color stem.'),
('d474',[89,90],'DEDR 474 gives Tamil iraṇṭu and Malayalam raṇṭu “two”; Kurumba ra ːḍɨ preserves the two numeral with source-internal spacing.'),
('d3655',[91],'DEDR 3655 includes Tamil nāṉku, Malayalam nāṅku and Kannada nāku “four”; Kurumba nākkɨ directly fits the four numeral.'),
('d5151',[94,95],'DEDR 5151 gives Tamil yār/ār and Malayalam ār “who”; Kurumba ārɨ is the person interrogative.'),
('d5151',[96,97],'DEDR 5151 contains the e-/en- interrogative family and Malayalam entu “what”; Kurumba ɛndɨ is a matching what form.'),
('d5151',[98,99],'DEDR 5151 gives Tamil eṅku and Malayalam eṅṅu “where”; Kurumba ɛ̄ŋkɨ is the locative interrogative.'),
('d5151',[101],'DEDR 5151 includes Tamil eppōtu and Malayalam eppōḷ “when”; Kurumba ɛppɨ preserves the temporal epp- base.'),
('d5151',[103],'DEDR 5151 gives Tamil ettaṉai and Malayalam ettira/etra “how many”; Kurumba ɛttɨnɨ has the same quantity-interrogative base.'),
('d410',[105,106],'DEDR 410 gives Tamil itu and Malayalam itu “this thing”; Kurumba idɨ is the proximal demonstrative.'),
('d1',[107,108],'DEDR 1 gives Tamil atu and Malayalam atu “that thing”; Kurumba adɨ is the distal demonstrative.'),
('d1124',[117,118,119],'DEDR 1124 includes Kannada kaḍi and Malayalam kaṭikka “bite”; Kurumba kaḍi/kaḍittɨ preserve the bite stem, with the past ending retained in the source response.'),
('d3794',[131,132],'DEDR 3794 includes Kannada nōḍu “look, see”; Kurumba nōḍɨ/noḍu match the visual verb despite the elicitation’s mixed present/past gloss.'),
('d5160',[133,134],'DEDR 5160 gives Tamil nāṉ and Malayalam ñān “I”; Kurumba nānɨ is the first-person singular pronoun.'),
('d3684',[135,136,138],'DEDR 3684 gives Tamil/Malayalam nī “you” singular; Kurumba ni is this second-person stem, also entered in a formal elicitation slot.'),
('d3688',[137,147,148],'DEDR 3688 includes Kannada nīm/nīvu with nim(m)- oblique for plural or honorific “you”; Kurumba nimma serves formal and plural slots.'),
('d1',[139,140],'DEDR 1 includes Tamil avaṉ and Malayalam avan “he”; Kurumba avanɨ retains the distal masculine pronoun.'),
('d1',[141,142],'DEDR 1 includes Tamil avaḷ and Malayalam avaḷ “she”; Kurumba avaḷɨ/avaḷu retain the distal feminine pronoun.'),
('d3647',[143,144],'DEDR 3647 includes Tamil/Malayalam nām and nam- oblique for inclusive “we”; Kurumba namma fits the inclusive stem.'),
('d5154',[145,146],'DEDR 5154 includes Tamil yām with em- oblique and Kannada em- for exclusive “we”; Kurumba ɛmma/emmā fit the exclusive series.'),
('d1',[149,150],'DEDR 1 includes Tamil and Kannada avar “they”; Kurumba avarɨ matches the distal plural pronoun.'),
]
proposals = {}
for parent, indices, evidence in groups:
    for n in indices:
        assert n not in proposals
        proposals[n] = ([parent], evidence)
for n in (92,93):
    proposals[n] = (['d3918','d990'], 'Kurumba patinɔṇṇɨ/patinɔṇṇu “eleven” joins the patiṉ- “ten in 11–19” form in DEDR 3918 and onnu/one in DEDR 990; the ordered constituents remain visible.')

hold_categories = {
    1:'body form appears śarīra-like, but its second element and immediate donor remain uncertain',
    2:'body variant remains a possible loan with no verified immediate donor',
    10:'blood form has no verified same-sense DEDR article',
    15:'mortar form appears to contain ural, but the preceding element is unanalysed',
    16:'mortar variant retains the unanalysed preceding element',
    17:'pestle compound contains kallɨ “stone”, but its first element and whole tool construction are unverified',
    18:'pestle duplicate retains the unresolved first component',
    19:'hammer form has no securely verified tool-family article',
    20:'hammer duplicate has no securely verified tool-family article',
    21:'axe form may be a regional loan, with immediate donor unverified',
    22:'axe duplicate has the same unresolved donor',
    23:'needle form resembles a sūci-type loan; immediate donor unverified',
    24:'needle duplicate retains the unresolved immediate donor',
    25:'needle affricate form retains an unresolved sūci-type donor',
    26:'needle affricate duplicate retains the unresolved donor',
    27:'ring form resembles a mudrā-type loan; immediate donor unverified',
    28:'ring variant retains the unresolved donor',
    29:'sun form has no securely matching same-sense article',
    30:'sun variant remains without a secure article',
    31:'river form has no secure same-sense article',
    32:'river form could relate to “ford/crossing”; river sense and derivation unverified',
    33:'rainbow compound is not securely segmented',
    34:'rainbow compound may involve Bhīma and a bow element; donor and segmentation unverified',
    35:'path form has no secure inherited article',
    36:'path variant has no secure inherited article',
    37:'ash form is bracketed in DEDR 4316 as a probable Sanskrit loan; immediate donor unverified',
    38:'ash variant is bracketed in DEDR 4316 as a probable Sanskrit loan; immediate donor unverified',
    39:'mud form does not securely match the cēṟu/kesaṟ article',
    40:'gold form is a svarṇa-type loan; immediate donor unverified',
    41:'wheat form is a godhuma-type loan; immediate donor unverified',
    42:'wheat duplicate retains the unresolved donor',
    43:'potato name is a compound or loan with unverified constituents',
    44:'potato variant has unresolved constituents and donor',
    45:'chilli name may combine the pepper stem with kai “fruit”; the second constituent and lexicalized construction need verification',
    47:'onion name is a regional loan with unverified immediate donor',
    48:'onion variant has unverified immediate donor',
    49:'tomato name has unresolved compound-like structure',
    50:'tomato name is a takkaḷi-type regional loan; immediate donor unverified',
    51:'tomato variant has no securely verified inherited family',
    52:'cabbage name is a recent loan; immediate donor unverified',
    53:'cabbage variant is a recent loan; immediate donor unverified',
    56:'mosquito form lacks a secure article',
    57:'mosquito variant lacks a secure article',
    59:'night form may be jāma-type loan; immediate donor unverified',
    60:'night variant retains the unresolved donor',
    61:'noon term may be a multi-part mad(y)āhna-type loan; immediate donor unverified',
    62:'noon variant retains the unresolved donor',
    63:'evening term is compound-like and its constituents are unverified',
    64:'evening duplicate retains unresolved constituents',
    65:'week term is vāra-type loan; immediate donor unverified',
    66:'week variant retains unresolved donor',
    67:'week duplicate retains unresolved donor',
    68:'month term is māsa-type loan; immediate donor unverified',
    69:'month variant retains unresolved donor',
    70:'year term is varṣa-type loan; immediate donor unverified',
    71:'year duplicate retains unresolved donor',
    72:'year variant retains unresolved donor',
    73:'year variant retains unresolved donor',
    74:'bad form has no verified same-sense article',
    75:'bad variant has no verified same-sense article',
    78:'long form may be a comparative regional loan; exact donor unverified',
    79:'long duplicate retains unresolved donor',
    80:'right compound has an apparent directional base plus sāri; construction unverified',
    81:'left compound has an apparent directional base plus sāri; construction unverified',
    82:'far form may be a dūra-type loan; immediate donor unverified',
    83:'far duplicate retains unresolved donor',
    84:'heavy phrase is compound-like with unverified segmentation',
    85:'light phrase is compound-like with unverified segmentation',
    88:'red form has no secure same-sense article',
    100:'when form looks interrogative plus “day”, but component segmentation is unverified',
    102:'quantity interrogative differs from the verified ettaṉai comparison and remains uncertain',
    104:'what-kind compound has unverified second constituent',
    109:'plural demonstrative may include an “all” element; whole response segmentation unverified',
    110:'plural distal demonstrative may include an “all” element; whole response segmentation unverified',
    111:'same phrase is compound-like and its second constituent may be a loan',
    112:'same form has no verified comparison',
    113:'few form has no verified same-sense article',
    114:'few variant has no verified same-sense article',
    115:'many phrase has unverified constituent structure',
    116:'many duplicate retains unresolved constituents',
    120:'hunger response is a multi-part inflected predicate, with segmentation unverified',
    121:'hunger variant is an inflected predicate, with segmentation unverified',
    122:'thirst response is multi-part and its constituents are unverified',
    123:'thirst variant lacks a secure parent analysis',
    124:'burn response is inflected and its stem has no secure article',
    125:'burn duplicate retains uncertain stem analysis',
    126:'negative die response is inflected and its verbal base is unverified',
    127:'negative die response combines a verbal base and prohibitive, unverified as a whole',
    128:'negative kill response contains koll- but the prohibitive construction needs ordered segmentation',
    129:'negative kill variant retains unresolved construction',
    130:'go response has an unverified negative or auxiliary ending',
}
assert set(hold_categories) == set(range(1,151)) - set(proposals), (sorted(set(range(1,151))-set(proposals)-set(hold_categories)), sorted(set(hold_categories)&set(proposals)))
ids = {p for ps,_ in proposals.values() for p in ps}
articles = {}
with gzip.open(D/'data/dedr/entry-texts-audit.csv.gz','rt') as f:
    for r in csv.DictReader(f):
        if r['Entry_ID'] in ids: articles[r['Entry_ID']] = r
assert set(articles)==ids
(P/'primary-evidence.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2)+'\n')
accepted=[];held=[]
for n,r in enumerate(records,1):
    if n in proposals:
        ps,evidence=proposals[n]
        x={'record':r,'parent':ps[0],'kind':'component' if len(ps)>1 else 'reflex',
           'citation':r['Source']+';'+','.join('dedr['+p[1:]+']' for p in ps),
           'evidence':evidence,'comparison':'Homepage exact search, full DEDR article, source item and dialect record checked; whole-article family identification.',
           'status':'pending-review'}
        if len(ps)>1:x['components']=ps
        accepted.append(x)
    else:
        held.append({'record':r,'candidates':[],'reason':r['Form']+' “'+r['Gloss']+'”: '+hold_categories[n]+'.'})
(P/'proposed-decisions.json').write_text(json.dumps({'status':'pending-review','language':'Kurumba','source':'varghese2015palakkad','accepted':accepted,'held':held},ensure_ascii=False,indent=2)+'\n')
(P/'research-holds.json').write_text(json.dumps(held,ensure_ascii=False,indent=2)+'\n')
print('proposed',len(accepted),'rows',sum(len(x.get('components',[x['parent']])) for x in accepted),'held',len(held),'articles',len(articles))
