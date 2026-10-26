"""Conservative pending Wolf Kota whole-article DEDR proposals."""
import csv,gzip,json,re
from pathlib import Path
P=Path(__file__).resolve().parent;DATA=P.parents[2]
records=json.loads((P/'survey-inventory.json').read_text());assert len(records)==907
search=[]
for start in range(0,907,35):
    end=min(start+35,907)
    search.extend(json.loads((P/f'discovery-{start+1:03d}-{end:03d}-results.json').read_text()))
assert len(search)==907
assert all(x['opts']['languageId']=='Kota' and x['opts']['params']['form']==r['Form'] for r,x in zip(records,search))
candidates={x['index']:x for x in json.loads((P/'dedr-kota-form-candidates.json').read_text())}
groups='''
19:122;26,842:31;51,52:2410;
77:276;119:2471;123:2356;133:2790;
139:2766;166:2703;168:2741;193:3375;224:910;
232:1389;239:1972;243:1941;245:2093;249:1659;251:1871;
256:1676;264:2552;267:868;269:410;276:2559;
309:1106;319:1414;320:1372;
347:1145;
392:1977;395:2007;408:2115;420:1851;431:1762;440:1646;
450:1911;454:1884;479:5102;
521:4947;529:4937;537:3582;540:3644;551:2917;554:3690;
564:616;587:3821;588:3829;633:4271;656,897:4205;
667:4491;675:4300;688:3082;
712:3068;727,728:3459;
764:2674;773:752;805:5485;807:5496;811:5440;
818:5526;839:1443
'''
parents={}
for clause in groups.replace('\n','').split(';'):
    if not clause.strip():continue
    indices,number=clause.strip().split(':')
    for item in indices.split(','):
        i=int(item);assert i not in parents
        parent='d'+number
        if i==839:assert parent=='d1443'
        elif i==440:assert parent=='d1646'
        else:assert any(parent in match[2] for match in candidates[i]['matches']),(i,parent)
        parents[i]=parent
articles={}
with gzip.open(DATA/'data/dedr/entry-texts-audit.csv.gz','rt') as fh:
    for article in csv.DictReader(fh):
        if article['Entry_ID'] in set(parents.values()):
            articles[article['Entry_ID']]={k:article[k] for k in ('Page','Citation','Raw_Article')}
            articles[article['Entry_ID']]['URL']='https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page='+article['Page']
assert set(articles)==set(parents.values())

accepted=[]
for i,parent in sorted(parents.items()):
    record=records[i-1]
    if i==839:form,gloss='kaṇ-/kāṇ-','to see, with kaḍ- stem and vow/divination extensions'
    elif i==440:form,gloss='kun','small'
    else:form,gloss,_,score=next(match for match in candidates[i]['matches'] if parent in match[2])
    full=articles[parent]['Raw_Article']
    evidence=f'DEDR {parent[1:]} full article (p. {articles[parent]["Page"]}) lists Kota {form} “{gloss}”. Wolf {record["Form"]} “{record["Gloss"]}” matches this specific form-and-sense family; the complete article is retained in primary evidence. The source wording and morphology are preserved.'
    if i==249:evidence+=' The same article also lists Kota guṛg- “to thunder” and Toda kuḍx- “to rumble”; this is an expressive sound family, not a precise deeper reconstruction.'
    if i==537:evidence+=' The article’s Kota naṛv- “to walk, proceed, happen” and naṛty “behaviour” support a conduct-family reading of this extended noun.'
    if i==839:evidence+=' Wolf also records conjugational stem kaḍ-, matching the DEDR Kota kaṇ-/kāṇ- (kaḍ-) paradigm; the article includes kaṇky, a payment of a vow to a god.'
    accepted.append({'record':record,'parent':parent,'kind':'reflex','citation':record['Source']+';dedr[entry '+parent[1:]+']','evidence':evidence,'comparison':'Homepage exact-form discovery and full primary DEDR article checked; family assignment pending review.','status':'pending-review','surveyIndex':i})

special={
    8:'Aḍmuṛtd is an extended bowing expression; DEDR 5123 cites aḍ mug- but refers its first component to another article, so the full construction needs component analysis.',
    20:'Aṇ- “pluck” only matches the fruit noun paṇ in DEDR 4004 by spelling; no verb family is secured.',
    35:'Arn “dried dung” is not saffron arcn in DEDR 220.',
    38:'Argal “hail” is not yet securely reconciled with DEDR 355 Kota ān-gal, whose first component is part of a compound hailstone expression.',
    66:'Āl “cowshed” is not aḷ “belly” in DEDR 294.',
    67:'The source calls āḷ “ruin” a Badaga borrowing; a direct reflex assignment to DEDR 4110 would bypass the immediate donor.',
    73:'Āpnīr “mixed dung and water” contains nīr “water” but its first component is unanalysed.',
    80:'Āṭm is glossed “dance” by Wolf, whereas DEDR 347 gives Kota āṭm as a turn in a game and āṭ as dance; the record may conflate two related forms.',
    111:'Calv- “become bored” does not match the money-for-expenses sense in DEDR 2781.',
    122:'Cark is glossed both “spices” and “country brew”; DEDR 2353 establishes the spice sense but does not resolve the second sense in this one source record.',
    136:'Ceḷ nīr is compound-like; the compared DEDR 5517 hot-water expression has a different modifier.',
    189:'Diṭic- “create” is not the “feed by hand” stem in DEDR 3263.',
    254:'Gurykaṭ occurs in DEDR 1786 only as a questioned Kota comparison, and the article explicitly assigns its -kaṭ component elsewhere; the full compound remains unanalysed.',
    302:'Kacv “loin cloth” has a lexical graph candidate but no compiled DEDR parent in the candidate list.',
    323:'Kan “virgin” is not red-hot kan in DEDR 1406.',
    324:'Kaṇ “section of koḷ” is not red-hot kan in DEDR 1406.',
    327:'Kaṇkaty “sickle” resembles the compound kaṇkīt in DEDR 1204, but its first component is separately referred to DEDR 1166 and needs analysis.',
    342:'Karpōy is discussed in DEDR 4112, but that article explicitly questions its second component; the pen-and-cowshed compound cannot yet be assigned as one reflex.',
    354:'Kavāl “ground in front of house” occurs in DEDR 5354, but the article questions its ka- component; the compound requires component analysis.',
    360:'Kaykōṭ is likely a compound; DEDR 2064 marks the hoe form as uncertain and refers its first component to another article.',
    368:'Kāl “area around a tree” may involve the base-of-tree sense in DEDR 1479, but the areal extension is not documented in that article.',
    375:'Kāṇ- is supplied only as a cross-reference to kaṇ-; this source record lacks a separately stated sense for verifying DEDR 1443.',
    384:'Kebgal “flint” occurs in DEDR 1931, a red-colour article, but gal “stone” is a separate unanalyzed component of this compound.',
    417:'Korgōṛ “damaged wall” contains a defect stem but the DEDR 1851 comparison is a defective horn expression; its components need separate analysis.',
    461:'Mand “funeral-ground area” may reflect a Toda settlement term, but the immediate route and referent of DEDR 4777 are unverified.',
    462:'Mand “Toda hamlet” may be a contact ethnographic term; the immediate donor route of DEDR 4777 remains unverified.',
    695:'Talār “area above” contains the tal “head/top” family of DEDR 3103, but its locative extension is unanalysed.',
    732:'Tinvāl “verandah” occurs in DEDR 3227, but the article expressly sends its vāḷ component to DEDR 5354; the full compound needs an ordered-component analysis.',
    742:'Ton “Toda man” is an ethnonym; borrowing and inherited-family status in DEDR 3504 are unresolved.',
    883:'Talār “above” contains the tal “head/top” family of DEDR 3103, but its locative extension is unanalysed.',
    602:'Pap “dust” is not “empty” pā in DEDR 4110.',
    762:'Uṇ- “start, set out” is not the pronoun ūn in DEDR 557.',
    889:'Kāl “area around a tree” may involve the base-of-tree sense in DEDR 1479, but the areal extension is not documented in that article.',
}
held=[]
for i,(record,result) in enumerate(zip(records,search),1):
    if i in parents:continue
    label=f'{record["Form"]} “{record["Gloss"]}”'
    if i in special:reason=special[i]
    elif ';' in record['Gloss'] or '?' in record['Gloss']:reason=f'{label} has a merged or qualified source gloss; no single article is secured.'
    elif ' ' in record['Form'] or len(record['Form'])>18:reason=f'{label} is multi-part or morphologically extended; ordered components remain unverified.'
    else:reason=f'{label} lacks a verified matching form and sense in a complete DEDR article at this checkpoint.'
    held.append({'record':record,'reason':reason,'status':'research-hold','surveyIndex':i,'searchResultCount':result['count']})
assert len(accepted)+len(held)==907
assert len({x['record']['ID'] for x in accepted+held})==907
def write(name,obj):(P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
write('proposed-decisions.json',{'status':'pending-review','accepted':accepted,'held':held})
write('research-holds.json',held);write('unexamined-inventory.json',[]);write('primary-evidence.json',articles)
print(json.dumps({'proposed':len(accepted),'held':len(held),'articles':len(articles)}))
