"""Review-only proposals from exact Bhat form/gloss and full DEDR Koraga articles."""
import csv, gzip, json, re, unicodedata
from pathlib import Path
P=Path(__file__).resolve().parent
DATA=P.parents[2]
records=json.loads((P/'survey-inventory.json').read_text())
assert len(records)==1369
search=[]
for start in range(0,1369,35):
    end=min(start+35,1369)
    search.extend(json.loads((P/f'discovery-{start+1:04d}-{end:04d}-results.json').read_text()))
assert len(search)==1369
assert all(result['opts']['languageId']=='Koraga' and result['opts']['params']['form']==record['Form']
           for record,result in zip(records,search))
direct={x['index']:x for x in json.loads((P/'direct-koraga-article-candidates.json').read_text())}

# Each indexed source gloss was individually compared with its Kor. passage.
# False friends such as aḍe “there”/ade “search”, kōru “give”/korru
# “kill”, and būru “sleep”/būru “fall” remain holds below.
groups='''
34:314;59:300;71:278;86:392;90,91:772;120:710;122:604;
130:636;138:652;145:751;146:687;162:851;173,174,175:3758;
182,183:839;184:835;209:562;210:933;219,220,221,222:946;225:1015;
245,246:988;247,248,249:760;271,272:1109;295:1211;307:1326;311,312:1249;
322:1278;331:1311;346:1099;353:1434;
354:1514;356:1594;358,359:1603;360:1616;363:1571;
372:1859;373,374:1851;376:1681;384,385,386,387:1873;
391:1892;396:1253;405:1953;406:1977;410:1574;411:1573;
414:1586;432,433,434:1863;435:2153;436:2128;439:2089;441:1731;
444,445,446,450,451,452:2149;447,448,449:2132;455,456,457:1829;
473:1223;478:1556;480:1616;492:1852;497:1355;498:1972;
503:2019;505:1660;506:1847;517:170;521:2996;524:2399;530:3393;
532,533,534:2548;536:2569;539:3273;540:2618;554,555,556:2865;
559:2838;567:2780;575,576:516;580:3682;581:2664;582:2621;
590:2883;596:88;597,598:3015;599:3025;601:3036;
605:3509;614:3435;615:2408;618:2399;626:3193;638:2599;
647:3307;650:3331;658:3406;662:3227;668,669:3436;
673:3480;678:3296;679:3370;681,682:3522;687:437;688:3020;
694:3192;709:3398;713:2901;724:2906;757:3726;
759,760:3697;768:3741;785:2907;812:3939;821,1285:4091;823:4135;
825,1290,1293:4423;827:4176;843:4354;847:4333;
854:4581;862:4509;871:5271;893,894:5292;896:5282;
895:5308;897,898,899:5305;908:5381;918:5379;920,921:5440;922,923:5517;
927:5415;929:5524;947,948:5430;958:5321;
960,961:5409;978,979:5437;982:4321;995:4626;997,998:4628;
999:5101;1007,1008,1009:4682;1042,1043,1046:4715;
1062:4794;1070,1071:5113;1099:5044;1102:4893;
1107:5034;1110,1111:5054;1168:2770;1205,1219:2704;
1209:2654;1227,1228:22;1232:3956;1235:4027;
1261:3999;1280:4121;1298:4157;1303,1304:4307;
1314,1315:4335;1323:4373;1346:4460
'''
parents={}
for clause in groups.replace('\n','').split(';'):
    if not clause.strip():continue
    indices,number=clause.strip().split(':')
    for item in indices.split(','):
        i=int(item);assert i not in parents
        parents[i]='d'+number

# Additional same-form, same-sense Koraga comparanda from Lindgren's separate
# source record; each family is checked against its full DEDR article below.
supplement='''
41,42,43:156;47:183;48:2826;51,52,53:215;65,68,69:1;
66:246;85,164,187,188,250,264,1143:5151;102:636;
112:2598;119:2644;128:2674;132:687;153:5154;161:848;
195,196,779,1141:5160;202:910;212:648;213,214,239:990;
216:946;252:698;261:1041;268,269,270:1147;285,286:1159;
288:1204;294:1355;309,310:2023;323:1278;342:1438;
348,349:1479;367,368:1671;412:1573;415,416,417:2017;
460:1873;461:2200;464,465:2053;476:1670;493:1253;
516,1172:2673;518,519:2426;546:2735;591,592:2872;
607,608,609:3098;611:3114;621:3170;625:364;627:3178;
634,635,636:3263;645,1195:3268;652,653:3266;655,1198:3319;
664,665:3419;671:3458;676:3302;686:3559;696:3178;
703,704:3230;714:63;726:81;742:3655;743:3633;
754,755:3690;756:3688;806:4148;814,1265:3999;
818,1278,1279:4063;830,1307,1347:4240;831:4172;841:4345;
850:4205;860,864:4517;861,1313:4410;867:4395;868:4544;
870:4572;877:5259;888,889:5270;911,912:5352;
931:5450;933:5450;936:5393;941:5433;953:5485;
959:5488;968,987:5528;972:5535;980:5496;
1002,1052,1053:4797;1011,1012,1131:4666;1034:4711;
1038,1039,1040:4753;1047:4990;1050:4729;1072:4841;
1076,1119:4985;1077:4878;1100:5024;1103,1104,1105:5052;
1125:4939;1147:474;1153:509;1202:2677;1218:2855;
1231:3821;1243,1244:4004;1256,1359:4316;
1283,1284:4020;1291:4137;1349:4494;1352:4559
'''
supplemental=set()
for clause in supplement.replace('\n','').split(';'):
    if not clause.strip():continue
    indices,number=clause.strip().split(':')
    for item in indices.split(','):
        i=int(item);assert i not in parents,i
        parents[i]='d'+number;supplemental.add(i)
def source_key(record):
    match=re.search(r'p\.\s*(\d+),\s*entry\s*([0-9a-zA-Z.-]+)',record['Source'])
    return match.group(1,2) if match else None
for i,parent in parents.items():
    if i in supplemental:continue
    if i in direct and any(h['parent']==parent for h in direct[i]['hits']):continue
    assert any(j!=i and p==parent and j in direct and source_key(records[j-1])==source_key(records[i-1])
               and records[j-1]['Gloss']==records[i-1]['Gloss'] for j,p in parents.items()),i

articles={}
with gzip.open(DATA/'data/dedr/entry-texts-audit.csv.gz','rt') as fh:
    for article in csv.DictReader(fh):
        if article['Entry_ID'] in set(parents.values()) and (article['Entry_ID']!='d5282' or 'DED(N) 4322' in article['Citation']):
            articles[article['Entry_ID']]={k:article[k] for k in ('Page','Citation','Raw_Article')}
            articles[article['Entry_ID']]['URL']='https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page='+article['Page']
assert set(articles)==set(parents.values())

def normalized(value):
    return ''.join(c for c in unicodedata.normalize('NFKD',value.lower())
                   if c.isalpha() and not unicodedata.combining(c))
lindgren=json.loads((P.parent/'koraga-lindgren-20260925/proposed-decisions.json').read_text())['accepted']
lindgren_by_form={}
for item in lindgren:
    lindgren_by_form.setdefault((normalized(item['record']['Form']),item['parent']),[]).append(item)
assert all((normalized(records[i-1]['Form']),parents[i]) in lindgren_by_form for i in supplemental)

accepted=[]
for i,parent in sorted(parents.items()):
    record=records[i-1]
    if i in supplemental:
        comparison=lindgren_by_form[(normalized(record['Form']),parent)][0]['record']
        full=articles[parent]['Raw_Article']
        excerpt=(re.findall(r'Kor\..{0,190}',full) or [full[:190]])[0]
        evidence=f'DEDR {parent[1:]} full article (p. {articles[parent]["Page"]}) gives {excerpt.strip()} Bhat {record["Form"]} “{record["Gloss"]}” matches Lindgren Koraga {comparison["Form"]} “{comparison["Gloss"]}” in form and sense; this cross-source match is a candidate lead, with the full DEDR article defining the family. Both source identities and dialect tags are preserved.'
    else:
        anchor_i=i if i in direct and any(h['parent']==parent for h in direct[i]['hits']) else next(j for j,p in parents.items() if j in direct and p==parent and source_key(records[j-1])==source_key(record) and records[j-1]['Gloss']==record['Gloss'])
        segment=next(h['segment'] for h in direct[anchor_i]['hits'] if h['parent']==parent)
        relation='the same cited Koraga form and sense' if anchor_i==i else f'a same-entry, same-sense dialect variant of {records[anchor_i-1]["Form"]}'
        evidence=f'DEDR {parent[1:]} full article (p. {articles[parent]["Page"]}) gives {segment.strip()} Bhat {record["Form"]} “{record["Gloss"]}” is {relation}; its original {record["Tags"]} dialect tag is retained. The article is used as family classification, not an independent second attestation of Bhat.'
    if i==896:evidence+=' The duplicate audit display number 5282 was resolved using the compiled d5282 node citation DED(N) 4322 (p. 477), which is the vali “pull” article.'
    accepted.append({'record':record,'parent':parent,'kind':'reflex','citation':record['Source']+';dedr[entry '+parent[1:]+']','evidence':evidence,'comparison':'Homepage exact-form discovery and full primary DEDR article checked.','status':'pending-review','surveyIndex':i})

special={
    26:'Aḍe “there” is a homograph of Koraga ade “to search” in DEDR 314; semantic mismatch.',
    124:'Ude “to cover” conflicts with the same-form “to break” passage in DEDR 946.',
    273:'Kaḍi “half of a coconut” is homographic with “send” in DEDR 1109.',
    274:'Kaḍi “to grind” is homographic with “send” in DEDR 1109.',
    332:'Kali “to shout like a demon” is homographic with “to open” in DEDR 1311.',
    333:'Kali “coconut” is homographic with “to open” in DEDR 1311.',
    338:'Koraga kaḷe “remove” is cited in both DEDR 1356 and 1373 with different broader family analyses; this source gloss does not select one.',
    339:'Koraga kale “remove” is cited in both DEDR 1356 and 1373 with different broader family analyses; this source gloss does not select one.',
    403:'Kedu “tank” does not match the rice-pot sense in DEDR 2015.',
    462:'Kōṇi “room” differs in sense from koṇi “dance” in DEDR 1863.',
    464:'Kōru “give” is a homograph of Koraga koru “kill” in DEDR 2132.',
    465:'Kōru “give” is a homograph of Koraga koru “kill” in DEDR 2132.',
    577:'Jāvu “speak” differs from the “descend” sense in DEDR 516.',
    639:'Tippi “coconut shell” differs from “sweep” in DEDR 2599.',
    667:'Teli “become clear” differs from “laugh” in DEDR 3436.',
    766:'Negi “say” differs from “lift” in DEDR 3730.',
    915:'Bāḷa “plantain” differs from “fish” in DEDR 5379.',
    917:'Bāḷa “very much” differs from “fish” in DEDR 5379.',
    945:'Būru “lie down, sleep” differs from “fall” in DEDR 5430.',
    946:'Būru “creeper” differs from “fall” in DEDR 5430.',
    989:'Bōḷa “lie down” differs from the cited “float” passage in DEDR 4321.',
    1001:'Maḍe “polluted food” does not fit the sūḍi “torch” passage in DEDR 2654.',
    1034:'Mara “tree” occurs within a compound in DEDR 3988, but this occurrence alone is insufficient to select the free stem family.',
    1051:'Maḷkaṇi “upside up” contradicts the DEDR 4754 “upside down” gloss.',
    1175:'Say “that is all!” does not fit Koraga ojji “say” in DEDR 648.',
    1178:'Sāḍa “fish” does not fit Koraga sada “rubbish” in DEDR 2770.',
    1231:'Haji “fresh, green” differs from “search” in DEDR 3956.',
}
held=[]
for i,(record,result) in enumerate(zip(records,search),1):
    if i in parents:continue
    label=f'{record["Form"]} “{record["Gloss"]}”'
    if i in special:reason=special[i]
    elif ';' in record['Gloss'] or '?' in record['Gloss']:reason=f'{label} has merged or uncertain source senses; a precise article match is not yet secure.'
    elif ' ' in record['Form'] or len(record['Form'])>17:reason=f'{label} is morphologically extended or multi-part; ordered components need research.'
    else:reason=f'{label} has no verified form-and-sense match in a full DEDR article at this checkpoint.'
    held.append({'record':record,'reason':reason,'status':'research-hold','surveyIndex':i,'searchResultCount':result['count']})
assert len(accepted)+len(held)==1369
assert len({x['record']['ID'] for x in accepted+held})==1369
def write(name,obj):(P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
write('proposed-decisions.json',{'status':'pending-review','accepted':accepted,'held':held})
write('research-holds.json',held)
write('unexamined-inventory.json',[])
write('primary-evidence.json',articles)
print(json.dumps({'proposed':len(accepted),'held':len(held),'articles':len(articles)}))
