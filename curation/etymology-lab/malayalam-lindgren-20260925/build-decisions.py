"""Conservative Lindgren Malayalam article-level review draft; no graph writes."""
import collections,csv,gzip,json,re
from pathlib import Path

P=Path(__file__).resolve().parent; R=P.parents[2]
records=json.loads((P/'survey-inventory.json').read_text()); assert len(records)==680
ranked=json.loads((P/'homepage-candidates.json').read_text())
tamil=json.loads((P.parent/'tamil-lindgren-20260925/proposed-decisions.json').read_text())['accepted']
by_cognate=collections.defaultdict(set)
for item in tamil: by_cognate[item['record']['Cognateset']].add(item['parent'])

# The shared survey cognate identifier is corroboration, not sufficient evidence.
# These entries have mismatched sense, contact vocabulary, or uncertain composition.
exclude={67,114,115,140,152,194,209,234,245,273,331,333,334,446,448,468,489,599,600,607,651}
mapping={}
for ix,rec in enumerate(records,1):
    ps=by_cognate.get(rec['Cognateset'])
    if ps and len(ps)==1 and ix not in exclude: mapping[ix]=next(iter(ps))

# Add only form-and-sense concordant homepage candidates. The selected full DEDR
# articles are retained below, and their Malayalam portions were checked manually.
homepage={30:4491,166:4452,217:2897,253:3774,295:3758,363:2883,364:2883,
372:4985,374:4099,384:2677,385:2677,386:2677,389:1670,422:3870,
425:1977,427:3676,431:3263,432:3263,449:3393,459:533,495:4921,
500:2017,510:5151,530:5304,534:651,538:5304,540:4460,550:234,
603:1571,651:1832,656:5503,679:3758}
for ix,num in homepage.items():
    pid='d'+str(num)
    assert ix not in mapping or mapping[ix]==pid,(ix,mapping[ix],pid)
    mapping[ix]=pid

# Additional direct lexical comparisons in the whole articles. Inflected verb
# forms are included only where the article gives the same base and sense.
manual='''1,2:3684;29:3876;32,33:1;34,35:410;49:5028;63:1249;66:2883;68:4955;70:529;72,75:811;81:4616;96:1957;99,423,424:1654;105:3263;121:2812;139:4004;143,294,471:4395;155:1652;156:4921;159:2017;160,162:410;169:7;171,172:5151;177:314;183:3987;185:5423;192:295;197:1025;200:990;211:5513;213:504;220:1254;222:1369;225:3679;228:4031;229:4031;231:2855;232:2855;236:2485;238:4004;239:4085;241:3323;246:2127;247:3302;248:1479;250:5388;251:1443;252:1443;254:4939;255:3918;257:1;259:1;260:3302;265:3774;267:3774;268:3633;269:3633;274:474;275:474;280:3647;281:3647;282:5160;283:5160;284:5160;285:5151;288:5151;289:5151;290:5151;291:2591;293:4395;301:410;302:1;318:1;319:410;336,337:77;341:2063;343:3105;345:472;352,353:1927;355:2456;356:2360;357:5299;363,364:2883;368:1048;375,376:811;377,378:1278;379:851;380,381,382,383:4366;387,388:2677;395:5388;402,403,404,405,406:5549;407,408:1291;413:1583;414:2091;416:1727;417:5478;427:3676;429,430:3263;436:385;437:360;451:1571;456,457:529;464:1479;467:4004;476,477,478,479:3098;480,481:4572;483:2781;484,485,486:815;492:228;497:2023;498:1298;505:410;515,516:698;520,521:314;522:1562;528:399;555:5485;556:5315;559:399;562,563,564:4054;566:2270;568:2897;571:4318;575:3303;576:3302;577,578:3446;583,584,585:4031;586,587:1443;589:910;596:4352;597:3302;598:3025;602:1851;606:3323;609:3675;612:5346;621:1;623:1298;629:5052;631:851;632,633:1369;642:1079;645,646:384;652,653,654,655:3639;660,661:819;662,663,664:854;665:2721;666,667,668,669:714;670:5151;676:1486'''
for clause in manual.split(';'):
    inds,num=clause.split(':');pid='d'+num
    for s in inds.split(','):
        ix=int(s)
        if ix in mapping and mapping[ix]!=pid:
            raise ValueError((ix,mapping[ix],pid))
        mapping[ix]=pid

# A second full-article pass rejects attractive but wrong short-form matches.
# Keep these records as explicit holds until the actual lexeme or donor is clear.
reject={29,49,68,70,96,109,121,143,155,183,185,192,197,211,213,220,222,225,227,228,
        231,232,236,238,239,241,246,247,248,250,251,252,254,255,259,260,
        265,267,268,269,274,275,282,283,284,288,289,290,291,293,
        357,368,380,381,382,383,395,402,403,404,405,406,413,416,
        429,430,436,437,484,485,486,515,525,555,556,559,562,563,564,566,575,576,577,578,
        582,597,602,612,623,631,632,633,645,646,652,653,654,655,
        660,661,662,663,664,666,667,668,669,676}
rejected={ix:mapping[ix] for ix in reject if ix in mapping}
for ix in reject:mapping.pop(ix,None)
mapping[200]='d234'
mapping.update({71:'d1278',351:'d2456',393:'d3844',416:'d1719',582:'d4666'})

articles={}; target=set(mapping.values())
with gzip.open(R/'data/dedr/entry-texts-audit.csv.gz','rt') as fh:
    for row in csv.DictReader(fh):
        if row['Entry_ID'] in target:
            articles[row['Entry_ID']]={k:row[k] for k in ('Page','Citation','Raw_Article')}
            articles[row['Entry_ID']]['URL']='https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page='+row['Page']
assert set(articles)==target,sorted(target-set(articles))

def ma_excerpt(pid):
    s=articles[pid]['Raw_Article']; m=re.search(r'\bMa\.\s',s)
    if not m:return None
    tail=s[m.end():];end=re.search(r'\s(?:Ir|Ko|To|Ka|Koḍ|Tu|Te|Kol|Nk|Pa|Ga|Go|Konḍa|Kui|Kuwi|Br)\.\s',tail)
    if end:tail=tail[:end.start()]
    return tail.strip()

missing_ma=[(ix,rec['Form'],rec['Gloss'],pid) for ix,rec in enumerate(records,1) if (pid:=mapping.get(ix)) and not ma_excerpt(pid)]
if missing_ma:print('No Malayalam subsection:',missing_ma)

def snippet(pid,form,gloss):
    s=ma_excerpt(pid)
    if not s:s=articles[pid]['Raw_Article']
    if len(s)>255:s=s[:255].rsplit(' ',1)[0]+'…'
    return s

accepted=[];held=[]
special_holds={
    29:'Betel-leaf crop vocabulary has no verified inherited article or immediate regional donor for this Byari form.',
    61:'pakki bird is a pakṣi-type contact form; the immediate donor into Byari is unresolved.',
    114:'abba father is an areal kinship form; the immediate Byari donor cannot be inferred from the Tamil appa cognate set.',
    115:'bāpa father is an areal bāp-type kinship form; its immediate donor is unresolved.',
    143:'belaŋŋɨ give is not the peṇ woman family; its actual giving stem needs identification.',
    185:'sumārɨ many resembles regional sumār vocabulary; DEDR 5423 is semantically unrelated and the donor is unresolved.',
    197:'mōndi night has no relationship to DEDR 1025 nine; the actual night lexeme is unresolved.',
    247:'naksatra star is nakṣatra-type contact vocabulary; no immediate Byari donor is established.',
    267:'samaya time is shared contact vocabulary and is not the nēram family of DEDR 3774.',
    273:'janḍɨ two may reflect Kannada-type contact phonology, but its immediate ancestry is not secure from the survey alone.',
    317:'veɻɻila betel leaf is a compound and its leaf element alone does not establish the whole etymology.',
    358:'pakṣi bird is Sanskrit-type vocabulary; the immediate Malayalam donor path is unresolved.',
    362:'rakt̪am blood is rakta-type contact vocabulary; DEDR 2883 cōra is a distinct family.',
    365:'raktam blood is rakta-type contact vocabulary; the immediate donor is unresolved.',
    447:'pitāvə father is pitṛ-type contact vocabulary; the immediate donor is unresolved.',
    468:'phalam fruit is phala-type contact vocabulary and does not belong to the inherited paɻam family.',
    484:'eraŋŋi descend was compared with DEDR 815 during screening, but that article is an ox family; its true verb family is unverified.',
    485:'iɻaŋŋ descend was compared with DEDR 815 during screening, but that article is an ox family; its true verb family is unverified.',
    486:'iraŋŋi descend was compared with DEDR 815 during screening, but that article is an ox family; its true verb family is unverified.',
    496:'rōmam hair is roman-type contact vocabulary; the immediate donor is unresolved.',
    502:'hrud̪eyam heart is hṛdaya-type contact vocabulary; the immediate donor is unresolved.',
    503:'manas heart is manas-type contact vocabulary with a semantic extension; immediate donor is unresolved.',
    525:'karaḷə liver cannot be assigned to DEDR 1298 stone through the shared Tamil cognate set; its own article remains unverified.',
    531:'kalyāṇam marriage is Sanskrit-type contact vocabulary; the immediate donor is unresolved.',
    532:'vivāham marriage is Sanskrit-type contact vocabulary; the immediate donor is unresolved.',
    547:'rāt̪ri night is Sanskrit-type contact vocabulary; the immediate donor is unresolved.',
    559:'puruṣan person is puruṣa-type contact vocabulary, unrelated to DEDR 399 āḷ person.',
    580:'tyāgəm sacrifice is tyāga-type contact vocabulary; the immediate donor is unresolved.',
    610:'nakṣatram star is Sanskrit-type contact vocabulary; the immediate donor is unresolved.',
    615:'sūryan sun is sūrya-type contact vocabulary; the immediate donor is unresolved.',
    616:'madʰuram sweet is madhura-type contact vocabulary; the immediate donor is unresolved.',
    643:'charddikk vomit is chardi-type contact vocabulary; the immediate donor is unresolved.',
}
for ix,rec in enumerate(records,1):
    pid=mapping.get(ix)
    if pid:
        dialect='Byari survey form' if ix<=298 else 'Malayalam survey form'
        evidence=f'The full DEDR {pid[1:]} article gives Malayalam {snippet(pid,rec["Form"],rec["Gloss"])}. The {dialect} {rec["Form"]} “{rec["Gloss"]}” is proposed within this article family.'
        if rec['Cognateset'] in by_cognate and pid in by_cognate[rec['Cognateset']]:
            evidence+=' Lindgren assigns the corresponding Tamil reflex to the same source cognate set.'
        accepted.append({'record':rec,'parent':pid,'kind':'reflex','citation':rec['Source']+f';dedr[entry {pid[1:]}]','evidence':evidence,'status':'pending-review','surveyIndex':ix})
    else:
        c=ranked[ix-1]['candidates']
        if ix in special_holds:reason=special_holds[ix]
        elif ix in rejected:reason=f'The DEDR {rejected[ix][1:]} candidate was checked in its full article, but the proposed form, sense, or dialectal pathway is not secure for this source record.'
        elif ix in exclude:
            other='/'.join(sorted(by_cognate.get(rec['Cognateset'],[])))
            reason=f'The shared survey cognate set ({other or "no matching Tamil proposal"}) is insufficient here: form, sense, formation, or immediate donor needs individual review.'
        elif rec['Gloss'].startswith(('third-person','second-person','we','you')):reason='The pronominal form, agreement category, or animate/honorific value is not independently verified against the full DEDR paradigm.'
        elif len(rec['Form'])>10 or ' ' in rec['Form']:reason='The elicited expression is complex; its full morphology or ordered component ancestry is unresolved.'
        elif c:reason=f'Exact-form homepage search returned possible comparanda (top {c[0]["parent"]}), but no full-article form-and-sense match was secure enough for a reflex proposal.'
        else:reason='Exact-form homepage search found no secure DEDR article with concordant form and sense; any contact donor or inherited family remains to be established.'
        held.append({'record':rec,'reason':reason,'status':'research-hold','surveyIndex':ix})

assert len(accepted)+len(held)==680
owners={x['record']['ID']:'proposed' for x in accepted};owners.update({x['record']['ID']:'held' for x in held});assert len(owners)==680
(P/'proposed-decisions.json').write_text(json.dumps({'status':'pending-review','accepted':accepted,'held':held},ensure_ascii=False,indent=2)+'\n')
(P/'research-holds.json').write_text(json.dumps(held,ensure_ascii=False,indent=2)+'\n')
(P/'unexamined-inventory.json').write_text('[]\n')
(P/'primary-evidence.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2)+'\n')
(P/'source-coverage-001.json').write_text(json.dumps({'source':'lindgren2023dravidian','language':'Malayalam','compiledSourceRecords':680,'proposedRecords':len(accepted),'heldRecords':len(held),'unexaminedRecords':0,'uniqueOwnership':True,'recordOwnership':owners},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'proposed':len(accepted),'held':len(held),'articles':len(articles)}))
