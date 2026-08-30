"""Cell-by-cell manual transcriptions from the raster word-list pages.

Each row was entered only after visual inspection of the named PDF page. Empty
strings are confirmed ruled blanks or explicitly absent/clipped cells.  The
three western columns retain the later digital transcription solely as a
comparison field; ``Forms`` is the manually checked source-image reading.
"""

R = "manual-source-image"


def row(page, printed, first, site, column, forms, *, uncertainties=None, digital=None):
    return {
        "PDF_Page": page,
        "Printed_Page": printed,
        "First_Item": first,
        "Site": site,
        "Column": column,
        "Review": R,
        "Forms": forms,
        "Uncertainties": uncertainties or [],
        "Alternate_Digital": digital or [],
    }


ROWS = [
    row(82,77,1,"JAG",1,["oleu","talakai","botʃu","mokam","kandu","kave","mosoɾ","pauɾu","palu","nalike","bʰomːa","dokːa","guda","kondakai","aɾikai","veliku","goɾu","kalu","tolu","bula"]),
    row(82,77,1,"CHI",2,["mendul","talakaia","kelːku","mokːam","kandu","kev","mosoɾi","pawiɾ","palːku","nalke","","dokːa","dʒabːa","kundawangai","labːa","tadugu","goɾu","kalu","tolːka","bula"]),
    row(82,77,1,"POD",3,["mendul","tala","kelku","mokom","kandku","kevuk","mosoɾ","paiur","palku","nalike","bomːa","dokːa","danda","kuggai","kailabe","vanstuku","gɾos","kalu","tolk","bula"]),
    row(83,78,21,"JAG",1,["gunde","netːuɾu","otsə","piu","naɾu","lonu","mutʃa","talapu","veɾike","epuɾu","ɾolu","uspal","sutːi","kaseɾ","godeli","tadu","daɾam/nol","sudi","guda","uŋgaɾam"]),
    row(83,78,21,"CHI",2,["gundikaia","netːuɾu","udukanadu","piː","naɾu","lonu","vetʃond","gʰumːam","veɾku","epuɾu","guntːa","utʃpal","sutːe","kaseɾ","gʰodel","tadu","nulu","sudi","gʰuda","uŋgaɾam"]),
    row(83,78,21,"POD",3,["gunde","netːuɾu","udukanadu","piu","naɾu","lonu","vesanadu","gumːam","veɾki","epuɾ","dʒatakal","uspal","sutːi","kasaɾi","godel","moɾsum","nul","sudi","tʃile","mudːa"]),
    row(84,79,41,"JAG",1,["podudu","nile","mabʰu","okːa","vanaŋ","eɾu","waŋgu","mabʰu","meɾupu","siŋgaɾi bili","gali","kalu","aɾi","uske","kisu","kumpeɾ/poggai","budidːe","buɾda","dubːa","baŋgaɾam"]),
    row(84,79,41,"CHI",2,["podudu","nelːe","mabʰu","ukːa","wana","eɾu","waŋgu","mabʰu","meɾsi","bimuduwulu","galdumːaɾam","kalu/ɾai","aɾu","utske/mandul","kisu","kumːpod","bʰude","bʰuɾde","dumːu","bʰaŋgaɾam"]),
    row(84,79,41,"POD",3,["podudu","leŋgu","mabʰu","ukːaku","wanːa","eɾu","sauɾa","mabʰu","meɾsi","biminwilː","gal","kal","aɾ","uske","kisu","umːa/pogːo","niɾ","buɾde","dundul","soːne"]),
    row(85,80,61,"JAG",1,["maɾa","aːki","veɾu","koɾːa","puŋgaɾi","pani","maɾke","aɾitʃi","goduma","dʒona","nukːa","alugadu","waŋkaia","veɾsinaga","miɾpakaia","pasupu","elipaia","wulipaia","podupoɾupu","tamata"]),
    row(85,80,61,"CHI",2,["maɾa","aːki","veɾu","koɾːi","puŋgaɾ","pandu","maɾkai","aɾatipandu","gʰoduma","dʒonːa","nukːa","bangal","waŋkaia","veɾutsenga","miɾpakaia","pasupu","velulipaia","niɾuli","","tamːate"]),
    row(85,80,61,"POD",3,["mada","aːku","veku","koɾːe","puŋgaɾ","pandi","maɾka","keda","goduma","dʒonːa","nukːa","bangalmati","hapːa","veɾtʃenaŋga","miɾiːak","kamːka","koɾulːi","kanda uli","kobːi","vaŋga"]),
    row(86,81,81,"JAG",1,["kabːadʒi","niːu","ouaɾi","siːa","kodauasi","kike","koɾupila","gaɾbʰam","godu","bare","palu","komːu","toka","meːka","naiu","pamu","koti","nulːe","petːe","bʰaludu"]),
    row(86,81,81,"CHI",2,["","niːu","ouaɾu","aŋgu/siːa","kodauas","kike","koɾu","gʰaɾbam","gʰodu","gede","palu","koku","toke","mekːa","niu","taɾs","kove","nulːe","alli/pete","bʰali"]),
    row(86,81,81,"POD",3,["kobːi","naiːu","ovʷaɾ","awuŋu","kodauosi","kike","koɾu","gaɾbam","godːu","bare","palu","kowuku","toka","mekːe","naiːu","taɾsu","kowi","nulːe","petːe","balu"]),
    row(87,82,101,"JAG",1,["pedeɾ","madusu","natodi","pilːa","iːla","iaua","aːna","tamːa","akːe","tʃelːe","maɾi","maiːadi","mutpal","mutːe","pekal","pakidi","ɾodʒu","naɾka","veɾie","paial"]),
    row(87,82,101,"CHI",2,["pedeɾ","manusond","natuad","pilːa","ia/tapːe","iawa","dʰada","tamːudu","iekːa","elad","maɾu","maiːad","mutpal","mutːe","pekːa","pekːi","paiːal/niːand","naɾka","veɾwe","madːanam"]),
    row(87,82,101,"POD",3,["pedːeɾ","manusun","natːad","pilːe","iamːal","iaio","dada","tamːud","iekːe","eːlad","maɾi","maiːad","mutːpal","mutːe","pekːal","pikːidi","dinːam/piːa","naɾke","naɾkom","madnam"]),
    row(88,83,121,"JAG",1,["mulpati","nine","niːadu","nadi","waɾam","nela","iadaɖi","patːa","kotːa","matʃa","tʃedːa","nanti","aɾte/watiatːe","podugu","kuɾsa","uduku/kasi","tʃali","tinːagai","ɾodagai","dagiɾe"]),
    row(88,83,121,"CHI",2,["mulpe","ninːe","niːandu","nadu","waɾam","nele","tʃamusaɾam","patːa","kote","meltːe","tʃedade","nantːe","watːe","podugu","kuɾsa","kasi","tʃali","tinːakai","edamakaia","dagːɾe"]),
    row(88,83,121,"POD",3,["mulpat","ninːe","niːandu","nade","waɾam","laŋgi","waɾse","panda","poːne","meltːadu","meltːadu iːo","nantːa","wastːa","poduːe","matːel","kank","tʃalːe","tinːakai","ɾodːaka","gaɾe"]),
    row(89,84,141,"JAG",1,["duɾam","pede","tʃinːe","baɾu","telika/alːaka","poːɾu","adugu/idupe","telːa","nalːa","eɾa","oɾoti","ɾendu","mudːu","naluku","idu","aːɾu","iedu","ienimidi","tomːidi","padi"]),
    row(89,84,141,"CHI",2,["duɾam","bʰeɾia","tʃinːa","bʰaɾu","telike","poɾo","idupe","telːam","nalːa","eɾa","oɾet","ɾendu","mundu","nalu","idu","aːɾu","edu","enimidi","tomːidi","padi"]),
    row(89,84,141,"POD",3,["duɾam","beɾiːa","tʃud","baɾu","telːke","poɾo","adigi","telːa","kaɾel","eɾa","oɾot","ɾendu","mundu","naːlu","eingu","aːɾu","satː","aːt","tomːidi/nai","das"]),
    row(90,85,161,"JAG",1,["padikondu","panendu","iɾauai","wanda","bano","bʰatadi","begːa","beske","betʃuku","bataɾakam","idu","adu","iwu","awu","oɾeteɾakam","veɾe veɾe","antːa","pagili","kodika","tʃena"]),
    row(90,85,161,"CHI",2,["padakondu","panːendu","iɾupai","nuɾu","bʰeno","batːa","begːa","betʃke","betʃotu","batːaɾakamu","idu","adu","iwu","awu","oke","veɾe veɾe","antːe","oɾtːa","utʃute/tʃudute","dibe"]),
    row(90,85,161,"POD",3,["paɾkondu","bʰaɾa","kode","wanda","benon","batːa","begːe","betʃot","betʃoɾ","betodnaɾ","idu","adu","iwu","awu","ondu putːi","veɾe veɾe","saɾe","paitːe","tʃukuk","dibːe"]),
    row(91,86,181,"JAG",1,["anːta","tinu","koɾuku","kuɾuɾesam","unːu","dopesonde","undʒa","patːa","kudːu","iːum","masa","dolːa","aukːa/sawadenga","tulːa","nadum","miɾa","anːude","wa","tiɾiːa","kenga"]),
    row(91,86,181,"CHI",2,["antːa","tin","katʃtːa","kaɾpoita","untanu","dapite","undʒa","patːa","kudːa","iːum","masa","dolite","awukta","tulte","nadaka","miɾa","anːu","waɾa","tiɾiːa","kenga"]),
    row(91,86,181,"POD",3,["tsaɾe","tinːu","katʃita","kaɾposond","un","eɾund sagwaɾte","undʒa","patːa","kudːa","ium","(kisu)masa","dolːto","awuka","langta","takita","miɾa","an","waɾa","tiɾka/wegʰitika","kenga"]),
]


def western(page, printed, first, site, column, forms):
    # The comparison transcription is deliberately duplicated, not substituted:
    # every Forms cell was checked visually against the 1985 scan first.
    return row(page, printed, first, site, column, forms, digital=forms)


ROWS += [
    western(92,87,1,"UTN",1,["mɛːnd̪ol","t̪ʌlʌʔ","kɛlk","mukʰʌmu","kʌdək","kɛːuʔ","mos̪əɾ","ʈoɖi","pʌlk","βɛndʒər","tsaˑt̪ʰi","pɛʈˑi","kai̯","ʈoŋɡi","t̪aɽkai̯","boʈʌ","t̪ʰɪdɪndʒ","kaˑl","ʈoˑl","bokˑʌ"]),
    western(92,87,1,"BHG",2,["mendʒul","t̪ala","kelˑku","mokam","kadu","kevu","mukˑu","t̪odˑi","palku","nalke","bʰomˑu","piru/bʰot̪a","d̪anda","kundakai","aɾikaiˑu","u̯elu","ɡʰoɾu","kalu","t̪olu","bʰokˑa"]),
    western(92,87,1,"BHM",3,["ment̪uli","t̪alə","kelːo","mokame","kanˑku","kevo","mosoɾe","pau̯aɡai","palku","nalka","bokˑu","bot̪ʰa","atæ","kunamkai","halˑa","aɖusku","idiŋɡ","kalku","t̪ol","bʰokˑe"]),
    western(93,88,21,"UTN",1,["bɔˑkʌ","nɛt̪u̯ru","uɽuku","peːk","naˑɾu","ɾɔːn","vɛːsvʌl","kaːvʌd","kʌʈːŋɡ","kai̯səɾ","tʃʰaki","ɾokʌl","sutːɛ","ɾuːs̪iʔ","maɾs̪","nɔːɖɛ","noːɭ","sou̯i","kʌpɖi","mudːʌ"]),
    western(93,88,21,"BHG",2,["ɡunde","net̪uɾu","utʃa","pelˑkodu","naɾu","lonu","","t̪alapu","kat̪e","epuɾu","dʒot̪ˑa","dʒot̪ˑa/bʰodu","sut̪i","kaseɾ","marsu","nonde","d̪aɾam","sud̪i","kapˑidi","uŋɡɾam"]),
    western(93,88,21,"BHM",3,["ɡunde","net̪ˑuɾu","udukuli","pelˑkumadi","naɣo","lonu","veʃmadi","t̪alupu","u̯eki","epˑuɣai","haɛta","usmali","sut̪ˑe","kasaɾ","maksu","nonde","nulu","sudi","bʰat̪a","mud̪ʰaʔ"]),
    western(94,89,41,"UTN",1,["suɾiə","tʃʌndɾʌ","aːbʌl","tʃukum","pir","ɛːru","nʌdi/lau̯di","tʌpʌʔ","mɪɾɪtʃʌnta","kʌmbaːɖ","vaɖi","bʌndʌ","səri","us̪kɛ","t̪ʌdɵ̯miʔ","poːi̯ʌ","niːɾu","tsɪklʌ","duɾdʌ","soːnɛʔ"]),
    western(94,89,41,"BHG",2,["podud̪u","u̯edisi","mabʰˑu","ukˑa","piɾu","eɾu","bʰeɾad","mabɖu","","","u̯adi","bʰanda","hʌɾ","uske","t̪adimi","poi̯a","bʰud̪i","bʰurd̪a","d̪ʰumˑu","baŋɡaɾam"]),
    western(94,89,41,"BHM",3,["poɖud̪u","nelˑa","mabʰu","ukˑa","peɡo","eɡu","bʰeɾˑdi","mabʰi","midisint̪e","bʰimanu̯ilˑli","vadiuˑ","bʰanda","ʌɡʰai","usˑka","kisu","onduli","bɦudt̪e","bʰuɾd̪e","d̪umˑu","bɦaŋɡɾam"]),
    western(95,90,61,"UTN",1,["mʌrʌ","aːkiʔ","seːr","tʃʰahk","buŋɡaːɾ","pʌɳɖi","mʌɾkʌ","kɛːɾɛŋɡ","ɡəhuk","dzonːʌ","pɛrɛk","aːlu","saːpʌŋ","bwuik","mɪɾtʃʌ","kʌmkʌ","lʌsn","ulːi","(pul)ɡɦobi","babɾɛ"]),
    western(95,90,61,"BHG",2,["mara","akˑi","ir","apˑu","put̪ˑa","pandi","mamidi","aɾat̪i","ɡʰod̪uma","dʒonˑa","nukˑa","alˑuɡʰada","u̯aŋ(kai̯a)","palˑi(kai̯a)","mirsa","pasupa","elˑipai̯a","ulˑi","(put̪ˑa)ɡʰobi","bʰed̪uɾu"]),
    western(95,90,61,"BHM",3,["maɾˑnu","akˑiˑ","iɾku","ʌpˑu","put̪ˑa","pani","makˑa","bʰaɡana(kai̯a)","ɡʰoduma","dʒonˑa","nukˑa","ʌluɡada","ʌpˑa(kai̯a)","bɦudʒalaŋɡa","miɾse","kamˑka","elpai̯","ul","ɡobi(ɡada)","bʰet̪ai(pani)"]),
    western(96,91,81,"UTN",1,["(paːnɡ)ɦobi","niˑu","soβər","sʌvɪŋ","koɖvɪnd̪ʒ","meːn","koɾ","mɛːsk","muːɾʌ","hɛrmiʔ","paːl","kohk","toːkr","hɛrːe","nɛiːu","tʌras̪","koːβɛ","d̪oima","pɛt̪ːɛŋ","kuduɾdokːɛ"]),
    western(96,91,81,"BHG",2,["(akˑi)ɡʰobi","nuiˑu","owoɾ","ʌviŋ","kodondʒi","dʒimˑa","koru","mesku","pelˑa","bare","palu","komˑukˑu","t̪okˑa","eti","naiˑu","t̪aras","kot̪i","t̪loma","pet̪e","bʰalˑi"]),
    western(96,91,81,"BHM",3,["","niuˑ","owaɾi","avaŋ","koduˑŋʒ","miŋkˑu","pisu","mesku","d̪uda","kond̪a","palu","koŋku","t̪eke","et̪i","naiu","t̪aɾsu","kou̯e","nulˑe","pet̪ˑe","kopˑanbaɡa"]),
    western(97,92,101,"UTN",1,["poroːl","mai̯nl","vɛloʔ","mod̪ːɛ","baβʌ","bʌiːɛ","daːdaʔ","tʌmuɾuʔ","baːi","sɛld","moɖʌl","mod̪ɛ","maːrsu","bai̯koʔ","kamdi","pɛːɖi","dɪn(amku)","nʌɾkʌ","sʌkɾɛ","d̪opaɖi"]),
    western(97,92,101,"BHG",2,["poˑɾoi","manikel","murt̪adaʔ","(tʃinˑu)poɾod","bʰalu","i̯awal","d̪ad̪al","t̪amˑal","ʌkˑai","tʃelˑe","mari","miˑadu","muid̪o","aŋɡe","mari","miˑado","ɾoʒu","narka(du)","akˑedu(ku)","piˑt̪el"]),
    western(97,92,101,"BHM",3,["ped̪iɾ","mankal","antʃadi","pila","t̪apˑe","au̯a","d̪ad̪al","t̪amˑa","iˑa","eladi","maɡʰai","miˑadi","muɾiˑo","mut̪o","pedal","pila","piˑali","nakˑa","nakˑami","pet̪epiˑali"]),
    western(98,93,121,"UTN",1,["sʌndaɾi","nɪnːe","nɛːnd","naːɖi","aʈʰɾodʒ","mahɪnʌ","saːl","pʌɖana","puːnaʔ","tʃokut","tʃʰokuʈsɛlːɛ/kaɾabaːt̪ʌ","pahana","βat̪ːa","laːm","tʃuduɾu","kaːsʈʌ","kɪnaː","t̪ɪnːa","d̪ɛmːa","karum"]),
    western(98,93,121,"BHG",2,["poduˑaŋɡi","nadi","ind̪ike","nadi","u̯aɾam","nele","vaɾusa","padana","poˑna","mantʃi","kaɾab","nant̪a(du)","u̯at̪ˑt̪a(d̪u)","saŋɡad","tʃinˑa","kasi","piˑni/salˑaɡa","t̪inˑa(pakˑa)","ɾodˑa(pakˑa)","d̪aɡʰiɾa"]),
    western(98,93,121,"BHM",3,["mulpe","ninˑe","nedu","nadi","u̯aɾam","nelˑla","ede","pant̪a","puna","mentʃda","melo","nant̪e","u̯at̪ˑʌ","dʒalu","udit̪a","kast̪e","iɡamˑt̪e","mut̪su","d̪emˑedʔ","d̪aɡˑɾe"]),
    western(99,94,141,"UTN",1,["lʋŋɡ","dʌɡur","tʃuduɾ","dʒaːdtʔ","hʌlkoʔ","porːo","buːɖ","dɦau̯ɾʌl","kaːɖi","ɾʌɡəl","undi","ɾendu","muːdu","naːluŋ","siːyuŋ","saːɾu","iɛːɖu","aːʈ","nawu","d̆aha"]),
    western(99,94,141,"BHG",2,["d̪uɾam","ped̪e","tʃinˑa","u̯adʒane","alkˑɡa","poro","idu","t̪ellad̪i","kared̪i","eradu","ond̪i","ɾndu","mund̪u","nalui","eiun","aɾun","ed̪un","enimid̪i","t̪omˑid̪i","pad̪i"]),
    western(99,94,141,"BHM",3,["dʒakˑu","bʰehiɾa","uˑdit̪a","puft̪e","ʌlkˑaɡa","poɡenu","idunu","t̪eldɡa","kaɾi","eɾˑaɡa","und̪i","ɾendu","mundu","nalu","eiˑũ","aɾũ","edũn","enimid̪i","t̪omˑidi","pad̪i"]),
    western(100,95,161,"UTN",1,["akɾa","baːɾa","βiːsʌ̯","nuːɾu","boːɾu","bʌtːʋl","baɡːʌ","bʌs̪kɛʔ","bʌtʃʰol","bʌt̪iɾd̪ʌ","ɪdɛʔ","ʌd̪uʔ","ɪvuʔ","ʌvuʔ","teːɾ","dusorʌ","ɡʌd̪si","uɾuŋtʌ","itʃuŋsi","vɛlːɛ"]),
    western(100,95,161,"BHG",2,["pad̪akondu","panˑendu","iruwai","noˑɾu","bʰor","bʰat̪ˑal","bʰekˑe","baske","bʰatʃun","bʰalˑeɡa","ind̪i","aɡˑa","u̯a","oɾu","vaɾa(ɾumanekel)","eɾe","bʰoŋɡa","uɾuɡʰi","d̪amˑu","u̯elle"]),
    western(100,95,161,"BHM",3,["pad̪akondu","panedu","iɾuwai","nuɾu","bʰoɡo","bʰat̪e","bʰeɡˑa","bʰeske","bʰtʃoku","bʰesonta","id̪u","ad̪a","ivˑu","avˑu","ont̪eɾu","eɣeeɣe","bʰoli","u̯iɾɣt̪e","utʃunu","bʰaɡa"]),
    western(101,96,181,"UTN",1,["sʌmd̪i","t̪ɪnβʌl","kʌskβʌl","karvasta","unβʌt","iɛːɾotkʌβʌstʌ","dʒop","ʌɾt̪on","udːʌ","siːm","βɛːnt̪ʌ","saːt̪ʌ","dʒau̯ɡtəɾ","dɛhk(βʌl)","taːk(βʌl)","vɪt(vʌl)","sonvʌ","vator/va","vaɾk(vʌl)","kɛndʒdvʌl"]),
    western(101,96,181,"BHG",2,["bʰot̪iɡa","t̪inˑudu","korkodu","pod̪uɾu","unˑudu","d̪upa","minˑdu","minˑdudu","ud̪u(du)","iˑu(du)","bʰodtʃu(du)","hai(madʔ)","au̯k(madʔ)","deina","t̪akna","u̯it̪(modʔ)","onmadʔ","u̯ai","u̯aduk(madʔ)","kendʒmadʔ"]),
    western(101,96,181,"BHM",3,["bʰot̪iɡa","t̪inˑu","kasˑkint̪e","kau̯soɾ","uˑmadi","eɣondeaseka","undʒa","(edne)aɾt̪o","eiˑmadi","i(madi)","u̯ai(madi)","ɖolt̪a","au̯k(madi)","dei(madi)","t̪akˑ(madi)","vit̪ˑ(madi)","amˑadi","(iɡa)va","vadoaŋka","kandʒa"]),
    western(102,97,201,"UTN",1,["soːɖ(vʌl)","nʌnːʌ","nimːɛ","meɾʌt","voːɾu","ʌd̪ːu","nʌmʌt̪u","mʌmaku̯","nɪmet̪u","βoːru"]),
    western(102,97,201,"BHG",2,["ud(madʔ)","nanˑa","nimˑu","nimˑetu","oɾu","ad̪u","mant̪oto","","","oɾu"]),
    western(102,97,201,"BHM",3,["od(madi)","nanˑa","nimˑu","miɾu","veɣʰo","ad̪u","manˑala","mamu","miɾu","oɾu"]),
]

def flags(length, mapping):
    values = [""] * length
    for offset, note in mapping.items():
        values[offset] = note
    return values


# Telugu and Oriya are source-internal comparison controls.  They are excluded
# from installation but transcribed here so the audit accounts for every cell.
ROWS += [
    row(113,108,1,"TEL",1,["ʃeɾiːɾamu","tala","dʒutːu","mukhamu","kanːu","tʃevi","mukku","noɾu","palːu","nalika","tʃanulu","kadupu/potːu","tʃeːi","mautʃei","aɾutʃei","velu","goɾu","kaːlu","tolːu","emuka"]),
    row(113,108,1,"ORI",2,["soɾiɾo","mundo","balo","muhõ","akhi","kano","nakho","patːi","danto","dʒipo","tʃati","petːo","hato","koini","tolohato","anguti","noːkʰo","gudo","tʃaɾmõ","haːdo"]),
    row(114,109,21,"TEL",1,["gunde/haɾudaiamu","ɾaktamu","otʃa","kaka","gɾaːmamu","ilːu","paikapːu","talupu","katelu","tʃipːu","ɾolu","patːɾamu","sutːi","katːi","godeli","taːdu","daɾam","suːdi","batːa/goda","aŋgaɾamu"]),
    row(114,109,21,"ORI",2,["haɾudaiɔ","ɾakto","mutːo","dʒaɾa","gɾaːmõ","gʰoɾo","tʃʰaːto","qobato","kaːto","tʃantʃoni","kotːuni","potʰoɾo","","qati","taŋgiːa","dowudi","suːtːa","sũntʃi","luːga","mudi"],uncertainties=flags(20,{12:"confirmed ruled blank"})),
    row(115,110,41,"TEL",1,["suɾiudu","tʃandɾudu","aːkaʃamu","tʃukːal","wanːa","nilu","nadːi","megʰamulu","meɾupu","indɾadhanusu","gaːli","ɾai","daɾi","isuka","nippu/manta","pogːa","budida","boɾada/banda","dumu/duli","baŋgaɾamu"]),
    row(115,110,41,"ORI",2,["sudʒo","dʒanha","akaʃau","nakʃatɾa","boɾosa","pani","nadi","megʰo","bidʒuli","indɾo dʰanasɔ","dʒoɾaka","potoɾa","ɾastɾa/bato","bali","niːa","dʰumaɾo","tʃaɾo","kaːdo","dʰuli","sunːa"]),
    row(116,111,61,"TEL",1,["tʃetːu","aːku","veɾu","mulːu","puʃpamu/pumu","pandu","maːmidikaia","aɾatipandu","goduma","dʒona","taːdu/biːamu","alugada/baŋgaladumpa","vaŋkaia","veɾusinaŋga(-kai)","miɾtʃi","pasupu","veluli","wulipaia","kaliflawʷaɾ","takaali/tamata"]),
    row(116,111,61,"ORI",2,["gotʃo","potɾo","tʃeɾo","konta","pʰulo","pʰolo","ambo","qodoli","gohomõ","","tʃawulo","alu","baiŋgonõ","tʃinabadam","moɾtʃa","holidi","ɾosono","piadʒo","qobipʰulo","bilatːi"],uncertainties=flags(20,{9:"confirmed ruled blank"})),
    row(117,112,81,"TEL",1,["kabadi","nuɾne","upu","mãːmsam","kou","tʃepalu","kodi","gudːu","awu","geda","paːlu","kumːu","toka","meːka","kukːa","pamu","koti","domalu","tʃemalu","sita kokatʃilaka"]),
    row(117,112,81,"ORI",2,["kobi","telo","nũno","mantso","buːso","maːtʃo","kukuda","onda","gai","moiːʃa","kʰiɾo","","landʒo","tʃeli","quguɾa","sapo","","moʃa","matʃi",""],uncertainties=flags(20,{11:"confirmed ruled blank",16:"confirmed ruled blank",19:"confirmed ruled blank"})),
    row(118,113,101,"TEL",1,["peɾːa","maniʃi","stɾi","bida","tandɾi","tali/ama","anːa","tambudu","akːa","tʃelːe","kumaɾudu/kodaku","kumaɾte/kutuɾu","baɾːta","bhaɾia","baludu","balika","ɾoz/dinamu","ɾaːtɾi","podutːa/udaiamo","madiahnam"]),
    row(118,113,101,"ORI",2,["naːmɔ","moniʃo","stɾi","pilːa","bapa","ma","nõna","bʰai","nani/didi","bʰouni","puːo","dʒiːo","suami","bʰadʒa","puːo/pula","dʒio pula","dino","ɾatːi","sokalu","upoɾano"]),
    row(119,114,121,"TEL",1,["saiukalam","nina","iːɾodʒ","ɾeːpu","waɾam","nela","samʷastɾamu","paːta","kɾotːa","mantʃi","tʃedːa","tadi","pudi/endina","podawu","kuɾutʃa/poti","vedi","tʃali","kudi","iedama","dagaɾa"]),
    row(119,114,121,"ORI",2,["sontiːa","kali/gotodino","adʒi","asonta kali","sotːa","maso","boso","poɾuna","nuːã","bʰolo","kaɾapo","kontʃa","suka","lomba","tsotia","goɾamo","sito","dahano","baːmo","pakʰo"]),
    row(120,115,141,"TEL",1,["doɾam","pedːa","tʃinːa","baɾuwu","telika","paina","kiɾunda","telːa","nalːa","ieɾːa","okati","ɾendu","muːdu","naːlgu","aidu","aːɾu","ieːdu","ienimidi","tomːidi","padi"]),
    row(120,115,141,"ORI",2,["duɾo","boɾo","sanõ","bʰaɾi","usaso","upoɾo","tolo","dʰola","kola","ɾongo","eko","duːi","tini","tʃaɾi","pantʃo","sato","atʰo","","nao","daso"],uncertainties=flags(20,{17:"source annotates eight as atʰo between rows"})),
    row(121,116,161,"TEL",1,["padakondu","panːendu","iɾawai","wanːda/nuɾu","ievaɾu","iemitːi","iekada","iepudu","ini","iemi ɾakamu","idi","adi","ivi","avi","ɾakamu","veɾu veɾu","sandɾamu","viɾiganu/pagilenu","kodi/takua","tʃala/iekua"]),
    row(121,116,161,"ORI",2,["egaɾo","bʰaɾo","kodiːe","eko saho","kie","konõ","keuntaɾe","kebe","kete","keo","eitːa","seitːa","eisabu","seisabu","ei ɾakamo","bʰino bʰino","kõnã","baŋgila/patːilo","qom","bohut"]),
    row(122,117,181,"TEL",1,["ani/antːa","tinu","kaɾatʃutːa","aːkaliga onadi","tɾaguta","dapikaga onadi","nedɾa","padi pouta/pandu konuta","kutʃonuta","etːutʃuta","manduta/kaluta","tʃanipowuta","tsampouta","eːguɾuta","nadu tʃuta","paɾigettuta","velːuta","votʃutːa/ɾandi","mataladuta","vinːuta"]),
    row(122,117,181,"ORI",2,["sobu","kaiba","tsubaila/kamudila","bʰoko hela","piːba","soso hala","nido","poɾigola","bosiba","deba","dʒoliba/dʒolitʃi","moɾonõ","maɾiba","udutʃi","tʃaliba","doudiba","dʒiba","aso","kota kuho","suno"]),
    row(123,118,201,"TEL",1,["tʃotʃuta","nenːu","niwu","miɾu","itadu","aːmi","manamu","memu","miɾu","waɾu"]),
    row(123,118,201,"ORI",2,["dekʰo","mũ/tometu","","aponõ","se/tanke","se","ampe/manːe","ampe","aponõ manːe/tome manːe","se manːe"],uncertainties=flags(10,{2:"confirmed ruled blank"})),
]


ROWS += [
    row(103,98,1,"MAL",1,["","talːa","kelk","mokom","kandu","kevuk","mosoɾ","paluɾ","palk","naːlke","bomːa","dooka","danda","minda","labːa","vanisk","insk","kal","tolka","buːla"],uncertainties=flags(20,{0:"confirmed ruled blank"})),
    row(104,99,21,"MAL",1,["dʒiːva","netːuɾ","udukanɖeɾ","piː","naɾu","loːn","buːd/vestadu","gumam/kapːat","veɾke","epuɾ","aku/dʒatːa","uspal","motːal","katʃeɾ","goːdal","noːda","nul","suːd","tʃile","mudːa"],uncertainties=flags(20,{10:"first variant has an illegible medial source character"})),
    row(105,100,41,"MAL",1,["podud","nelːa","bondaɾ","uːkaːk","wanːa","ieːɾ","potːaɾ/gadːa","mabːu","midukuta","bʰimun wil","gaːl","kalboːde","aɾ","usko","kiːs","poga/pogo","niɾaik/kisis","mantːa","dumːu","baŋgaɾam"]),
    row(106,101,81,"MAL",1,["mutːa","nai","ovʷoɾ","aːwʷuŋ","koduwos","kiːke","koɾ","gaɾbam","godʰu","bare","paːl","koːk","tokːa","moke","naiu","naitːaɾs","mundʒu","nulːe","pudu/oɾŋu",""],uncertainties=flags(20,{19:"confirmed ruled blank"})),
    row(107,102,101,"MAL",1,["naːwa/pedeɾ","menːe","mutːe","pilːa","baːbal","iaːia","dada","tamud","akːa","tʃele","maɾ","maiːad","mutpal","mutːe","pekːal","piki","paiːal","ikaːd","naɾkom","madana"],uncertainties=flags(20,{10:"collector/source question mark",11:"collector/source question mark"})),
    row(108,103,121,"MAL",1,["mulpe","oːndin/ninːa","inge/neːdu/dinam","naːdu","","nelːgi","ieːlat","pantːe","pʰunad","melted","tʃaiːle","pahna","watːa","laːti","mond","kasted","kirinted","tinːe","rode","bokːet"],uncertainties=flags(20,{4:"confirmed ruled blank"})),
    row(109,104,141,"MAL",1,["duɾam","baɾːie","tʃudul","baɾubu","sudːul","peɾo","neːlan","veɾtakʰ","gudilː","ieɾa","ondi","ɾendu","muːndu","nalgu","aiːndʒ","aːɾu","ieːdu","aːt","tomːidi","das"]),
    row(110,105,161,"MAL",1,["","","kodi","soːie","beːɾon","batːa","begːa","bepoɾ","besoɾ","betːtːad","idːu","adːu","iwu","awu","studor","veɾa veɾa","","kunda/oɾtːa","itʃad","naɾge"],uncertainties=flags(20,{0:"confirmed ruled blank",1:"confirmed ruled blank",16:"confirmed ruled blank"})),
    row(111,106,181,"MAL",1,["saɾe","tin","katsa","kaɾuːa","un","","undʒ","patːa","kud","iːum","","dolt","wadsta","paɾsom","naːdta","miɾ","anːu","wada","",""],uncertainties=flags(20,{5:"confirmed ruled blank",10:"confirmed ruled blank",18:"confirmed ruled blank",19:"item 200 absent/clipped between source pages"})),
    row(112,107,201,"MAL",1,["uːda","nanːa","nimːa","nimːa","uːndu","dan/adː","mamːa","mam(-iɾuːam)","miɾ(-iɾuːiɾ)","uɾ(-iɾuːiɾ)"]),
]
