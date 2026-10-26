"""Original-first complete specimen76 initial page, retaining physical continuation."""
from stage_manual_page import stage

LINES='''
Ōkōn=One|mās=man(-to)|iddar=two|bālā=sons|aṇḍēr.=were.|Sinnām=The-younger|bālā=son|bāpam=the-father-to
muṭten,=said,|‘bā,=‘father,|jiṅgānī=property(-of)|anya=my|hissā=share|vāṭīp-kō,’=divide-give.’|Sinnām=The-younger
bālā=son|jiṅgānī=property|miḷāpten.=gathered,|dusrē=other|ūrn=village|dhān=far|sedden,=went,|udhaḷvārī=riotously|jiṅgānī=property
attēk=all|nās-gakten.=destroyed-made.|Dukaḷ=Famine|paṭṭīn,=arose,|paisā=money|kharsipen=to-spend|puṭṭed.=was-not-got.
Dōḍhā=Big|mās=man(-of)|avande=he|dhandā-takten.=service-lived.|Avan=He|turel=swine|mipen=to-feed|kēnut=field-in
kōstel.=sent.|Mārī=Then|turel=swine|poṭṭa=husks|tīniṅg=to-eat|uttel,=began,|avande,=he,|‘poṭṭa=‘husks|ān=I|tyalātūn=will-eat
ipāt.’=bread.’|Yēna-nā=Anybody|avandu=him-to|tāna=anything|siyaṅg-tōdī.=gave-not.|Avandun=Him-to|mārī=then|sūd=sense
vattīn,=came,|‘anye=‘my|bāpne=father’s|yallāt=house-in|naukarnā=servants-of|pōṭāṅg=belly-for|ipāt=bread|puṭtād.=is-got.|Ān=I
kāran=with-hunger|tiksātan.=die.|Ān=I|bāpam-phōkān=father-near|sāhādūn=having-gone|bāpam=father-to|moḍātūn,=will-say,
“bā,=“father,|inna=thy|samōr=before|diyām=God’s|hukum=command|puṭtān,=I-broke,|pāp=sin|aktān.=I-did.|Innen=Thy|bālā=son
inne-kādūn=having-said|anuṅg=me-to|lajjā=shame|vahā.=comes.|Anuṅg=Me|chākāudār-laṅgānu=servant-like|āp.”’=make.”’|Bāp-=Father-
phōkān=near|śedden.=he-went.|Bāp=The-father|keiken=the-son|ōlten=saw|dhāvād,=distance-from,|avadūn=him-to|mayā=pity
vattīn.=came.|Saddan=Having-gone|keiken=son’s|mākad=neck-on|paṭṭen,=fell,|avande=his|mukkā=kiss|samten.=took.|Mārī=Then
bālā=son|muṭten,=said,|‘bā,=‘father,|inna=thy|samōr=before|diyām=God’s|hukum=command|puṭtān,=I-broke,|pāp=sin|aktān.=I-did.
Innen=Thy|bālā=son|innekādūn=having-said|anuṅg=me-to|lajjā=shame|vahā.’=comes.’|Tāk=The-father|jharān=quickly|gaḍyākūn=servants
kōkten,=called,|gaḍyākūn=servants-to|iṭten,=said,|‘dhaḍāpan=‘good|āṅgē=cloth|ivande=this-of|mēnōt=body-on|tepp.=put.|Ivande=This-of
keit=hand-on|uṅgāram=ring|tapp,=put,|gēṭṭāt=foot-on|kērkul=shoes|tapp.’=put.’|Mārī=Then|avand=he|majjā=merry|ākī=to-make
'''

if __name__=='__main__':stage(568,76,LINES,'District Basim / so-called Bhili (approved base Naikri)',{
    11:'Final Bāp- is a physical line-break fragment continued by phōkān on line12; retain both physical keys, join only in reviewed atomic/expression derivative.',
    17:'Final clause continues onto569; do not insert an unprinted period or closing quotation.'})
