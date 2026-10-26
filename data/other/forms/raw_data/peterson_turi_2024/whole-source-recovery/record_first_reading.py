"""Native-layer assisted first reading; all records require original-page review."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
records=[]
def add(page,number,forms,glosses,translation,*,lect='Odisha',note='',kind='interlinear',raw=None):
    fs=forms.split('|'); gs=glosses.split('|')
    assert len(fs)==len(gs),(number,fs,gs)
    records.append(dict(source_unit_key=f'peterson2024turi:p{page}:example:{number}',printed_page=page,pdf_page=page-260,example=str(number),kind=kind,lect=lect,forms=fs,aligned_glosses=gs,translation=translation,raw_surface=raw or ' '.join(fs),note=note,review_status='first_reading_pending_original_review'))
add(298,44,'ɛlɑ|rɛ|ʧhɑʋɑ=kun|sɛn=ɑ=bu|iskul','come.imp|voc|child=pl|go=fin=1pl.incl|school','Come along, children! Let’s go to school!',note='Printed instruction: repeat once. Punctuation: comma after child, final full stop.')
add(298,45,'iskul=rɛ|ɑbu|gɔɳitɔ|pɑɽh=ɑ=bu','school=loc|1pl.incl|math|learn=fin=ipl.incl','Let’s learn math at school!',note='Printed instruction: repeat once. Printed interlinear has ipl.incl (letter i), retained without silently replacing with numeral 1.')
add(298,46,'ɑlɛ|itu-ɔ=ɑ=bu|lɛkəɛ=ɑ=bu','1pl.excl|learn-mid=fin=1pl.incl|count=fin=1pl.incl','Let’s learn, let’s count!',note='Source contrasts exclusive independent pronoun with inclusive verbal indices; do not harmonize.')
add(298,47,'miɑʔn|bɑrɛɑ|pɛɑ|ʧɑr=gɔɽ|pɑnʧ=gɔɽ|lɛkəɛ=ɑ=bu','one|two|three|four=count|five=count|count=fin=1pl.incl','Let’s count 1 – 2 – 3 – 4 – 5!')
add(298,48,'bɑndrɑ|ʈɔpi','monkey|hat','The monkeys’ hats',kind='text_title')
add(298,49,'miɑʔn|phɛriʋɑlɑ|ʈɔpi|idi=kɛtɛ|hɑʈ|sɛn-kɛn=ə=ɛ','one|hawker|hat|take=cvb|market|go-pst.mid=fin=3sg.anim','A hawker took some hats to market (= having taken hats, went to market).')
add(299,50,'sɛ-sɛn|sɛ-sɛn=tɛ|thəkɑ-ɛn=ə=ɛ','rdp-go|rdp-go=sim|become.tired-pst.mid=fin=3sg.anim','Walking along he became tired.')
add(299,51,'miɑʔn|dubɑ|pɛndɑrɛ|dubʔm-ɛn=ə=ɛ','one|tree|under|sit.down-pst.mid=fin=3sg.anim','He sat down under a tree.')
add(299,52,'dub|dub=tɛ|gitiʔ-ɛn=ə=ɛ','sit.down|sit.down=sim|sleep-pst.mid=fin=3sg.anim','While sitting there he fell asleep.')
add(299,53,'dubɑ|tɛŋrɛ|bɑndrɑ=kun|ɲɛl-ɛkɛn=ɑ=ku','tree|on.top.of|monkey=pl|see-pst.ipfv.act=fin=3pl','At the top of the tree the monkeys watched/were watching.')
add(299,54,'phɛriʋɑlɑ|gitiʔ-ɛn=ə=ɛ|mɛnthã|hiʔc=kɛtɛ|ʤɛtɛ|ʈɔpi=kɛ|pindh=kɛtɛ|dubɑ=rɛ|rɑkɑʔm-ɛn=ɑ=ku','hawker|sleep-pst.mid=3sg.anim|then|come=cvb|all|hat=obj|put.on=cvb|tree=loc|climb-pst.mid=fin=3pl','The hawker fell asleep then, having come, they [= the monkeys] put on all the hats and climbed up (= in) the tree.',note='Printed sleep interlinear omits fin despite surface =ə; preserve printed analysis.')
add(299,55,'phɛriʋɑlɑ=rɛn|durum|bhɑŋɑ-ɛn=ɑ','hawker=gen|sleep(n.)|open(itr.)-pst.mid=fin','The hawker woke up (= the hawker’s sleep opened).')
add(299,56,'ɲɛl-lɑʔ=ə=ɛ|ʤɛ|miɑʔn|ɑu|ʈɔpi|kɑnɔʔɔ','see-pst.perf.act=fin=3sg.anim|then|one|and|hat|neg.prs.cop','He saw then that there was (= is) not one single hat.')
add(299,57,'mɔn|dukh=kɛtɛ|dubɑ|tɛŋkɛ|sɑŋgil-lɑʔ=ə=ɛ','mind(n.)|be.sad=cvb|tree|on.top.of|look.upwards-pst.perf.act=fin=3sg.anim','Feeling sad he looked up at the tree.')
add(299,58,'ɲɛl-lɑʔ=ə=ɛ|ʤɛ|ʤɛtɛ|ʈɔpi=kɛ|bɑndrɑ=kun|pindh-ɑkɑn=ɑ=ku','see-pst.perf.act=fin=3sg.anim|then|all|hat=obj|monkey=pl|put.on-perf.mid=fin=3pl','Then he saw that all the monkeys were wearing (= have put on) all the hats.')
add(300,59,'bɑndrɑ=kun=kɛ|kɑthɑ-lɑʔ=ə=ɛ|dɑn|rɛ|bɑndrɑ=kun|ʈɔpi=kɛ|hɑʈ|sɛn=ɛŋ','monkey=pl=obj|say-pst.perf.act=fin=3sg.anim|give.imp(IA)|voc|monkey=pl|hat=obj|market|go=1sg','He said to the monkeys “Give the hats [to me], oh monkeys! I will go to market.',note='Direct quotation opens before dɑn and closes in example 60; retain numbered unit boundary and cross-reference both examples.')
add(300,60,'ʈɔpi|ɑkriŋ=kɛtɛ|ʧhɑʋɑ=kun=rɛn|lɑhiʔɲ=kɛ|dɑnɑ|ɛm=ku=ɛŋ','hat|sell=cvb|child=pl=gen|belly=obj|food|give=3pl.obj=1sg','I will sell (= having sold) the hats [and] give the children food”.',note='Closes direct quotation opened in example 59; no invented independent narrative context.')
add(300,61,'sɔbu|kɑthɑ|ɑjum=kɛtɛ|bɑndrɑ=kun|sɔbu|ʈɔpi|ɛm-tɑd=i=ɑ=ku','all|story|hear=cvb|monkey=pl|all|hat|give-pst.act=3sg.anim.obj=fin=3pl','After hearing his whole story, the monkeys gave him all the hats.')
add(300,62,'phɛriʋɑlɑ|lɑndɑ|lɑndɑ=tɛ|hɑʈ|sɛn-lɛn=ə=ɛ','hawker|laugh|laugh=sim|market|go-pst.perf.mid=fin=3sg.anim','The hawker went off happily (= laughing, went) to market.')
if __name__=='__main__':
    (HERE/'texts-first-reading.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    print(f'{len(records)} numbered units, {sum(len(r["forms"]) for r in records)} aligned source words; original review pending')
