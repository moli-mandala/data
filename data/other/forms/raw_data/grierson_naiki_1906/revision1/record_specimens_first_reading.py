"""Record source-aligned first readings, preserving physical line/word keys.

Source page numbers and aligned units are retained before any analysis/reuse.
This does not install forms. Every transcription remains subject to independent
comparison with the original, including vowel lengths and damaged type.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Each item is one visible aligned pair; speaker labels are real printed pairs.
PAGES = {
    572: [
        [('Oko','One'), ('mankyākon','man-to'), ('iroṭēr','two'), ('pōrākōr','sons'), ('aṇḍēn.','were.'), ('Aunnē-lōpul','Them-in'), ('lakkā','the-younger'), ('bānun','father-to')],
        [('yeṇḍēn,','said,'), ('bā,','father,'), ('aun','which'), ('dhan-sampat','property-wealth-(of)'), ('hissā','share'), ('ānun','me-to'), ('varlēn','comes'), ('ād','that'), ('sī.','give.'), ('Māṅg','Then')],
        [('aunns','he'), ('aun','to-him'), ('sampat','property'), ('vāṭā-kāktun','share-making'), ('sītēn.','gave.'), ('Māṅg','Then'), ('āṅgun','few'), ('pōd','days'), ('eddūn','having-become')],
        [('lakkā','the-younger'), ('pōrā','son'), ('ittēn','all'), ('jamāktun','having-gathered'), ('khōmbāḍ','far'), ('dēśāntun','country-to'), ('seddēn,','went,'), ('ajuk','and'), ('attān','there'), ('bēnāk','evil')],
        [('kharoha-kaktēn','expenditure-made'), ('īpal','his'), ('sampat','property'), ('uḍāptēn.','squandered.'), ('Māṅg','Then'), ('aun','he'), ('ittēn','all'), ('kharohiptēn,','spent,')],
        [('ā','that'), ('dēśām-lōpul','country-in'), ('phari','big'), ('mahāg','famine'), ('paṭṭul.','fell.'), ('Ādi-yauga-lōpul','That-reason-in'), ('aunun','him-to'), ('aḍchan','difficulty')],
        [('paṭṭun.','fell.'), ('Ād','That'), ('vaktun','time-at'), ('aun','he'), ('ād','that'), ('dēśām-lōpul','country-in'), ('oko','one'), ('mankyākani','man-near'), ('seddun','having-gone')],
        [('aṇḍēn.','remained.'), ('Aun','He'), ('aunun','him'), ('turrēl','swine'), ('mēpēn','to-feed'), ('āplē','his'), ('vāvarhun','field-to'), ('sōrtān.','sent.'), ('Atrō','Then'), ('turrēl','swine')],
        [('aun','which'), ('pōllē','husk'), ('tīfchinaṇtē','eating-were'), ('aunē-lōpul','that-on'), ('aunun','him-to'), ('āplē','his'), ('poṭṭā','belly'), ('indurpēn','will-fill'), ('indēn','saying')],
        [('aunun','him-to'), ('vāṭāylun.','it-appeared.'), ('Ajuk','And'), ('yēnī','anyone'), ('aunun','him-to'), ('tānī','anything'), ('siyētēn.','gave-not.'), ('Māṅg','Then'), ('aun','he'), ('suddhīn','sense-in')],
        [('vāttēn','came'), ('yeṇḍēn,','said,'), ('annē','my'), ('bānē','father-of'), ('ettā','how-many'), ('chākarhārākun','servants-to'), ('bharpūr','sufficient'), ('āṭl','bread'), ('antē,','was,')],
        [('ajuk','and'), ('ān','I'), ('sākālā','hunger-with'), ('tīkēntān.','am-dying.'), ('Ān','I'), ('suyitēn','arising'), ('āplē','my'), ('bānēkil','father-to'), ('sērtān','go'), ('ajuk','and'), ('aunun','him-to')],
        [('ān','I'), ('entān,','say,'), ('arē','O'), ('bā,','father,'), ('ān','I'), ('akāsnē','heaven-of'), ('urpaṭlyā','against'), ('ajuk','and'), ('inē','thy'), ('murtōn','face-to'), ('pāp','sin'), ('kaktēn','doer')],
        [('anlēn.','am.'), ('Indiṭlā','Henceforward'), ('inē','thy'), ('kikēn','son'), ('eṇḍēn-sāṭī','saying-for'), ('ānun','me-to'), ('yaug','fitness'), ('nāhī.','not.'), ('Āplē','Your'), ('oko','one')],
        [('chākrā-lāik','servant-like'), ('ānun','me'), ('īḍ.','keep.'), ('Māṅg','Then'), ('aun','he'), ('suyitēn','arising'), ('āplē','his'), ('bānēkil','father-to'), ('seddēn.','went.'), ('Atrō','Then')],
        [('aun','he'), ('khōmbāḍ','far'), ('antē','was'), ('ittē-lōpul','that-in'), ('aunē','his'), ('bān','father'), ('aunun','him'), ('siktēn','saw'), ('lōbh-eddūn','pity-coming'), ('ajuk','and'), ('aun','he')],
        [('ḍuṭēn','ran'), ('aunē','his'), ('guḍunga-lōpulu','neck-on'), ('nujī','embracing'), ('tāptēn,','struck,'), ('ajuk','and'), ('aunē','his'), ('chumā','kiss'), ('ēktēn.','took.')],
    ],
    573: [
        [('Māṅg','Then'), ('pōrā','the-son'), ('aunun','him-to'), ('eṇḍēn,','said,'), ('bā,','father,'), ('akāsnē','heaven-of'), ('urpaṭlyā','against'), ('inē','thy'), ('murtōn','presence-in')],
        [('ān','I'), ('pāp','sin'), ('kaktēn','doer'), ('anlēn,','am,'), ('ajuk','and'), ('indiṭlā','Henceforth'), ('inē','thy'), ('kikēn','son'), ('eṇḍēn-sāṭī','saying-for'), ('ānun','me-to')],
        [('yaug','fitness'), ('nāhī.','not.'), ('Par','But'), ('bān','the-father'), ('āplē','his'), ('chākrākun','servants-to'), ('iḍuktēn,','said,'), ('chāṅglā','good'), ('jhagā','cloth'), ('kōttun','bringing')],
        [('ivunu','this-to'), ('tāpp.','put.'), ('Ajuk','And'), ('iunē','his'), ('kayi-lōpul','hand-on'), ('uṅgyām','ring'), ('ajuk','and'), ('kāl-lōpul','foot-on'), ('jōḍē','shoes'), ('tāpp.','put.')],
        [('Māṅg','Then'), ('āpan','we'), ('tinnun','eating'), ('majā','feast'), ('kaknām.','shall-make.'), ('Tāndun,','Because,'), ('kā','that'), ('iun','this'), ('annē','my'), ('pōrā','son'), ('tīktēn','dead')],
        [('aṇḍēn,','was,'), ('aun','he'), ('maltun','again'), ('jitē','alive'), ('eddēn;','became;'), ('ajuk','and'), ('davḍitēn','lost'), ('aṇḍēn,','was,'), ('aun','he'), ('sāpḍitēn.','was-found.')],
        [('Atrō','Then'), ('add','they'), ('majā','merry'), ('kakēl','to-make'), ('lagtēn.','began.')],
        [('Ād','That'), ('vaktun','at-time'), ('aunē','his'), ('vaḍil','big'), ('kikēn','son'), ('vāvar-lōpul','field-in'), ('aṇḍēn.','was.'), ('Māṅg','Then'), ('aunu','he'), ('vattēn','came')],
        [('āpal-mērān','house-near'), ('vattēn,','came,'), ('aun','he'), ('vājā','music'), ('ajuk','and'), ('ēnd','dancing'), ('vēṇḍēn.','heard.'), ('Atrō','Then'), ('chākar-lōpul','servants-in')],
        [('okkon','one'), ('karūktu','calling'), ('aun','he'), ('pusāyitēn,','asked,'), ('id','this'), ('tāndēn?','what-is?'), ('Aun','He'), ('aunun','him-to'), ('iḍuktēn','said'), ('ki,','that,')],
        [('innē','thy'), ('tōlēn','brother'), ('vattēn,','came,'), ('ajuk','and'), ('aun','he'), ('inē','thy'), ('bānun','father-to'), ('sukhāchā','safely'), ('mirātun,','having-met,'), ('iūn-sāṭīn','this-for')],
        [('aun','he'), ('phar','big'), ('pāṅgat','feast'), ('kaktēn.','made.'), ('Atrō','Then'), ('aun','he'), ('rāgunu','anger-to'), ('vattēn','came'), ('lōpul','inside'), ('sērēn.','went-not.')],
        [('Iūn-sāṭīn','Therefore'), ('aunē','his'), ('bān','father'), ('bāhēr','outside'), ('vattēn','came'), ('aunun','him'), ('samjiptēn.','entreated.'), ('Pan','But'), ('aunē','his'), ('bānun','father-to')],
        [('utar','answer'), ('sītēn','gave'), ('ki,','that,'), ('aik,','lo,'), ('ān','I'), ('iṅgōn','so-many'), ('vars','years'), ('innē','thy'), ('chākrī','service'), ('kākēntān,','do,'), ('ajuk','and')],
        [('innē','thy'), ('ādnyā','command'), ('ān','I'), ('etrōs','ever'), ('mōḍāpētān,','broke-not,'), ('yetrō','still'), ('ān','I'), ('āplē','my'), ('dōstā-barōbar','friends-with'), ('majā','merry')],
        [('kakkēn','might-make'), ('mhūn','saying'), ('īn','thou'), ('ānnun','me-to'), ('etrō','ever'), ('piyēṭē','kid'), ('siyētun.','gavest-not.'), ('Ajuk','And'), ('aun','he'), ('innē','thy')],
        [('sampat','property'), ('kalātnē','harlots'), ('barōbar','with'), ('tindunu','eating'), ('surtē,','squandered,'), ('aun','that'), ('id','this'), ('innē','thy'), ('kikēn','son'), ('vattēn,','came,')],
        [('atrō','then'), ('īn','thou'), ('aunē-sāṭī','him-for'), ('phar','great'), ('pāṅgat','feast'), ('kaktēn.','madest.'), ('Atrō','Then'), ('aun','he'), ('aunun','him-to'), ('aṇḍēn,','said,'), ('pōrā,','son,')],
        [('īn','thou'), ('hamēśā','always'), ('annē','me'), ('barōbar','with'), ('anlēn.','art.'), ('Ajuk','And'), ('annē','my'), ('ittēnī','all'), ('sampat','property'), ('innēd','thine'), ('anlēn.','is.')],
        [('Pan','But'), ('majā','feast'), ('ajuk','and'), ('khushī','merry'), ('kakkēn','to-make'), ('id','this'), ('yaug','fit'), ('antēr,','was,'), ('kā-ki','because'), ('id','this'), ('innē','thy'), ('tōlēn','brother')],
        [('tīktēn','dead'), ('aṇḍēn,','was,'), ('aun','he'), ('partun','again'), ('jitē','alive'), ('yeddēn;','became;'), ('ajuk','and'), ('harpitēn','lost'), ('aṇḍēn,','was,'), ('aun','he'), ('sāpḍitēn.','was-found.')],
    ],
    574: [
        [('Oko','One'), ('mankyākun','man-to'), ('iroṭēr','two'), ('lakkā','small'), ('pōrākōr','children'), ('aṇḍēn,','were,'), ('oko','one'), ('pōrā','boy'), ('ajuk','and'), ('oko','one')],
        [('pōrī.','girl.'), ('Pōrō','Boy'), ('aun','he'), ('challā','appearance'), ('pharī','very'), ('chokkōṭ','good'), ('aṇḍur,','was,'), ('pōrī','girl'), ('sāvatāṅg','common'), ('aṇḍur.','was.')],
        [('Oko','One'), ('pōd','day'), ('at','those'), ('iroṭēr','two'), ('pōrākōr','children'), ('ārasā-mērān','glass-near'), ('āḍsīn','playing'), ('aṇḍēr,','were,'), ('pōrā','boy')],
        [('pōrīn','girl-to'), ('yantēr,','said,'), ('ari,','well,'), ('id','this'), ('ārasā-lōpul','glass-in'), ('ām','we'), ('āykēkād,','will-see,'), ('chokkōṭ','handsome'), ('ēn','who')],
        [('chōvāylēn.','appears.'), ('Pōrīn','Girl-to'), ('ād','that'), ('kharāb','bad'), ('anlēn.','is.'), ('Innē','To-her'), ('samjītun','it-appeared'), ('ki,','that,')],
        [('aun','he'), ('id','this'), ('ānun','me'), ('inbarēntōtēn','to-lower'), ('iūn.','said.'), ('Māṅg','Then'), ('idd','she'), ('bānē','father'), ('mērān','near'), ('sēdun','having-gone')],
        [('tōlēnē','brother-of'), ('garhān','complaint'), ('iduktēn,','said,'), ('enlēn,','says,'), ('bā,','father,'), ('ārasā-lōpul','glass-in'), ('challā','figure'), ('siktān','seeing')],
        [('samādhān','satisfaction'), ('pāvāytunē','is-felt'), ('idd','this'), ('bāykōnē','women-of'), ('kām.','work.'), ('Ōnē-lōpul','That-on'), ('mankyāk','man'), ('jīvām','mind')],
        [('tābgārā.','should-not-put.'), ('Bān','The-father'), ('iroṭērun','both'), ('pōṭāl','belly-to'), ('samtēn','took'), ('auntun','them'), ('samjīptēn.','entreated.')],
        [('Aun','He'), ('eṇḍēn,','said,'), ('pōrākōr,','children,'), ('īm','you'), ('jhagḍīlnēr.','quarrel-not.'), ('Inēṭlā','Hence'), ('īm','you'), ('iroṭēr','both'), ('rōjjā','daily')],
        [('ārasā-lōpul','glass-in'), ('sīk.','look.')],
        [('Pōrī.—','Daughter.—'), ('Bā,','Father,'), ('Sōmā','Soma'), ('gavaḍyāk','milkman'), ('pāl','milk'), ('samtun','taking'), ('kōtēn','bringing'), ('antēn.','was.'), ('Aun','He')],
        [('yantēn,','said,'), ('ettē','how-much'), ('pāl','milk'), ('sīyān?','shall-give?')],
        [('Bān.—','Father.—'), ('Pōrī,','Girl,'), ('aunu','him-to'), ('iḍuk','say'), ('ki,','that,'), ('inēu','to-day'), ('ār-sōlam','one-seer'), ('pāl','milk'), ('ērāl,','enough,')],
        [('vēgēn','to-morrow'), ('yeraṇḍi','two'), ('sēr','seer'), ('tōnān','taking'), ('vā.','come.')],
    ],
    575: [
        [('Pōrī.—','Daughter.—'), ('Bā,','Father,'), ('gavaḍyāk','milkman'), ('pāl','milk'), ('yadāḍin','whence'), ('kōrtēn?','brought?')],
        [('Bān.—','Father.—'), ('Inun','Thee-to'), ('thāvkā','known'), ('nāi','not'), ('kā','that'), ('aunē','his'), ('apāt','house'), ('khutēl','cows'), ('antē,','are,'), ('sirlā','buffaloes')],
        [('antē.','are.'), ('Aunē','Their'), ('pāl','milk'), ('pustun','drawing'), ('aun','he'), ('kōrtēn.','brought.')],
        [('Pōrī.—','Daughter.—'), ('Bā,','Father,'), ('khutēl','cow'), ('yettē','how-much'), ('pāl','milk'), ('siylēn,','gives,'), ('ajuk','and'), ('sir','buffalo')],
        [('yettē','how-much'), ('pāl','milk'), ('siylēn?','gives?')],
        [('Bān.—','Father.—'), ('Oko','One'), ('oko','one'), ('khutēl','cow'), ('yeraṇḍi','two'), ('yeraṇḍi','two'), ('sērla','seer'), ('pāl','milk'), ('siylēn,','gives,')],
        [('ajuk','and'), ('sir','buffalo'), ('nāli','four'), ('nāli','four'), ('siylēn.','gives.')],
    ],
}

def main():
    rows = []
    for page, lines in sorted(PAGES.items()):
        for line, pairs in enumerate(lines, 1):
            for word, (form, gloss) in enumerate(pairs, 1):
                rows.append(dict(source_unit_key=f'grierson1906naiki:p{page}:specimen:line{line}:word{word}',
                                 printed_page=page, physical_djvu_page=page+20,
                                 specimen='II-popular-tale' if page >= 574 else 'I-prodigal-son',
                                 line=line, word=word, raw_form=form, raw_gloss=gloss,
                                 source_locality='District Chanda',
                                 role='glossed-speaker-label' if form.endswith('.—') else 'aligned-token',
                                 review_status='root first reading; independent original/output audit pending'))
    assert len({r['source_unit_key'] for r in rows}) == len(rows)
    (HERE/'specimens-first-reading.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    (HERE/'specimens-first-reading-status.json').write_text(json.dumps({
        'stage':'in-progress first reading; no canonical installation',
        'pages_recorded': sorted(PAGES), 'aligned_units': len(rows),
        'page_counts': {str(p):sum(len(line) for line in lines) for p,lines in PAGES.items()},
        'whole_specimen_pages_required':[572,573,574,575],
        'remaining_pages':sorted(set([572,573,574,575])-set(PAGES)),
        'free_translation_policy':'p575 English prose is contextual evidence, not additional inferred word alignment',
        'uncertainty':'All first-reading glyphs require independent comparison; no automatic normalization of vowel lengths or source contradictions.',
    }, ensure_ascii=False, indent=2)+'\n')
    print(f'{len(rows)} aligned units on {len(PAGES)} pages; source recovery remains in progress')

if __name__ == '__main__':
    main()
