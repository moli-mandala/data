"""Explicit contextual POS decisions from root's original-page review572–575.

These are editorial analyses of aligned tokens, not printed POS labels. No
gender, number or case is inferred merely from an English noun translation.
Only the listed exact form/gloss pairs are eligible, ignoring capitalization.
"""

NOUNS = '''
utar|answer
chhallā|appearance
poṭṭā|belly
pōrā|boy
pōrō|boy
āṭl|bread
tōlēn|brother
sir|buffalo
jhagā|cloth
ādnyā|command
garhān|complaint
khutēl|cow
khutēl|cows
pōrī|daughter
pōd|day
pōd|days
aḍchan|difficulty
mahāg|famine
bā|father
bān|father
bānē|father
majā|feast
pāṅgat|feast
chhallā|figure
yaug|fitness
pōrī|girl
kalātinē|harlots
apāṭ|house
pōllē|husk
piyēṭē|kid
chumā|kiss
mankyāk|man
pāl|milk
gavaḍyāk|milkman
jīvām|mind
vājā|music
sampat|property
uṅgryām|ring
samādhān|satisfaction
sēr|seer
sērla|seer
chākrī|service
hissā|share
jōḍē|shoes
pāp|sin
pōrā|son
kikēn|son
bān|the-father
pōrā|the-son
kām|work
vars|years
'''

CONJUNCTIONS = '''
ajuk|and
par|but
pan|but
tāndun|because
kā̃-kī̃|because
'''

# Reviewed in the complete aligned clauses on original pp572–575. These lists
# assign only contextual POS: an English past/participial translation alone
# does not establish the source's morphological tense/aspect/voice category.
VERBS = '''
tīkēntān|am-dying
chōvāylēn|appears
antē|are
suyitēn|arising
pusāyitēn|asked
eddēn|became
yeddēn|became
lagtēn|began
kōttun|bringing
kōtēn|bringing
mōḍāpēttān|broke-not
kōrtēn|brought
karūktu|calling
vāttēn|came
vattēn|came
vā|come
varlēn|comes
kākēntān|do
pustun|drawing
tīñchināntē|eating-were
samjiptēn|entreated
kharcha-kaktēn|expenditure-made
sītēn|gave
siyētun|gavest-not
sērtān|go
eddūn|having-become
sēdun|having-gone
mirāitun|having-met
vēṇḍēn|heard
pāvāytunē|is-felt
vāṭāytun|it-appeared
samjiltum|it-appeared
āik|look
davḍiltēn|lost
harpiltēn|lost
kaktēn|made
kaktēn|madest
kakkēn|might-make
āḍsīn|playing
tāpp|put
ṭuṭēn|ran
aṇḍēn|remained
eṇḍēn|said
iḍuktēn|said
aṇḍēn|said
yantēr|said
iūn|said
yantēn|said
aiktēn|saw
entān|say
iḍuk|say
indēn|saying
mhūn|saying
enlēn|says
aiktān|seeing
sōrtān|sent
sīyān|shall-give
kaknām|shall-make
vāṭā-kāktun|share-making
tābgārē|should-not-put
kharchiptēn|spent
uḍāptēn|squandered
surtē|squandered
tāptēn|struck
sumtun|taking
ēktēn|took
sumtēn|took
aṇḍēn|was
antēr|was
aṇḍur|was
antēn|was
sāpḍiltēn|was-found
aṇḍēn|were
aṇḍēr|were
indutpēn|will-fill
āykēkād|will-see
'''

ADJECTIVES = '''
jitē|alive
kharāb|bad
phari|big
vaḍil|big
phar|big
sāvatāṅg|common
bēnāk|evil
yaug|fit
chāṅglē|good
chokkōṭ|good
phar|great
chokkōṭ|handsome
lakkā|small
bharpūr|sufficient
lakkā|the-younger
'''

ADVERBS = '''
maltun|again
partun|again
hamēśā|always
rōjjā|daily
etrōs|ever
etrō|ever
khōmbāḍ|far
inēṭlā|hence
indiṭlā|henceforth
indiṭlā|henceforward
lōpul|inside
bāhēr|outside
sukhāchā|safely
yetrō|still
māṅg|then
atrō|then
attān|there
iūn-sāṭin|therefore
inēn|to-day
vēgēn|to-morrow
pharī|very
yadāḍin|whence
'''


def decisions():
    result = {}
    for specification, tag in [(NOUNS, 'noun'), (CONJUNCTIONS, 'conj'),
                               (VERBS, 'verb'), (ADJECTIVES, 'adj'),
                               (ADVERBS, 'adv')]:
        for line in specification.strip().splitlines():
            form, gloss = line.split('|')
            pair = form.casefold(), gloss.casefold()
            assert pair not in result
            result[pair] = dict(tags=[tag],
                basis='Editorial contextual POS analysis against original aligned specimen; not a printed POS abbreviation.',
                scope='Only this form/gloss pair; no automatic gender, number, case, tense or graph relationship.')
    for pair in [('kaktēn', 'made'), ('kaktēn', 'madest'),
                 ('kharcha-kaktēn', 'expenditure-made')]:
        result[pair]['uncertainty'] = dict(type='source-analysis', reason=(
            'Contextual verbal analysis follows the aligned clause, but p571 says '
            'kaktēn in kaktēn anlēn and similar forms are apparently nouns of '
            'agency. This token may participate in that analysis; no categorical '
            'verb-versus-agency-noun resolution or morphological decomposition '
            'is asserted.'))
    return result
