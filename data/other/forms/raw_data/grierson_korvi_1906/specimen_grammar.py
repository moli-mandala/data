"""Conservative tags from explicit interlinear glosses, without new lemmas.

Relational English 'to' and 'by' are retained without inventing a particular
source case. Grammatical attribution concerns the printed gloss, not a new
analysis of the source language's morphology.
"""
PRON = {'i': ['1sg'], 'me': ['1sg'], 'my': ['1sg','poss'], 'thou': ['2sg'], 'thee':['2sg'], 'thy':['2sg','poss'], 'thine':['2sg','poss'], 'you':['second-person'], 'your':['second-person','poss'], 'he':['3sg'], 'him':['3sg'], 'his':['3sg','poss'], 'she':['3sg'], 'her':['3sg'], 'they':['3pl'], 'them':['3pl'], 'we':['1pl'], 'us':['1pl'], 'our':['1pl','poss']}
PAST=set('asked became began called came did entreated felt gave gave-away inquired kept passed put remained said sent squandered stood wasted went wrote divided-gave kiss-gave sin-did'.split())
SIMPLE=set('call come comes die eatest enjoy enjoyest give go keep live perform prepare write'.split())
PART=set('becoming building coming dancing eating embracing existing feeling filling going hearing leaving living remembering rising saying seeing singing standing taking telling'.split())
INFINITIVES={'to-beg','to-eat','to-feed','to-make-for','to-praise','to-write','in-order-to-bring','to-be-called'}
PASSIVE={'is-found','is-obtained','was-made-out','was-not-found','was-obtained-not','being-troubled','being-vexed','to-be-called'}
TEMP=set('afterwards then now afterwards again always ever ago daily forthwith meanwhile to-day to-morrow in-the-morning'.split())
SPACE=set('here there near far out outside thence there-from north'.split())
NUM=set('one two three four five-persons two-hundred one-one'.split())
PROPER={'kṛishṇa','mādūrāya','purandargad','śaraṇya','śidaliṅgappa','śindagi','śirśād'}
ADJ=set('alive bare best crafty elder false former great happy little mighty miserly old poor pregnant proper worthy younger youngest'.split())
NOUN=set('brāhman father god king mamlatdār rāo-sāhib survey-number accused banker feast kiss ring anger answer area backyard banking belly body brother business care charitable-acts charity child cloth clothes company copper-coins cottage country court day days deed disguise distance door expenditure fair famine feet field finger food forest friends greatness happiness harlots hours house hunger husband husk husks identification information man memory mercy merit mind miserliness money month morning news nose nose-ring office ornaments pearl-ring performer pity possession poverty pride property registration satisfaction security servant servants service share shoes sin son state stomach swine thread-ceremony time trouble village wife wives woman word work work-man work-men work-people year years'.split())

def classify(unit):
 g=unit['gloss'].lower();tags=list(unit.get('tags',[]));observations=[]
 if g=='time-is' and unit.get('source_commentary'):
  return tags+['uncertain'],['Printed interlinear/free-translation mismatch; no grammatical interpretation inferred from the anomalous English label.']
 base=g
 for prefix in ['the-','a-','of-']:
  base=base.removeprefix(prefix)
 for suffix in ['-near-from','-in-from','-according-to','-among','-from','-near','-alone','-all','-even','-indeed','-with','-of','-to','-in','-on','-as','-for','’s']:
  if base.endswith(suffix):base=base[:-len(suffix)]
 if base in PRON:tags+=['pron','personal']+PRON[base]
 elif base in {'that','this','the-same','same'}:tags+=['demonstrative']
 elif base in {'who','what','whose','how-many'}:tags+=['interr']
 elif base in {'anyone','anybody','anything','whatever','some-one','a-certain','certain'}:tags+=['indef']
 if base in NUM:tags+=['num']
 if base in PROPER:tags+=['proper-noun']
 elif base in NOUN:tags+=['noun']
 elif base in ADJ or base.removeprefix('a-') in ADJ:tags+=['adj']
 if base in {'best','elder','younger','youngest'}:tags+=['degree']
 if g in {'as-for-myself'}:tags+=['pron','1sg','refl']
 if g in {'all','many','much','more','some','so-many','a-few'}:tags+=['quantifier']
 if g in {'a','a-certain'}:tags+=['determiner','indef']
 if g in {'very','in-the-least','so-as-to-exceed'}:tags+=['adv','degree']
 if g in {'dead','a-big','a-far','male','safe-and-sound','past'}:tags+=['adj']
 if g in {'goat-young','male-children','his-father-to','the-kulkarṇi','performer-not'}:tags+=['noun']
 if g in {'children','male-children','days','friends','hours','husks','wives','years','clothes','copper-coins','ornaments','shoes','servants-to','work-men-in','work-men-to'}:tags+=['pl']
 if g in {'that-after','that-reason-for'}:tags+=['demonstrative']
 if g in {'what-is-all','why-if-said'}:tags+=['interr']
 if g in TEMP:tags+=['adv','temporal']
 if g in SPACE:tags+=['adv','spatial']
 if g in {'and','but','however','if','as','as-soon-as','like','so','therefore','thus','for-that-reason','in-this-way'}:tags+=['conj'] if g in {'and','but','if','as','as-soon-as'} else ['adv']
 if g in {'also','even','certainly'}:tags+=['part']
 if g.startswith('having-') or '-having-' in g:tags+=['verb','conjunctive-participle']
 elif g in INFINITIVES:tags+=['verb','inf']
 elif g in PAST or g.removesuffix('-not') in PAST:tags+=['verb','pret']
 elif g in SIMPLE or g.removesuffix('-not') in SIMPLE:tags+=['verb']
 elif g in PART or g.endswith(('-being','-becoming')) or g in {'when-coming','running-going','eaten-that','given-so','whatever-being-though'} or g.startswith('being-') or g.endswith(('-being','-becoming','-when','-while','-after')) and any(g.startswith(x) for x in PART):tags+=['verb','participle']
 elif g in {'am','is','art','was','were','is-not'}:tags+=['verb','copula','pret' if g in {'was','were'} else 'pres']
 elif g.startswith(('am-','is-')) or g.endswith(('-is','-am-not')):tags+=['verb','pres']
 elif g.startswith('was-') or g.endswith('-was'):tags+=['verb','pret']
 elif g.startswith(('has-','hast-','have-','had-')):tags+=['verb','perfect']
 elif g.startswith('will-'):tags+=['verb','fut']
 elif g.startswith('used-to-'):tags+=['verb','pret','ipfv']
 elif g in {'broke-not','gavest-not','heard-not','performed-not','saw-not','lost-went','he-sat'}:tags+=['verb','pret']
 elif g in {'to-write-caused','to-cause-to-abandon-in-order'}:tags+=['verb','caus']
 elif g in {'call-do-not','let-us-become','put-on'}:tags+=['verb']
 elif g in {'said-as','entreated-according-to','to-be-heard-came'}:tags+=['verb','pret']
 elif g in {'care-is-not'}:tags+=['verb','pres','neg']
 elif g=='that-can-eat':tags+=['verb','modal']
 elif g=='that-has-devoured':tags+=['verb','perfect']
 if g in PASSIVE:tags+=['pass']
 if 'verb' in tags:
  if g.startswith('am-') or g.endswith('-am-not') or g=='am':tags+=['1sg']
  if g.startswith('hast-') or g in {'art','eatest','enjoyest','gavest-not'}:tags+=['2sg']
  if g.startswith('he-'):tags+=['3sg']
  if 'not' in g.split('-'):tags+=['neg']
  if g.startswith(('am-','was-','is-')) and any(x in g for x in ['dying','performing','coming','filling','going-on']):tags+=['progressive']
 if g.endswith('-of') or g.endswith('’s') or g.startswith('of-'):tags+=['gen']
 if g.endswith('-from'):tags+=['abl']
 if g.endswith('-in'):tags+=['loc']
 if g.endswith('-to'):observations.append('Source English to retained without choosing accusative, dative or directional case.')
 return list(dict.fromkeys(tags)),observations
