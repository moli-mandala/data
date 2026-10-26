"""Conservative structured reading of literal Dhakal interlinear labels.

Unmapped labels remain explicit source labels, never guessed equivalent tags.
Caller must block incompatible source analyses and ambiguous multiword alignment.
"""
import re

LABELS = {
    'PST':['pret'], 'NPST':['non-past'], 'PFV':['pfv'],
    'PRF':['perfect'], 'PROG':['progressive'], 'COND':['conditional'],
    'FUT':['fut'], 'OPT':['opt'], 'PROH':['prohibitive','neg'],
    'PASS':['pass'], 'VOC':['voc'], 'ECHO':['echo'],
    'HON':['honorific'], 'FEM':['f'], 'EMP':['emph'],
    'ADJZ':['adjectivizer'], 'ADJV':['adjectivizer'],
    'HAB':['habitual'], 'HOR':['hortative'], 'HORT':['hortative'],
    'HS':['hearsay'], 'INFR':['inferential'], 'MIR':['mirative'],
    'DESID':['desiderative'], 'DUB':['dubitative'], 'OBLIG':['obligation'],
    'POSB':['possibility'], 'PURP':['purposive'],
    'SEQ':['sequential-converb'], 'SIM':['simultaneous-converb'],
    'IMI':['imminent'], 'FRUS':['frustrative'], 'HYP':['hypothetical'],
    'GEN':['gen'], 'OBL':['obl'], 'DAT':['dat'], 'ERG':['erg'],
    'ABL':['abl'], 'LOC':['loc'], 'ACC':['acc'], 'COM':['comitative'],
    'INST':['instr'], 'NOM':['nom'], 'PL':['pl'], 'SG':['sg'],
    '1SG':['1sg'], '2SG':['2sg'], '3SG':['3sg'],
    '1PL':['1pl'], '2PL':['2pl'], '3PL':['3pl'],
    'INF':['inf'], 'ABS':['abs'], 'IMP':['impv'], 'PROS':['prospective'],
    'CAUS':['caus'], 'POSS':['poss'], 'F':['f'], 'NEG':['neg'],
    'PART':['part'], 'CLF':['classifier'], 'REFL':['refl'], 'RECI':['reciprocal'],
    'EMPH':['emph'], 'INDEF':['indef'],
    'COP':['copula'], 'AUX':['auxiliary'], 'ONO':['onomatopoeia'],
    'REDU':['reduplicated'], 'H':['honorific'], 'HH':['honorific','high-honorific'],
    '3':['third-person'],
    'NOMN':['nominalizer'], 'NOML':['nominalizer'], 'ADVZ':['adverbializer'],
}
# These source abbreviations have no exact registered equivalent or require
# later context review. Do not silently normalize apparent source typos either.
PRESERVED = {'SIM','SEQ','MIR','HOR','HORT','OBLIG','ALL',
             'BEN','INFR','HS','HAB','POSB','PVF','IINF','nP','FOC',
             'DESID','INSTAN','EXCL','IMI','DUB','HYP','PURP','COMP',
             'FRUS','MAN','PRT','CONC','NMLZ','RED','AUXP','3SGPOSS',
             'CNFC','POSS1SG','APROX','Y','2SGPOSS','2POSS','NF',
             'PROB','NMLN','2','POS','QPART'} - set(LABELS)
ALL_CODES = set(LABELS) | PRESERVED
VERBAL = {'PST','NPST','PFV','INF','ABS','IMP','PROS','CAUS','COP','AUX',
          'SIM','SEQ','MIR','HOR','HORT','OBLIG','INFR','HS','HAB','POSB',
          'PRF','PROG','COND','FUT','OPT','PROH','PASS','DESID','DUB',
          'HYP','PURP','FRUS','IMI'}


def parse_source_grammar(label):
    """Interpret only the source's grammar field, never prose lexical definitions."""
    literal=label
    exact={'Habitual':'HAB','Progressive':'PROG','Past/perfect':'PST.PRF',
           'Prospective':'PROS','Conditional':'COND','past':'PST',
           'non-past':'NPST','future':'FUT','‘COP.PST’':'COP.PST',
           'LOC (EMPH)':'LOC.EMPH','DAT/ACC':'DAT.ACC','PST/PRF':'PST.PRF'}
    special={'negative prefix':['neg','prefix'], 'negative suffix':['neg','suffix'],
             'prohibitive prefix':['prohibitive','neg','prefix'],
             'verb':['verb'], 'auxiliary in dative clauses':['auxiliary','verb']}
    if label in special:
        return {'gloss':'','tags':special[label][:],'labels':[literal],
                'raw_source_grammar':literal,'unmapped_source_labels':[]}
    label=exact.get(label,label)
    if re.fullmatch(r'non-past [123](?:SG|PL)(?:\.H)?',label):
        label='NPST.'+label.removeprefix('non-past ')
    match=re.fullmatch(r'([123](?:sg|pl))(?: (non-honorific|honorific|high-honorific))?(?: Set[12])?',label)
    if match:
        label=match[1].upper()+{'honorific':'.H','high-honorific':'.HH'}.get(match[2],'')
    result=parse(label)
    result['raw_source_grammar']=literal
    if result['gloss']:
        result['unmapped_source_grammar']=result['gloss']
    result['gloss']=''
    return result


def parse(gloss, *, ambiguous=False, conflicting=False):
    if ambiguous:
        return {'gloss':gloss, 'tags':[], 'labels':[],
                'held_analysis':'Source alignment is ambiguous; do not assign group labels to individual words.'}
    # Explicit person-number punctuation occurs in this source (3.SG).
    normalized = re.sub(r'([123])\.([SP]L|SG)',r'\1\2',gloss)
    normalized = normalized.replace('[OBL]', '-OBL-')
    # Parse exact whole labels and delimiter-separated suffix labels. Lexical
    # English I, names, daughter-in-law, all-day etc. remain lexical text.
    pieces = re.split(r'([.\-])', normalized)
    labels = []
    lexical = []
    for index,piece in enumerate(pieces):
        token=piece.strip()
        if token in ALL_CODES:
            labels.append(token)
        elif token not in ('','-','.'):
            lexical.append((index,piece))
    # Preserve original separators between untouched lexical pieces.
    cleaned=''
    previous=None
    for index,piece in lexical:
        if previous is not None:
            between=''.join(pieces[previous+1:index])
            cleaned += between if not any(c.strip() in ALL_CODES for c in pieces[previous+1:index]) else ' '
        cleaned += piece.strip()
        previous=index
    tags=[]
    for label in labels:
        tags.extend(LABELS.get(label,[]))
    # A conditional particle is not a lexical verb merely because COND
    # expresses a verbal category elsewhere. Exact PART scopes to this token.
    if set(labels) & VERBAL and 'PART' not in labels:
        tags.append('verb')
    if conflicting:
        tags=[t for t in tags if t not in {'1sg','2sg','3sg','1pl','2pl','3pl'}]
    return {'gloss':cleaned,'tags':list(dict.fromkeys(tags)),
            'labels':labels,'unmapped_source_labels':[l for l in labels if l not in LABELS],
            'raw_gloss':gloss,'analysis_status':'Proposed source-label mapping; contextual review pending'}
