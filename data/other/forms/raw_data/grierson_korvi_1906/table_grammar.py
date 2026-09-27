"""Explicit English prompt grammar for the complete LSI standard table."""

def table_tags(n):
 if n<=13:return ['num']
 if n<=31:
  person=('1sg' if n<=16 else '1pl' if n<=19 else '2sg' if n<=22 else '2pl' if n<=25 else '3sg' if n<=28 else '3pl')
  return ['pron','personal',person]+(['gen','poss'] if n not in [14,17,20,23,26,29] else [])
 if n<=76:return ['noun']
 if n<=85:return ['verb','impv']
 if n<=91:return ['adv','spatial']
 if n in [92,93]:return ['pron','interr']
 if n==94:return ['adv','interr']
 if n<=97:return ['conj']
 if n in [98,99,100]:return ['interj']+(['neg'] if n==99 else [])
 if n<=118:
  q=n-101 if n<=109 else n-110
  return ['noun','pl' if q>=4 else 'sg']+({1:['gen'],2:['dat'],3:['abl'],6:['gen'],7:['dat'],8:['abl']}.get(q,[]))
 if n<=127:
  q=n-119
  return ['noun','adj','pl' if q>=4 else 'sg']+({1:['gen'],2:['dat'],3:['abl'],6:['gen'],7:['dat'],8:['abl']}.get(q,[]))
 if n<=131:return ['noun','adj']+(['pl'] if n==130 else ['sg'])
 if n<=137:return ['adj']+(['degree'] if n in [133,134,136,137] else [])
 if n<=155:return ['noun','pl' if n in [140,141,144,145,148,149,152,155] else 'sg']
 if n<=167:return ['verb','copula','pres' if n<=161 else 'pret',['1sg','2sg','3sg','1pl','2pl','3pl'][(n-156)%6]]
 if n==168:return ['verb','copula','impv']
 if n==169:return ['verb','copula','inf']
 if n==170:return ['verb','copula','participle']
 if n==171:return ['verb','copula','conjunctive-participle']
 if n in [172,173,174]:return ['verb','copula','1sg']+{172:['subjunctive'],173:['fut'],174:['conditional']}[n]
 if n<=178:return ['verb']+{175:['impv'],176:['inf'],177:['participle'],178:['conjunctive-participle']}[n]
 if n<=190:return ['verb','pres' if n<=184 else 'pret',['1sg','2sg','3sg','1pl','2pl','3pl'][(n-179)%6]]
 if n<=194:return ['verb','1sg']+{191:['pres','progressive'],192:['pret','progressive'],193:['pret','perfect'],194:['subjunctive']}[n]
 if n<=200:return ['verb','fut',['1sg','2sg','3sg','1pl','2pl','3pl'][n-195]]
 if n==201:return ['verb','1sg','conditional']
 if n<=204:return ['verb','1sg','pass',{202:'pres',203:'pret',204:'fut'}[n]]
 if n<=216:return ['verb','pres' if n<=210 else 'pret',['1sg','2sg','3sg','1pl','2pl','3pl'][(n-205)%6]]
 if n==217:return ['verb','impv']
 if n in [218,219]:return ['verb','participle']+(['pret'] if n==219 else [])
 return ['sentential']
