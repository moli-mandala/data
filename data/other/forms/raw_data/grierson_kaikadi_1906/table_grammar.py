"""Metadata from explicitly labelled standard-list contrasts, not guessed morphology."""
def classify(n):
 if n<=13:return ['num']
 if n<=31:
  group=(n-14)//3;person=['1sg','1pl','2sg','2pl','3sg','3pl'][group]
  return ['pron',person]+(['gen','poss'] if (n-14)%3 else [])
 if n<=76:return ['noun']
 if 77<=n<=85:return ['verb','impv']
 if 86<=n<=91:return ['adv','spatial']
 if n in [92,93]:return ['pron','interr']
 if n==94:return ['adv','interr']
 if n in [95,96]:return ['conj']
 if n in [98,100]:return ['interj']
 if n==99:return ['part','neg']
 if 101<=n<=118:
  local=(n-101)%9;case={1:'gen',2:'dat',3:'abl',6:'gen',7:'dat',8:'abl'}.get(local)
  return ['noun','sg' if local<4 else 'pl']+([case] if case else [])
 if 119<=n<=127:
  local=n-119;case={1:'gen',2:'dat',3:'abl',6:'gen',7:'dat',8:'abl'}.get(local)
  return ['noun','m','sg' if local<4 else 'pl']+([case] if case else [])
 if 128<=n<=131:return ['noun', 'm' if n==129 else 'f','pl' if n==130 else 'sg']
 if 132<=n<=137:return ['adj']+(['degree'] if n not in [132,135] else [])
 if 138<=n<=155:
  local=n-138
  if n>=150:return ['noun']+(['m'] if n in [150,153] else ['f'] if n in [151,154] else [])+['pl' if n in [152,155] else 'sg']
  return ['noun','m' if local%4 in [0,2] else 'f','sg' if local%4<2 else 'pl']
 if 156<=n<=167:return ['verb','copula','pres' if n<=161 else 'pret',['1sg','2sg','3sg','1pl','2pl','3pl'][(n-156)%6]]
 if n in [168,175,217]:return ['verb','impv']
 if n in [169,176]:return ['verb','inf']
 if n in [170,177,218,219]:return ['verb','participle']
 if n in [171,178]:return ['verb','conjunctive-participle']
 if n in [172,194]:return ['verb','modal','1sg']
 if n==173:return ['verb','fut','1sg']
 if n in [174,201]:return ['verb','conditional','1sg']
 for start,stop,tense in [(179,184,'pres'),(185,190,'pret'),(195,200,'fut'),(205,210,'pres'),(211,216,'pret')]:
  if start<=n<=stop:return ['verb',tense,['1sg','2sg','3sg','1pl','2pl','3pl'][n-start]]
 if n in [191,192,193]:return ['verb','1sg','pres' if n==191 else 'pret','perfect' if n==193 else 'progressive']
 if n in [202,203,204]:return ['verb','pass','1sg',{202:'pres',203:'pret',204:'fut'}[n]]
 if n>=220:return ['sentential']
 return []
