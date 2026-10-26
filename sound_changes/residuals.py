#!/usr/bin/env python3
"""residuals.py — classify every miss of a lect's chain (best variant, as the extension-aware score
picks it) and list the commonest predicted→attested ending pairs, with random examples per class.

    python3 sound_changes/residuals.py sound_changes/rules/OIA-G.sc  [> report.txt]
"""
import sys, random, unicodedata as U
from collections import Counter, defaultdict
sys.path.insert(0,'sound_changes'); import sca, evaluate as ev
p=sys.argv[1] if len(sys.argv)>1 else 'sound_changes/rules/OIA-S.sc'
rules=sca.load_rules(p); _,tr,te=ev.datasets(p)
phon=[r for r in rules if '@morph' not in r.tags]
isV=lambda t: sca.kind(t)=='V'
base=lambda t: ''.join(c for c in U.normalize('NFD',t) if not U.combining(c))
IMP=set('ɠʄᶑɓ')
def ops(a,b):
    n,m=len(a),len(b); D=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): D[i][0]=i
    for j in range(m+1): D[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1): D[i][j]=min(D[i-1][j]+1,D[i][j-1]+1,D[i-1][j-1]+(a[i-1]!=b[j-1]))
    out=[];i,j=n,m
    while i or j:
        if i and j and D[i][j]==D[i-1][j-1]+(a[i-1]!=b[j-1]):
            if a[i-1]!=b[j-1]: out.append(('sub',i-1,j-1))
            i,j=i-1,j-1
        elif i and D[i][j]==D[i-1][j]+1: out.append(('del',i-1,j)); i-=1
        else: out.append(('ins',i,j-1)); j-=1
    return out[::-1]
def classify(q,g):
    cats=set(); L=len(g)
    for k,i,j in ops(q,g):
        a=q[i] if k!='ins' else None; b=g[j] if k!='del' else None
        pos_end = (j>=L-2) if k!='del' else (i>=len(q)-2)
        if k=='sub' and i==0 and j==0 and (a in IMP or b in IMP): cats.add('initial implosive'); continue
        if pos_end: cats.add('ending (last 2 segs)'); continue
        if k=='del' and a and isV(a) and ((i>0 and isV(q[i-1])) or (i+1<len(q) and isV(q[i+1]))): cats.add('uncontracted vowel sequence'); continue
        if k=='del' and a and isV(a): cats.add('syncope missing (extra medial vowel)'); continue
        if k=='ins' and b and isV(b): cats.add('vowel missing (epenthesis/over-syncope)'); continue
        if k=='sub' and isV(a) and isV(b):
            if base(a)==base(b):
                if ('̃' in U.normalize('NFD',a))!=('̃' in U.normalize('NFD',b)): cats.add('nasalisation')
                else: cats.add('vowel length')
            elif base(a)=='a' and base(b)=='i' and j+1<L and g[j+1] in ('ṇ','n'): cats.add('a > i before ṇ')
            else: cats.add('vowel quality')
            continue
        if k=='sub' and not isV(a) and not isV(b):
            if a.replace('ʰ','')==b.replace('ʰ',''): cats.add('aspiration')
            else: cats.add('consonant quality')
            continue
        cats.add('consonant inserted/deleted' if (a and not isV(a)) or (b and not isV(b)) else 'other')
    return cats or {'other'}
rows=[]
for ds in (tr,te):
    keep=[]; ev.ext_metrics(ds, rules, keep=keep)   # the variant and prediction the score picks
    raws={(i,g):raw for i in range(len(ds.etyma)) for g,raw in ds.golds[i]}
    for i,n,q,g in keep:
        d=sca.edit_distance(q,g)/max(len(q),len(g),1)
        if d>0: rows.append((d,ds.raw[i],'' if n=='bare' else n,q,g,raws.get((i,g),''.join(g))))
N=len(rows); tot=sum(r[0] for r in rows)
cnt=Counter(); mass=Counter(); ex=defaultdict(list)
single=Counter()
for d,e,n,q,g,raw in rows:
    cs=classify(q,g)
    if len(cs)==1: single[next(iter(cs))]+=1
    for c in cs:
        cnt[c]+=1; mass[c]+=d/len(cs); ex[c].append(f"{e}{'('+n+')' if n else ''} → {''.join(q)} | {raw}")
print(f"{N} missed pairs; mean NED of misses {tot/N:.3f}; big misses (NED>0.6): {sum(r[0]>0.6 for r in rows)}")
print(f"{'class':42s} {'pairs':>6s} {'sole cause':>10s} {'err share':>9s}")
for c,v in cnt.most_common():
    print(f"{c:42s} {v:6d} {single[c]:10d} {mass[c]/tot:9.1%}")
random.seed(3)
for c,_ in cnt.most_common():
    print(f"\n## {c}"); [print("  ",x) for x in random.sample(ex[c], min(10,len(ex[c])))]

print("\n\n## ending pairs (predicted tail → attested tail), all misses whose error touches the ending")
pairs=Counter(); pex=defaultdict(list)
for d,e,n,q,g,raw in rows:
    if 'ending (last 2 segs)' not in classify(q,g): continue
    # the differing tail after the longest common prefix
    k=0
    while k<min(len(q),len(g)) and q[k]==g[k]: k+=1
    a,b=''.join(q[k:])[-4:] or '∅',''.join(g[k:])[-4:] or '∅'
    key=(a,b); pairs[key]+=1
    if len(pex[key])<3: pex[key].append(f"{e}{'('+n+')' if n else ''}→{''.join(q)}|{raw}")
for (a,b),v in pairs.most_common(30): print(f"  -{a:5s} → -{b:5s} {v:4d}   {'; '.join(pex[(a,b)])}")
