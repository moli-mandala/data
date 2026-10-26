#!/usr/bin/env python3
"""score_all.py — every lect's scores (extension-aware NED, verb-menu NED, extension-aware accent and tone;
exact rates) for a rules directory, with the current engine. (Plain bare-headword NED is deprecated.)

    python3 sound_changes/score_all.py [RULES_DIR] OUT.json [--workers 2]
"""
import sys, os, csv, json
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

def one(args):
    import sca, evaluate as ev
    d, stem = args
    cen = os.path.join(os.path.dirname(os.path.abspath(d)), 'changes.sc')
    if os.path.abspath(d) != os.path.abspath(os.path.join(HERE, 'rules')) and os.path.exists(cen):
        sca.CENTRAL = cen   # a snapshot: its own central changes (names may differ from the live file)
    p = os.path.join(d, f"OIA-{stem}.sc")
    rules = sca.load_rules(p); _, tr, te = ev.datasets(p)
    out = {}
    for nm, ds in (("train", tr), ("test", te)):
        x = ev.ext_metrics(ds, rules)
        out[nm] = dict(ext=x["ext_ned"], ext_exact=x["ext_exact"],
                       verb=x["verb_ned"] if x["verb_pairs"] else None, verb_exact=x["verb_exact"] if x["verb_pairs"] else None,
                       ext_acc=x.get("ext_aned") if ds.n_apairs else None,
                       ext_tone=x.get("ext_tned") if ds.n_tpairs else None,
                       pairs=ds.n_pairs, apairs=ds.n_apairs, tpairs=ds.n_tpairs)
    return stem, out

if __name__ == "__main__":
    import sca
    argv = list(sys.argv[1:])
    w = 2
    if "--workers" in argv:
        k = argv.index("--workers"); w = int(argv[k + 1]); del argv[k:k + 2]
    d, outp = (argv[0], argv[1]) if len(argv) == 2 else (os.path.join(HERE, "rules"), argv[0])
    stems = [r["File"] for r in csv.DictReader(open(os.path.join(HERE, "languages.tsv"), encoding="utf-8"), delimiter="\t")
             if sca.has_chain(os.path.join(d, f"OIA-{r['File']}.sc"))]
    res = {}
    with ProcessPoolExecutor(w) as ex:
        for s, o in ex.map(one, [(d, s) for s in stems]):
            res[s] = o
    json.dump(res, open(outp, "w"), indent=0)
    print(f"scored {len(res)} lects → {outp}")
