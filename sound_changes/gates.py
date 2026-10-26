#!/usr/bin/env python3
"""gates.py — compare a candidate rule file against the installed one for the same lect.

    python3 sound_changes/gates.py sound_changes/rules/OIA-G.sc /path/to/candidate/OIA-G.sc

The candidate must keep the lect's file name (OIA-<stem>.sc) so it finds the same pairs. Prints
train/test extension-aware NED, verb-menu NED and extension-aware accent/tone, the deltas, exact rates,
and the verdict under the audit gates (notes/BRIEF-audit.md):
  primary  ext NED train improves (for an accent/tone-only rule, which leaves every segmental score
           exactly unchanged: ext_tone or ext_acc train improves instead);
  ext test not worse; verb-menu train and test not worse by more than 0.001; extension-aware accent
  (ext_acc: the accent of the variant the segmental score picks) not worse; extension-aware tone
  (ext_tone: accent position + tone class, scored only on sources that write tone types) not worse.
Plain (bare-headword) NED, accent and tone are deprecated (2026-09-30) and no longer reported or gated.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sca, evaluate as ev

def scores(path):
    rules = sca.load_rules(path); _, tr, te = ev.datasets(path)
    out = {}
    for nm, ds in (("train", tr), ("test", te)):
        x = ev.ext_metrics(ds, rules)
        out[nm] = dict(ext=x["ext_ned"], verb=x["verb_ned"],
                       ext_acc=x["ext_aned"], ext_tone=x["ext_tned"], ext_exact=x["ext_exact"])
    return out

def net_train(a, b):
    """Train reflexes the candidate makes better minus worse (best variant, as ext_metrics picks)."""
    res = []
    for path in (a, b):
        rules = sca.load_rules(path); _, tr, _ = ev.datasets(path)
        keep = []; ev.ext_metrics(tr, rules, keep=keep)
        res.append([sca.edit_distance(q, g) / (max(len(q), len(g)) or 1) for _, _, q, g in keep])
    better = sum(y < x - 1e-9 for x, y in zip(*res)); worse = sum(y > x + 1e-9 for x, y in zip(*res))
    return better, worse


def main():
    a, b = sys.argv[1], sys.argv[2]
    A, B = scores(a), scores(b)
    for nm in ("train", "test"):
        print(f"{nm:5s} " + "  ".join(f"{k} {A[nm][k]:.4f}→{B[nm][k]:.4f} ({B[nm][k]-A[nm][k]:+.4f})"
                                   for k in ("ext", "verb", "ext_acc", "ext_tone")) +
              f"  ext-exact {A[nm]['ext_exact']:.1%}→{B[nm]['ext_exact']:.1%}")
    d = lambda nm, k: B[nm][k] - A[nm][k]
    ok = (d("train", "ext") < 0 and d("test", "ext") <= 0 and d("train", "verb") <= 0.001
          and d("test", "verb") <= 0.001 and d("train", "ext_acc") <= 1e-9 and d("test", "ext_acc") <= 1e-9
          and d("train", "ext_tone") <= 1e-9 and d("test", "ext_tone") <= 1e-9)
    # accent/tone-only change: segmental scores exactly unchanged, the prosodic score is primary
    seg_same = all(abs(d(nm, k)) < 1e-12 for nm in ("train", "test") for k in ("ext", "verb"))
    if seg_same:
        prim = next((k for k in ("ext_tone", "ext_acc") if d("train", k) < -1e-12), None)
        pros_ok = prim is not None and all(d(nm, k) <= 1e-12 for nm in ("train", "test") for k in ("ext_acc", "ext_tone"))
        print("VERDICT:", f"ACCEPT (prosody-only: {prim} improves, segmental scores unchanged)" if pros_ok else "REJECT")
        return
    bt, wt = net_train(a, b)
    print(f"train reflexes better {bt}, worse {wt}, net {bt - wt:+d}")
    print("VERDICT:", "ACCEPT" if ok else "REJECT")

if __name__ == "__main__":
    main()
