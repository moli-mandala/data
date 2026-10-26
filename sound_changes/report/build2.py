#!/usr/bin/env python3
"""Assemble the interactive report: report/template2.html + JSON data → report/index.html.

    python sound_changes/report/build2.py

All data is embedded (the artifact sandbox blocks runtime fetches): per-lect scores for both
rounds, the shared-prefix tree, the accent-test grid and accent table, categorical features, per-lect changes, order evidence,
chronology positions, and a world basemap (world-atlas countries-50m TopoJSON). Charts are drawn
in the page with Observable Plot + d3 + topojson-client from cdn.jsdelivr.net.
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.dirname(HERE)
sys.path.insert(0, SC)
import sca  # noqa: E402
DATA = os.path.dirname(SC)
RES = os.path.join(SC, "results")

GROUP_LABEL = {
    "mia-insular": "MIA & Insular", "dardic-north": "Dardic (north)", "dardic-south": "Dardic (south)",
    "nuristani": "Nuristani", "northwest": "Northwest", "west-pahari": "West Pahari",
    "central-east-pahari-romani": "C/E Pahari & Romani", "hindi": "Hindi belt",
    "rajasthani-gujarati": "Rajasthani–Gujarati", "marathi-bhil": "Marathi–Bhil",
    "bihari-tharu": "Bihari–Tharu", "eastern": "Eastern",
}


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x, nd=4):
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None


def main():
    langs = {r["ID"]: r for r in read(os.path.join(DATA, "cldf", "languages.csv"))}
    summ = {r["Lang"]: r for r in read(os.path.join(RES, "summary.csv"))}
    r1 = {r["Lang"]: r for r in read(os.path.join(RES, "summary-round1.csv"))}
    feats = json.load(open(os.path.join(RES, "features.json"), encoding="utf-8"))
    ext = {r["Lang"]: r for r in read(os.path.join(RES, "extension.csv"))} if os.path.exists(os.path.join(RES, "extension.csv")) else {}
    lects = []
    with open(os.path.join(SC, "languages.tsv"), encoding="utf-8") as f:
        table = list(csv.DictReader(f, delimiter="\t"))
    for L in table:
        lang = L["Lang"]
        if lang == "MIA" or lang not in summ or not summ[lang].get("hand_test_ned"):
            continue
        s, o = summ[lang], r1.get(lang, {})
        g = langs.get(lang, {})
        lects.append({
            "id": lang, "name": L["Name"], "group": L["Group"], "glabel": GROUP_LABEL[L["Group"]],
            "lat": fnum(g.get("Latitude"), 3), "lon": fnum(g.get("Longitude"), 3),
            "etyma": int(L["Etyma"]), "test": int(float(s.get("test_pairs") or 0)),
            "rules": int(float(s["hand_rules"])), "null": fnum(s["null_test_ned"]),
            "hand": fnum(s["hand_test_ned"]), "exact": fnum(s["hand_test_exact"]),
            "r1": fnum(o.get("hand_test_ned")), "r1exact": fnum(o.get("hand_test_exact")),
            "r1rules": int(float(o["hand_rules"])) if o.get("hand_rules") else None,
            "excl": s.get("excluded_sources", ""),
            "ext": fnum(ext.get(lang, {}).get("test_ext_ned")),
            "extch": {k[len("test_chosen_"):]: fnum(v, 3) for k, v in ext.get(lang, {}).items()
                      if k.startswith("test_chosen_") and v not in ("", None)},
            "f": feats["lects"].get(lang, {}).get("f", {}),
            "changes": feats["lects"].get(lang, {}).get("changes", []),
        })

    # order evidence between rule identities ("changes") found in ≥ 8 lects; rules are never pooled by tag
    tag_lects = defaultdict(set)
    for r in read(os.path.join(RES, "tags.csv")):
        if r.get("change"):
            tag_lects[r["change"]].add(r["Lang"])
    orders = []
    for r in read(os.path.join(RES, "order_consensus.csv")):
        a, b = r["first"], r["then"]
        if len(tag_lects[a]) < 8 or len(tag_lects[b]) < 8:
            continue
        orders.append({"a": a, "b": b, "s": int(r["lects_supporting"]), "c": int(r["lects_contradicting"]),
                       "d": fnum(r["sum_delta_ned"], 3),
                       "sl": re.sub(r"\([^)]*\)", "", r["supporting"]).split(),
                       "cl": re.sub(r"\([^)]*\)", "", r["contradicting"]).split()})
    chrono = [{"tag": r["change"], "lects": int(r["lects"]), "med": fnum(r["med"], 3),
               "q1": fnum(r["q1"], 3), "q3": fnum(r["q3"], 3)}
              for r in read(os.path.join(RES, "chronology.csv"))]
    stages = [{"stage": int(r["stage"]), "tag": r["change"], "lects": int(r["lects"]), "after": r["must_follow"]}
              for r in read(os.path.join(RES, "chronology_stages.csv"))]

    # innovation-based subgrouping (innovations.py tree): clades with bootstrap support + per-lect rates
    innov = json.load(open(os.path.join(RES, "innov_clades.json"), encoding="utf-8"))
    ids = {l["id"] for l in lects}
    innov_out = {"reps": innov["reps"], "chars": innov["chars"],
                 "rates": {l: {c: kn for c, kn in r.items() if kn[1]} for l, r in innov["rates"].items() if l in ids},
                 "clades": {k: [{"m": c["members"], "s": c["support"],
                                 "d": [[d["char"], d["in"], d["out"], d["n"]] for d in c["diagnostic"]]}
                                for c in v] for k, v in innov["clades"].items()}}
    # accent score: every lect with accent-placement rules; none / Vedic only / final chain, + its rules
    acchyp = []
    files = {L["Lang"]: L["File"] for L in table}
    for r in read(os.path.join(RES, "accent.csv")):
        path = os.path.join(SC, "rules", f"OIA-{files.get(r['Lang'], '')}.sc")
        if not r.get("train_aned_chain") or not sca.has_chain(path):
            continue
        place = [ln.split(";")[0].strip() for _, ln in sca.lect_lines(path)
                 if "+stress" in ln.split(";")[0] or "-stress" in ln.split(";")[0]]
        if not place:
            continue
        acchyp.append({"id": r["Lang"], "n": int(r["train_apairs"]) + int(r["test_apairs"]), "rules": place,
                       "v": {m: [fnum(r[f"train_aned_{m}"]), fnum(r[f"test_aned_{m}"])] for m in ("none", "vedic-only", "chain")}})
    pt = json.load(open(os.path.join(RES, "prefix_tree.json"), encoding="utf-8"))
    changes = json.load(open(os.path.join(RES, "changes.json"), encoding="utf-8"))
    ptree = {"tree": pt["tolerant"], "tol": pt["tol"]}
    accent = json.load(open(os.path.join(HERE, "accent_tests.json"), encoding="utf-8"))
    accent.pop("_doc", None)

    with open(os.path.join(HERE, "assets", "countries-50m.json"), encoding="utf-8") as f:
        world = f.read()
    blobs = {
        "LECTS": json.dumps(lects, ensure_ascii=False),
        "FEATS": json.dumps(feats["features"], ensure_ascii=False),
        "ORDERS": json.dumps(orders, ensure_ascii=False, separators=(",", ":")),
        "CHRONO": json.dumps(chrono, ensure_ascii=False),
        "STAGES": json.dumps(stages, ensure_ascii=False),
        "GROUPS": json.dumps(GROUP_LABEL, ensure_ascii=False),
        "WORLD": world,
        "INNOV": json.dumps(innov_out, ensure_ascii=False, separators=(",", ":")),
        "ACCENT": json.dumps(accent, ensure_ascii=False),
        "ACCHYP": json.dumps(acchyp, ensure_ascii=False),
        "PTREE": json.dumps(ptree, ensure_ascii=False, separators=(",", ":")),
        "CHANGES": json.dumps(changes, ensure_ascii=False, separators=(",", ":")),
    }
    frag_path = os.path.join(HERE, "fragments2.json")
    frags = json.load(open(frag_path, encoding="utf-8")) if os.path.exists(frag_path) else {}
    frags.update({"PT_NSHARED": str(pt["n_shared"]), "PT_HOMO": str(len(pt["homoplasy"])), "PT_TOL": str(pt["tol"])})

    with open(os.path.join(HERE, "template2.html"), encoding="utf-8") as f:
        page = f.read()
    for k, v in blobs.items():
        page = page.replace("/*{{" + k + "}}*/null", v)
    for k, v in frags.items():
        page = page.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{[A-Z_0-9]+\}\}", page)
    assert not left, left
    with open(os.path.join(HERE, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {os.path.join(HERE, 'index.html')} ({len(page) / 1e6:.2f} MB, {len(lects)} lects)")


if __name__ == "__main__":
    main()
