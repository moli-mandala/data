#!/usr/bin/env python3
"""Assemble sound_changes/report/index.html from template.html + results/*.csv + plots/*.png.

    uv run --group plots python sound_changes/report/build.py

Placeholders in template.html ({{NAME}}) are filled with generated HTML fragments; figures are
embedded as data: URIs so the page is self-contained.
"""
import base64
import csv
import html
import json
import os
import re
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.dirname(HERE)
RES = os.path.join(SC, "results")
PLOTS = os.path.join(SC, "plots")

GROUP_LABEL = {
    "mia-insular": "MIA & Insular", "dardic-north": "Dardic (north)", "dardic-south": "Dardic (south)",
    "nuristani": "Nuristani", "northwest": "Northwest", "west-pahari": "West Pahari",
    "central-east-pahari-romani": "C/E Pahari & Romani", "hindi": "Hindi belt",
    "rajasthani-gujarati": "Rajasthani–Gujarati", "marathi-bhil": "Marathi–Bhil",
    "bihari-tharu": "Bihari–Tharu", "eastern": "Eastern",
}
GROUP_ORDER = list(GROUP_LABEL)


def esc(s):
    return html.escape(str(s), quote=True)


def img(name, alt, cls="plate"):
    path = os.path.join(PLOTS, name)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<figure class="{cls}"><img src="data:image/png;base64,{b64}" alt="{esc(alt)}"></figure>'


def read(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f3(x):
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return ""


def stats(summary):
    rows = [r for r in summary if r.get("hand_test_ned")]
    nulls = sorted(float(r["null_test_ned"]) for r in rows)
    hands = sorted(float(r["hand_test_ned"]) for r in rows)
    med = lambda v: v[len(v) // 2]
    pairs = [(float(r["hand_test_ned_at_k"]), float(r["auto_test_ned_at_k"])) for r in rows
             if r.get("hand_test_ned_at_k") and r.get("auto_test_ned_at_k")]
    wins = sum(h < a for h, a in pairs)
    gaps = sorted(a - h for h, a in pairs)
    return {
        "N_LECTS": len(rows),
        "N_IA": sum(r["Group"] != "nuristani" for r in rows),
        "N_RULES": sum(int(float(r["hand_rules"])) for r in rows),
        "MED_NULL": f"{med(nulls):.3f}", "MED_HAND": f"{med(hands):.3f}",
        "WINS": wins, "N_CMP": len(pairs), "MED_GAP": f"{med(gaps):.3f}",
        "N_PAIRS": sum(int(r["Pairs"]) for r in rows),
    }


def lect_table(summary):
    data = []
    for r in summary:
        if not r.get("hand_test_ned"):
            continue
        data.append({
            "lang": r["Lang"], "name": r["Name"], "group": GROUP_LABEL.get(r["Group"], r["Group"]),
            "gkey": r["Group"], "etyma": int(r["Etyma"]), "test": int(float(r["test_pairs"] or 0)),
            "rules": int(float(r["hand_rules"])), "null": round(float(r["null_test_ned"]), 3),
            "hand": round(float(r["hand_test_ned"]), 3),
            "exact": round(100 * float(r["hand_test_exact"]), 1),
            "k": int(float(r["compare_n"])) if r.get("compare_n") else None,
            "handk": round(float(r["hand_test_ned_at_k"]), 3) if r.get("hand_test_ned_at_k") else None,
            "autok": round(float(r["auto_test_ned_at_k"]), 3) if r.get("auto_test_ned_at_k") else None,
            "excl": r.get("excluded_sources", ""),
        })
    return json.dumps(data, ensure_ascii=False)


def group_options():
    return "\n".join(f'<option value="{k}">{esc(v)}</option>' for k, v in GROUP_LABEL.items())


def stages_html():
    rows = read("chronology_stages.csv")
    by = defaultdict(list)
    for r in rows:
        by[int(r["stage"])].append(r)
    out = []
    for st in sorted(by):
        chips = "".join(
            f'<span class="chip" title="{esc(r["must_follow"] or "no strongly supported predecessor")}">'
            f'{esc(r["tag"].lstrip("@"))}<small>{r["lects"]}</small></span>'
            for r in sorted(by[st], key=lambda r: -int(r["lects"])))
        out.append(f'<div class="stage"><div class="stage-n">{st}</div><div class="chips">{chips}</div></div>')
    return "\n".join(out)


def consensus_table(limit=28):
    rows = read("order_consensus.csv")
    tags = defaultdict(set)
    for r in read("tags.csv"):
        tags[r["tag"]].add(r["Lang"])
    keep = []
    for r in rows:
        a, b = r["first"], r["then"]
        if len(tags[a]) < 15 or len(tags[b]) < 15:
            continue
        keep.append(r)
        if len(keep) >= limit:
            break
    tr = "\n".join(
        f'<tr><td><code>{esc(r["first"].lstrip("@"))}</code></td><td>before</td>'
        f'<td><code>{esc(r["then"].lstrip("@"))}</code></td><td class="num">{r["lects_supporting"]}</td>'
        f'<td class="num">{r["lects_contradicting"]}</td><td class="num">{float(r["sum_delta_ned"]):.2f}</td></tr>'
        for r in keep)
    return tr


def key_orders():
    """Hand-picked orders discussed in the text, with support by group."""
    rows = {(r["first"], r["then"]): r for r in read("order_consensus.csv")}
    want = [
        ("@int-stop-loss", "@degem-plain", "Loss of single intervocalic stops happened while MIA geminates still protected their stops."),
        ("@int-aspirate-h", "@degem-plain", "Aspirates became h only between vowels, before geminates simplified."),
        ("@sibilant-s", "@s-stop-aspirate", "The sibilant merger fed s + stop → aspirated geminate (śuṣka > sukkʰa)."),
        ("@palatal-Cy", "@assim-sonorant", "ty dy → cc jj applied first and bled the general C + y assimilation."),
        ("@int-stop-loss", "@cluster-simplify", "Dardic and Nuristani: clusters simplified (without geminating) only after lenition."),
        ("@degem-lengthen", "@final-vowel-loss", "Compensatory lengthening needed the old disyllable, so it precedes apocope."),
        ("@flap", "@degem-plain", "Flap first (west): old ḍḍ stays ḍ. The reverse order (Bihari, Bengali, Hindi-Urdu) flaps it too."),
        ("@final-vowel-loss", "@hiatus", "A real split: apocope before hiatus contraction in the west and Dardic, after it in the east."),
    ]
    out = []
    for a, b, note in want:
        r = rows.get((a, b))
        rev = False
        if not r:
            r = rows.get((b, a)); rev = True
        if not r:
            continue
        sup = Counter(re.findall(r"\(([^)]+)\)", r["supporting"]))
        con = Counter(re.findall(r"\(([^)]+)\)", r["contradicting"]))
        s, c = (int(r["lects_supporting"]), int(r["lects_contradicting"]))
        if rev:
            s, c, sup, con = c, s, con, sup
        spread = ", ".join(f"{GROUP_LABEL.get(g, g)} {n}" for g, n in sup.most_common())
        against = ", ".join(f"{GROUP_LABEL.get(g, g)} {n}" for g, n in con.most_common()) or "none"
        out.append(f'<tr><td><code>{esc(a.lstrip("@"))}</code> → <code>{esc(b.lstrip("@"))}</code></td>'
                   f'<td class="num">{s}</td><td class="num">{c}</td>'
                   f'<td><span class="note">{esc(note)}</span><br><small>for: {esc(spread)}; against: {esc(against)}</small></td></tr>')
    return "\n".join(out)


def clusters_html():
    with open(os.path.join(RES, "clusters.txt"), encoding="utf-8") as f:
        txt = f.read()
    m = re.search(r"== 16 clusters.*?\n(.*?)\n\n", txt, re.S)
    if not m:
        return ""
    names = {r["Lang"]: r["Name"] for r in read("summary.csv")}
    items = []
    for line in m.group(1).splitlines():
        _, members = line.split(":", 1)
        ms = members.split()
        label = ", ".join(names.get(x, x) for x in ms)
        items.append(f"<li><strong>{len(ms)}</strong> {esc(label)}</li>")
    return "\n".join(items)


def main():
    summary = read("summary.csv")
    summary = [r for r in summary if r["Lang"] != "MIA"]
    with open(os.path.join(HERE, "template.html"), encoding="utf-8") as f:
        page = f.read()
    fills = {k: str(v) for k, v in stats(summary).items()}
    fills.update({
        "LECT_DATA": lect_table(summary),
        "GROUP_OPTIONS": group_options(),
        "STAGES": stages_html(),
        "CONSENSUS_ROWS": consensus_table(),
        "KEY_ORDERS": key_orders(),
        "CLUSTERS": clusters_html(),
        "FIG_SCORES": img("family-scores.png", "Test NED by group: no rules, greedy baseline at the same rule count, hand-written chain"),
        "FIG_CHRONO": img("family-chronology.png", "Median relative position of each tagged change in its chain, with interquartile range"),
        "FIG_ISO": img("family-isoglosses.png", "Maps of South Asia showing which lects have each of twelve changes"),
        "FIG_HEAT": img("family-heatmap.png", "Lects by tagged changes, ordered by hierarchical clustering", cls="plate tall"),
        "FIG_PK": img("OIA-Pk.frontier.png", "Prakrit: rules vs test NED and exact match, hand-written vs greedy"),
    })
    # hand-written fragments for groups that finished last (e.g. dardic-north)
    frag = os.path.join(HERE, "fragments.json")
    if os.path.exists(frag):
        with open(frag, encoding="utf-8") as f:
            fills.update(json.load(f))
    for k, v in fills.items():
        page = page.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{[A-Z_]+\}\}", page)
    assert not left, left
    with open(os.path.join(HERE, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {os.path.join(HERE, 'index.html')} ({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
