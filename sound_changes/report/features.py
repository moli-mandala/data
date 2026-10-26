#!/usr/bin/env python3
"""Derive per-lect categorical features from the tagged rule files, for the report's maps.

Each feature reads the rules carrying a given tag and summarises *what* the change does in that
lect (its outcome or conditioning), not just whether it is present. The heuristics are simple and
documented per feature; every value keeps the rule text it came from, so the report can show it.

    python sound_changes/report/features.py   → results/features.json
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.dirname(HERE)
sys.path.insert(0, SC)
import synth  # noqa: E402


def rhs(rule: str) -> str:
    body = re.sub(r"^(repeat:\s*|if:\S+\s*)+", "", rule)
    return body.split(">", 1)[1].split("/", 1)[0].strip() if ">" in body else ""


def lhs(rule: str) -> str:
    body = re.sub(r"^(repeat:\s*|if:\S+\s*)+", "", rule)
    return body.split(">", 1)[0].strip()


def env(rule: str) -> str:
    body = rule.split("//", 1)[0]
    return body.split("/", 1)[1].strip() if "/" in body.split(">", 1)[-1] else ""


def with_tag(chain, *tags):
    return [(r, t) for r, t, _ in chain if set(t) & set(tags)]


FEATURES = {}


def feature(key, label, note):
    def deco(fn):
        FEATURES[key] = (label, note, fn)
        return fn
    return deco


@feature("cl", "Compensatory lengthening",
         "Where degemination lengthens the preceding vowel, read from the lengthening rule's environment.")
def f_cl(chain):
    rs = with_tag(chain, "@degem-lengthen")
    if not rs:
        return ("no lengthening", "") if with_tag(chain, "@degem-plain") else ("no geminates", "")
    r = rs[0][0].text
    e = env(r).split("|")[0].replace(" ", "")
    left, _, right = e.partition("_")
    lh, rh = "#" in left, "#" in right
    if lh and rh:
        return "disyllables only", r        # first syllable = penult: # (C) _ V (C) #
    if lh:
        return "first syllable", r
    if rh:
        return "penultimate", r
    return "other / unconditioned", r


@feature("sib", "Sibilants", "The sibilant merger rule's output (s, ś, or partial), or none.")
def f_sib(chain):
    rs = with_tag(chain, "@sibilant-s", "@sibilant-ś", "@sibilant-keep")
    if not rs:
        return "three kept / no rule", ""
    for r, t in rs:
        out, inp = rhs(r.text), lhs(r.text)
        if "@sibilant-s" in t and out == "s":
            return ("s (ś ṣ > s)" if "ś" in inp or "Sib" in inp else "partial > s"), r.text
        if "@sibilant-ś" in t:
            return ("ś (all > ś)" if "s" in inp.replace("ś", "").replace("ṣ", "") or "Sib" in inp
                    else "ṣ > ś, s kept"), r.text
    return "kept", rs[0][0].text


@feature("rv", "Vocalic r̩", "Default (least conditioned) outcome of r̩.")
def f_rv(chain):
    rs = with_tag(chain, "@r-vocalic", "@r-vocalic-ri")
    if not rs:
        return "no rule", ""
    plain = [r for r, _ in rs if not env(r.text)] or [rs[-1][0]]
    out = rhs(plain[0].text).replace(" ", "")
    return (out or "∅"), plain[0].text


@feature("ks", "kṣ", "Output of the kṣ cluster rule.")
def f_ks(chain):
    for r, _ in with_tag(chain, "@ks"):
        L = lhs(r.text)
        if ("k ṣ" in L) or ("k s" in L) or L.startswith("{k"):
            out = rhs(r.text)
            if out.startswith("{"):
                out = re.findall(r"\{([^}]*)\}", out)
                out = "".join(x.split()[0] for x in out) if out else "?"
            return out.replace(" ", "") or "∅", r.text
    return "no rule", ""


@feature("intstop", "Intervocalic stops", "Loss, voicing only, or retention of single intervocalic stops.")
def f_int(chain):
    if with_tag(chain, "@int-stop-loss"):
        return "lost", with_tag(chain, "@int-stop-loss")[0][0].text
    if with_tag(chain, "@int-voicing"):
        return "voiced", with_tag(chain, "@int-voicing")[0][0].text
    return "kept", ""


@feature("clusters", "Cluster treatment", "Assimilation to geminates (MIA type) vs simplification without geminates.")
def f_clu(chain):
    a = bool(with_tag(chain, "@assim-stop", "@assim-sonorant", "@assim-r"))
    s = bool(with_tag(chain, "@cluster-simplify"))
    return ("geminates" if a and not s else "simplify" if s and not a else "both" if a else "neither"), ""


@feature("flap", "Intervocalic ḍ", "Output of the flap rule (ṛ, r, ḷ, l) or none.")
def f_flap(chain):
    for r, _ in with_tag(chain, "@flap"):
        out = rhs(r.text)
        m = re.findall(r"[ṛrḷl]", out)
        if m:
            return m[0], r.text
    return "ḍ kept / no rule", ""


@feature("nasvoice", "Post-nasal voicing", "nt > nd etc.")
def f_nv(chain):
    r = with_tag(chain, "@nasal-voicing")
    return ("yes", r[0][0].text) if r else ("no", "")


@feature("sh", "s > h", "Debuccalisation of s, by conditioning environment (each is a different change).")
def f_sh(chain):
    # the conditioning is the feature: unconditioned s > h and intervocalic s > h are different changes
    r = [x for x in with_tag(chain, "@s-h") if re.match(r"^\{?[sśṣ ]+\}? >", lhs(x[0].text) + " >")]
    if not r:
        return ("no", "")
    e = env(r[0][0].text)
    lab = {"": "everywhere", "V _ V": "between vowels", "V _": "after a vowel", "# _": "word-initial",
           "_ V": "before a vowel", "# _ V | V _ V": "initially and between vowels"}.get(e, e or "everywhere")
    return (f"s > h {lab}", r[0][0].text)


@feature("nasal", "Nasal + stop", "Anusvāra → nasal vowel vs nasal consonant kept.")
def f_nas(chain):
    if with_tag(chain, "@anusvara-vowel"):
        return "nasal vowel", with_tag(chain, "@anusvara-vowel")[0][0].text
    if with_tag(chain, "@nasal-cluster"):
        return "anusvāra only", with_tag(chain, "@nasal-cluster")[0][0].text
    return "cluster kept", ""


@feature("apocope", "Final short vowels", "Which final vowels are lost.")
def f_apo(chain):
    rs = with_tag(chain, "@final-vowel-loss")
    if not rs:
        return "kept", ""
    vs = set()
    for r, _ in rs:
        vs |= set(re.findall(r"[aiu]", lhs(r.text).replace("V", "aiu")))
    lab = "/".join(v for v in "aiu" if v in vs) or "some"
    return f"-{lab} lost", rs[0][0].text


@feature("vb", "Initial v-", "v- > b-.")
def f_vb(chain):
    r = with_tag(chain, "@v-b")
    return ("b-", r[0][0].text) if r else ("v- kept", "")


@feature("asp", "Aspirates", "Deaspiration / tone.")
def f_asp(chain):
    if with_tag(chain, "@tone"):
        return "tone", with_tag(chain, "@tone")[0][0].text
    r = with_tag(chain, "@deaspiration")
    if r:
        return "deaspiration", r[0][0].text
    return "kept", ""


@feature("lret", "l > ḷ / l > r", "Intervocalic l.")
def f_l(chain):
    for r, t in chain_rules(chain):
        L, R = lhs(r.text), rhs(r.text)
        if L == "l" and R in ("ḷ", "r"):
            return f"l > {R}", r.text
    return "l kept", ""


@feature("stress", "Accent-conditioned rules", "Rules tagged @stress (conditioned on the Vedic accent).")
def f_st(chain):
    r = with_tag(chain, "@stress")
    return (f"{len(r)} rule(s)", r[0][0].text) if r else ("none", "")


def chain_rules(chain):
    return [(r, t) for r, t, _ in chain]


def main():
    out = {"features": {k: {"label": v[0], "note": v[1]} for k, v in FEATURES.items()}, "lects": {}}
    for L in synth.languages():
        rp = synth.rules_path(L["File"])
        if not os.path.exists(rp):
            continue
        chain = synth.rule_tags(rp)
        vals = {}
        for key, (_, _, fn) in FEATURES.items():
            v, rule = fn(chain)
            vals[key] = [v, rule]
        # "changes": the lect's phonological rules as rule identities (no pooling by tag)
        changes = sorted({synth.change_id(r) for r, ts, _ in chain if synth.is_phon(ts, r.text)})
        out["lects"][L["Lang"]] = {"f": vals, "changes": changes}
    path = os.path.join(SC, "results", "features.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"wrote {path}: {len(out['lects'])} lects × {len(FEATURES)} features")


if __name__ == "__main__":
    main()
