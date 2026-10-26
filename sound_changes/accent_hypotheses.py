#!/usr/bin/env python3
# run from data/: python sound_changes/accent_hypotheses.py LECT_FILE_STEM   (results collected in results/accent_hypotheses.csv)
"""Score accent-placement hypotheses for one lect: each inserts `$1+stress` rules into a scratch
copy of the chain; the accent-blind NED must not move."""
import sys, subprocess, re, os, tempfile
lang = sys.argv[1] if len(sys.argv) > 1 else "Sh"
sys.path.insert(0, "sound_changes"); import sca  # noqa: E402
src = sca.lect_text(f"sound_changes/rules/OIA-{lang}.sc").split("\n")   # a one-lect scratch copy is edited below
code = lambda l: l.split(";")[0].strip()
fvl = [i for i, l in enumerate(src) if "@final-vowel-loss" in l and code(l) and not l.lstrip().startswith("%")]
last_chain = max(i for i, l in enumerate(src) if code(l) and not code(l).startswith(("%", "class")) and not any(
    src[j].startswith("%source") for j in range(i, -1, -1) if src[j].startswith(("%source", "%end")) and src[j].startswith("%source")
    and all(not src[k].startswith("%end") for k in range(j, i))))
apoc = fvl[-1] if fvl else last_chain + 1
V = {
    "none (chain)": ("chain", []),
    "Vedic accent carried": ("vedic", []),
    "penult, before final-vowel loss": ("chain", [(apoc, "V > $1+stress / _ C* V C* #")]),
    "first syllable (end of chain)": ("chain", [(last_chain + 1, "V > $1+stress / # C* _")]),
    "last syllable (end of chain)": ("chain", [(last_chain + 1, "V > $1+stress / _ C* #")]),
    "first long vowel, else first": ("chain", [(last_chain + 1, "Long > $1+stress / # !Long* _"),
                                              (last_chain + 2, "Short > $1+stress / # C* _ // _ !#* Long")]),
    "Vedic, else penult": ("vedic", [(apoc, "V > $1+stress / _ C* V C* # // _ !#* Acc")]),
}
d = tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "/tmp"))
for name, (model, ins) in V.items():
    lines = list(src)
    for pos, txt in sorted(ins, reverse=True):
        lines.insert(pos, txt + "   ; @stress")
    p = f"{d}/OIA-{lang}.sc"
    open(p, "w", encoding="utf-8").write("\n".join(lines))
    out = subprocess.run(["python3", "sound_changes/evaluate.py", "score", p, "--pairs", f"sound_changes/pairs/OIA-{lang}.tsv",
                          "--errors", "0", "--accent-model", model], capture_output=True, text=True)
    if out.returncode:
        print(name, out.stderr[-300:]); continue
    rows = [l for l in out.stdout.split("\n") if "rules  NED" in l]
    g = lambda l: re.findall(r"NED ([0-9.]+)", l) + re.findall(r"exact ([0-9.]+)%", l)[1:]
    tr, te = g(rows[0]), g(rows[1])
    print(f"{name:34s} NED {tr[0]}/{te[0]}  accent NED train {tr[1]} test {te[1]}  accent-exact test {te[2]}%")
