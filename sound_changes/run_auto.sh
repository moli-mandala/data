#!/bin/sh
# Greedy induction baseline for every language in languages.tsv (sequential, single-threaded).
# Skips lects that already have a curve unless --force.
cd "$(dirname "$0")/.." || exit 1
tail -n +2 sound_changes/languages.tsv | cut -f1,2 | while IFS="$(printf '\t')" read -r L S; do
  [ -f "sound_changes/results/OIA-$S.auto.frontier.csv" ] && [ "$1" != "--force" ] && continue
  python3 sound_changes/evaluate.py induce --pairs "sound_changes/pairs/OIA-$S.tsv" \
    --out "sound_changes/rules/auto/OIA-$S.auto.sc" \
    --curve "sound_changes/results/OIA-$S.auto.frontier.csv" --max-rules 60 \
    > "sound_changes/results/logs/OIA-$S.induce.log" 2>&1
  echo "$(date +%H:%M:%S) $L $(tail -1 sound_changes/results/OIA-$S.auto.frontier.csv | cut -d, -f1,3)"
done
echo DONE
