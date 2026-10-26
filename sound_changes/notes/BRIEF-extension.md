# Brief, round 4: hill-climb the extension-aware score

The constraints are the same as before:
- write only your lects' rule files, your notes file and `tmp/sound_changes/<group>/`;
- do not edit the engine; ask the coordinator if you need an engine change;
- run one Python process at a time;
- tune on **train** only;
- never use a broad `pkill`: kill only the PIDs you started.

## Why
About 54% of the remaining edit operations fall in the word ending (a uniform spread would put
42% there). The largest single cause is the choice of stem extension and gender ending:
- -∅ vs -ā, -ī, -o, -a, -i, -u (both directions);
- -∅ → -rā / -rī / -lā / -ro.

The plain NED cannot credit a chain for deriving *ghōṭaka > ghoṛā correctly when the pair is
filed under ghōṭa.

## The score
`python3 sound_changes/evaluate.py score RULES --ext` prints an **extension-aware NED** for train
and test.
- The headword is also run through your chain with each extension of a fixed menu, replacing its
  final vowel: -aka, -ikā, -uka, -aḍaka, -allaka, -araka (`evaluate.EXT_MENU`).
- Verbs are never extended.
- Each attested reflex is scored against its closest variant. That is one choice per reflex,
  because several endings often coexist.
- No cost is charged for using an extension, so the score is optimistic. It is reported next to
  the plain NED, never instead of it.
- The output also shows the share of reflexes choosing each ending. For Hindi, test NED goes from
  0.359 to 0.268 extension-aware, with bare 62%, aka 22%, ikā 10%.

`score --ext-loo` gives each rule's leave-one-out value on the train extension-aware NED. It is
slow, so run it once per lect, not in a loop.

## What to do (until 14:40)
1. For each lect, largest first, look at how your chain treats the extended variants:
   - does -aka come out as the lect's masculine ending (-ā, -o, -u, -au, -ɔ)?
   - does -ikā give -ī / -i, -uka give -ū / -u, -aḍaka give -ṛā / -ṛo, -allaka give -lā / -lo,
     and -araka give -rā / -ro?

   Fix the chain where it mangles these: the usual problems are intervocalic k loss followed by
   vowel contraction, and shortening or apocope ordered wrongly.
2. Check how your `@morph` citation rules interact with the variants. For example,
   `if:starred a > o / _ #` also fires on extended variants and can double an ending.

   Keep a rule if the plain NED needs it; otherwise restrict it (e.g. to un-extended forms,
   through its environment).
3. Adopt only changes that meet all three conditions:
   - they improve train extension-aware NED;
   - they do not worsen plain train NED by more than 0.001, and do not change the accent score
     for lects that have one;
   - they read as sound changes or as regular morphology.

   Tag them as usual (`@aka`, `@morph`, and so on).

## Reporting
Stop editing at **14:40**. Rerun `frontier` for every lect you changed. Then add a section
**"§10 Extension-aware score"** to your notes, giving:
- per lect: plain NED and extension-aware NED (train and test), before and after;
- the ending-choice shares per lect;
- the rules you changed, with their LOO values;
- what the choices say, e.g. which lects prefer -aka over the bare stem, and where -ḍa, -la and
  -ra extensions are productive.

Be done by **14:55**. Your final message should be at most 150 words.

## Also: should the menu be expanded? (question from the user)
In §10, say whether the fixed menu should gain or lose items for your lects. Do not edit the menu;
the coordinator will decide. Back each proposal with evidence:
- the most frequent ending mismatches that remain under the extension-aware score (the
  `score --ext` choices, plus your own residual inspection);
- for each candidate (e.g. -iya/-īya, -ikka, plain feminine -ī, -ālu/-āla, -ṭa/-ḍa without -ka,
  -ra without -ka, Dardic -ī/-u endings), roughly how many train reflexes it would fix, and
  whether it is real, well-attested MIA/NIA morphology or just a surface fix;
- menu items your lects never choose.

## Menu change (13:29)
The menu now also contains **-akka** (geminate -kk-, as in Pk -kka / Nuristani -k), per the
Nuristani agent's recommendation: `EXT_MENU = ("aka", "akka", "ikā", "uka", "aḍaka", "allaka",
"araka")`. Rescore your before/after numbers with the new menu.

## Menu and engine after round 4 (coordinator)
- Menu now: aka, akka, ikā, ikka, iyaka, uka, aḍaka, allaka, araka, akam (added -ikka, NIA -iyā
  from -iyaka, neuter -akam for Gujarati -ũ / Marathi -ẽ, per the groups' §10 recommendations).
- Extended variants carry the word flag `ext`; word-final @morph citation rules got `!ext` where
  that improved the extension-aware train NED (plain NED cannot change).
