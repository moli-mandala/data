# Consolidated integration proposal — ESR 2015-012 Noira

These source-specific changes are now applied. This document remains the exact
integration contract; consolidated build, global audit, browser, and shipping
gates remain deferred.

## Proposed bibliography

```bibtex
@article{varghesekumar2015noira,
  author  = {Varghese, Bezily P. and Kumar D., Sunil},
  title   = {Noira Bhils and a Few Other Groups: A Sociolinguistic Study},
  journal = {SIL Electronic Survey Reports},
  volume  = {2015-012},
  year    = {2015},
  pages   = {1--90},
  url     = {https://www.sil.org/resources/archives/69229},
  note    = {Appendix A3 manually reviewed cell by cell; canonical PDF SHA-256 cb93db089a21e55e878f436632d8282c64c98fca85afe18179f8f3383db35280}
}
```

Use citation suffix
`[Appendix A3, printed p. N, item I, list CODE]` exactly as staged.

## Proposed language rows

`Noiri` and `DungraBhili` are also proposed by the ESR 2013-004 package; add
each parent only once. Glottolog identifies Dungra Bhil as `dung1251` and the
existing Dhule proposal identifies Noiri as `noir1238`. No new Kotli parent is
proposed: apply the standing-policy mapping below as a provisional
source-supported dialect route under Noiri.

```csv
Noiri,Noiri,noir1238,,,Bhil,"Northern Maharashtra and adjacent Madhya Pradesh; source parent for Noiri and Barutiya regional lists",C
DungraBhili,Dungra Bhil,dung1251,,,Bhil,"Western India; source parent for the Mathwad and Ambadungar Dungra Bhili lists",C
```

Reuse existing parents `Goj` (Gujari), `ko` (Korku), and `Ni` (Nihali).
The primary report calls Kotli a reported Noiri dialect, publishes two named
Kotli wordlists, and later concludes that Kotli has a distinctive identity
requiring further research. Provisionally route both sites under canonical
Noiri, preserve the source labels and uncertainty, leave dialect
Glottocode/coordinates blank, and do not equate them with historical
Kotali/Khandesi. This is not a genealogical determination.

## Proposed dialect rows

```csv
sil-noira-2015-noiri-chillare,dialect:Noiri:sil-noira-2015-noiri-chillare:Chillare,Noiri,sil-noira-2015-noiri-chillare,Chillare,noir1238,,,Bhil,"Noiri wordlist, Chillare, Shirpur tahsil, Dhule district, Maharashtra",C
sil-noira-2015-noiri-pannali,dialect:Noiri:sil-noira-2015-noiri-pannali:Pannali,Noiri,sil-noira-2015-noiri-pannali,Pannali,noir1238,,,Bhil,"Noiri wordlist, Pannali, Pansemal tahsil, Madhya Pradesh",C
sil-noira-2015-noiri-gomon,dialect:Noiri:sil-noira-2015-noiri-gomon:Gomon,Noiri,sil-noira-2015-noiri-gomon,Gomon,noir1238,,,Bhil,"Noiri wordlist, Gomon, Akkalkua tahsil, Nandurbar district, Maharashtra",C
sil-noira-2015-dungra-bhili-mathwad,dialect:DungraBhili:sil-noira-2015-dungra-bhili-mathwad:Mathwad,DungraBhili,sil-noira-2015-dungra-bhili-mathwad,Mathwad,dung1251,,,Bhil,"Dungra Bhili wordlist, Mathwad, Alirajpur, Madhya Pradesh",C
sil-noira-2015-dungra-bhili-ambadungar,dialect:DungraBhili:sil-noira-2015-dungra-bhili-ambadungar:Ambadungar,DungraBhili,sil-noira-2015-dungra-bhili-ambadungar,Ambadungar,dung1251,,,Bhil,"Dungra Bhili wordlist, Ambadungar, Kawant tahsil, Gujarat",C
sil-noira-2015-kotli-narayanpur,dialect:Noiri:sil-noira-2015-kotli-narayanpur:Narayanpur,Noiri,sil-noira-2015-kotli-narayanpur,Narayanpur,,,,Bhil,"Kotli wordlist, Papiner Narayanpur; provisional source-supported Noiri dialect routing; distinctive identity requires further research; not equated with historical Kotali/Khandesi",C
sil-noira-2015-kotli-taradi,dialect:Noiri:sil-noira-2015-kotli-taradi:Taradi,Noiri,sil-noira-2015-kotli-taradi,Taradi,,,,Bhil,"Kotli wordlist, Taradi; respondent label Adivasi Bhil-Taradi retained; provisional source-supported Noiri dialect routing; not equated with historical Kotali/Khandesi",C
sil-noira-2015-gujari-taradi,dialect:Goj:sil-noira-2015-gujari-taradi:Taradi,Goj,sil-noira-2015-gujari-taradi,Taradi,guja1253,,,Rajasthanic,"Gujari wordlist, Taradi, Shahada tahsil, Maharashtra",C
sil-noira-2015-korku-tembhi,dialect:ko:sil-noira-2015-korku-tembhi:Tembhi,ko,sil-noira-2015-korku-tembhi,Tembhi,kork1243,,,Munda,"Source labels this Nihali-Tembhi; report concludes the community has shifted to Korku",C
sil-noira-2015-korku-tukaithad,dialect:ko:sil-noira-2015-korku-tukaithad:Tukaithad,ko,sil-noira-2015-korku-tukaithad,Tukaithad,kork1243,,,Munda,"Korku wordlist, Tukaithad, Khaknar block, Madhya Pradesh",C
sil-noira-2015-nihali-jamod,dialect:Ni:sil-noira-2015-nihali-jamod:Jamod,Ni,sil-noira-2015-nihali-jamod,Jamod,niha1238,,,Nihali,"Nihali wordlist, Jamod-Jalgaon, Buldana district, Maharashtra",C
```

## Republication and exclusion policy

The report's table 4 and prose identify Astambha, Mundalwad/Mutalwad, and
Toranmal as Dhule-team wordlists. They are the ESR 2013-004 source lists and
must not be installed a second time. Reuse the earlier dialect identities
`sil-dhule-2013-noiri-astamba` and
`sil-dhule-2013-noiri-mundalwad`; Toranmal remains comparison-only in that
package. `dhule_republication_reconciliation.tsv` accounts for all 630 cells.
The three literal-ledger-exact and 627 representation-different labels in that
crosswalk reflect that the Dhule audit embeds printed similarity labels in its
manual field while Noira stores them separately. They are not treated as
lexical disagreements and never verify a Noira reading; the report's explicit
source-team/list identity establishes the republication exclusion.

Gujarati, Marati, and Hindi are also audit-only controls. In total, exclude
1,260 conceptual cells / 1,671 printed responses while retaining their exact
source evidence in `exhaustive_audit.tsv`.

## Proposed raw-form and profile routing

- Copy source-local `staged_forms.csv` byte-for-byte to
  `data/other/forms/20260828-sil-noira.csv` (2,714 rows; no header; SHA-256
  `c82983a319d6d6fbf5c07063f0655ae3e4e8e3890d625e1bfc2a38f95c811746`).
- Install `conversion_profile.tsv` as `conversion/sil-noira.txt`.
- Add exact source-key route `varghesekumar2015noira -> sil-noira`; do not
  route other SIL sources through it implicitly.
- Keep `Source_Cognate_Labels` in the audit only. Do not interpret similarity
  numbers as database cognateset assignments.
- Map the 210 standard source glosses during the consolidated build; inspect
  and resolve any unmapped or multi-mapped parameters rather than changing the
  diplomatic source glosses.

## Source-local validation and remaining shared work

```sh
python3 data/data/other/forms/raw_data/sil_noira_2015/import_noira_2015.py \
  --all --write --pdf tmp/pdfs/noira_2015/silesr2015_012.pdf
UV_CACHE_DIR=/tmp/uv-cache uv run --project data python -m pytest -q \
  data/data/other/forms/raw_data/sil_noira_2015/test_preintegration_contract.py \
  data/data/other/forms/raw_data/sil_noira_2015/manual_chunks/test_items_*_hand_keyed.py
```

Expected counts: 3,570 reviewed cells; 3,526 attested; 44 source blanks;
4,385 expanded responses; 2,714 staged forms; 834 republished Dhule responses
excluded; 837 language-control responses excluded; zero unresolved.

The relevant focused source-local and shared registry/reference/profile/parser
tests are now run. The consolidated `make all`, generated diff/error review,
full pytest and graph validation, global source-audit regeneration, browser
database refresh with representative checks from each of the eleven installed
lists, and commit remain expressly deferred from this lane.
