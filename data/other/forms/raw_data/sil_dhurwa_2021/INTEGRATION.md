# Future shared integration proposal — JLSR 2021-034 Dhurwa

The source-local review is exhaustive and the complete profile validates all 809 target forms. Apply this proposal only in the root integration lane.

## Proposed bibliography

```bibtex
@article{josephmichael2021dhurwa,
  author  = {Joseph, D. Selwyn and Michael, Selvi},
  title   = {A Sociolinguistic Survey among the Dhurwa of Madhya Pradesh and Orissa},
  journal = {Journal of Language Survey Reports},
  volume  = {2021-034},
  year    = {2021},
  pages   = {1--19},
  url     = {https://www.sil.org/resources/archives/89899},
  note    = {Data collected in 1986; Appendix B manually reviewed from rendered pages}
}
```

## Proposed language and dialect routing

Reuse canonical Duruwa `[pci]` as the base language. Proposed dialect identities for the four explicit source headers:

```csv
sil-dhurwa-2021-tiriya,dialect:pci:sil-dhurwa-2021-tiriya:Tiriya,pci,sil-dhurwa-2021-tiriya,Tiriya,pci,,,,Dravidian,"Tiriya dialect region as labeled by the 1986 survey; exact wordlist collection village not printed in Appendix B",C
sil-dhurwa-2021-nethanar,dialect:pci:sil-dhurwa-2021-nethanar:Nethanar,pci,sil-dhurwa-2021-nethanar,Nethanar,pci,,,,Dravidian,"Nethanar dialect region as labeled by the 1986 survey; exact wordlist collection village not printed in Appendix B",C
sil-dhurwa-2021-dharba,dialect:pci:sil-dhurwa-2021-dharba:Dharba,pci,sil-dhurwa-2021-dharba,Dharba,pci,,,,Dravidian,"Dharba dialect region as labeled by the 1986 survey; exact wordlist collection village not printed in Appendix B",C
sil-dhurwa-2021-kukanar,dialect:pci:sil-dhurwa-2021-kukanar:Kukanar,pci,sil-dhurwa-2021-kukanar,Kukanar,pci,,,,Dravidian,"Kukanar dialect region as labeled by the 1986 survey; exact wordlist collection village not printed in Appendix B",C
```

Do not register or route `U5`. Its printed header is blank on every Appendix B page, and the report does not identify it authoritatively.

## Proposed profile policy

- Install the final complete-source profile as `conversion/sil-dhurwa-2021.txt`.
- Route only bibliography key `josephmichael2021dhurwa` through it.
- Preserve source transcription in `Original`/`Phonemic`; convert `ʈ ɖ ɳ` to house `ṭ ḍ ṇ`, `ʌ ɛ ɪ` to `a e i`, source `j` to `y`, `dʒ` to `j`, and colon length to `ː` only after full-source review.
- Re-run corpus-wide profile coverage after installing the already complete source-local profile.

## Remaining shared integration work

Copy the 809 final source-local target forms to the dated shared raw-form path, add the bibliography/four dialect rows/profile routing, run focused registry/profile tests, then the consolidated `make all`, inspect `errors.txt` and generated diffs, run full pytest and graph validation, and perform browser QA. Keep the 200 `U5` cells (199 responses) audit-only unless authoritative identity evidence is found. None of these shared gates belongs to this source-local lane.
