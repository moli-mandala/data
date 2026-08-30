# Deferred shared integration

## Bibliography

Add to `cldf/sources.bib`:

```bibtex
@techreport{mathew-chamberlain2022bonda-didayi,
  author = {Mathew, Chacko and Chamberlain, Bradford},
  title = {The Bonda and the Didayi from Malkangiri District, Orissa: A Preliminary Study},
  year = {2022},
  institution = {SIL International},
  series = {Journal of Language Survey Reports},
  number = {2022-004},
  url = {https://www.sil.org/resources/archives/92608},
  note = {Survey conducted in October 1997; researchers Chacko Mathew and Faith Adimathara}
}
```

## Target dialect rows

Append these exact rows to `cldf/dialects.csv` (coordinates are intentionally
blank because the report does not give surveyed-site coordinates):

```csv
sil-bonda-didayi-1997-biapada-u-didayi,dialect:gt:sil-bonda-didayi-1997-biapada-u-didayi:Biapada%20U.%20Didayi,gt,BIA,Biapada U. Didayi,gata1239,,,Munda,"Biapada, Upper Didayi; October 1997 SIL survey",C
sil-bonda-didayi-1997-chitrakonda-l-didayi,dialect:gt:sil-bonda-didayi-1997-chitrakonda-l-didayi:Chitrakonda%20L.%20Didayi,gt,CHI,Chitrakonda L. Didayi,gata1239,,,Munda,"Chitrakonda, Lower Didayi; October 1997 SIL survey",C
sil-bonda-didayi-1997-kaluguda-u-didayi,dialect:gt:sil-bonda-didayi-1997-kaluguda-u-didayi:Kaluguda%20U.%20Didayi,gt,KAL,Kaluguda U. Didayi,gata1239,,,Munda,"Kaluguda, Upper Didayi; October 1997 SIL survey",C
sil-bonda-didayi-1997-orapadar-u-didayi,dialect:gt:sil-bonda-didayi-1997-orapadar-u-didayi:Orapadar%20U.%20Didayi,gt,ORA,Orapadar U. Didayi,gata1239,,,Munda,"Orapadar, Upper Didayi; October 1997 SIL survey",C
sil-bonda-didayi-1997-oringi-l-didayi,dialect:gt:sil-bonda-didayi-1997-oringi-l-didayi:Oringi%20L.%20Didayi,gt,ORI,Oringi L. Didayi,gata1239,,,Munda,"Oringi, Lower Didayi; October 1997 SIL survey",C
sil-bonda-didayi-1997-rasabeda-l-bonda,dialect:re:sil-bonda-didayi-1997-rasabeda-l-bonda:Rasabeda%20L.%20Bonda,re,RAS,Rasabeda L. Bonda,bond1245,,,Munda,"Rasabeda, Lower Bonda; October 1997 SIL survey",C
sil-bonda-didayi-1997-kendhuguda-l-bonda,dialect:re:sil-bonda-didayi-1997-kendhuguda-l-bonda:Kendhuguda%20L.%20Bonda,re,KEN,Kendhuguda L. Bonda,bond1245,,,Munda,"Kendhuguda, Lower Bonda; October 1997 SIL survey",C
sil-bonda-didayi-1997-kadamguda-l-bonda,dialect:re:sil-bonda-didayi-1997-kadamguda-l-bonda:Kadamguda%20L.%20Bonda,re,KAD,Kadamguda L. Bonda,bond1245,,,Munda,"Kadamguda, Lower Bonda; October 1997 SIL survey",C
sil-bonda-didayi-1997-dumripada-u-bonda,dialect:re:sil-bonda-didayi-1997-dumripada-u-bonda:Dumripada%20U.%20Bonda,re,DUM,Dumripada U. Bonda,bond1245,,,Munda,"Dumripada, Upper Bonda; October 1997 SIL survey",C
```

Base language rows `gt` (Gtaʔ/Didayi) and `re` (Remo/Bonda), and control rows
`gu`, `go`, `AdivasiOriya`, and `Or`, already exist; add no language rows.

## Routing/build

- Register file stem `20260828-sil-bonda-didayi` with source key
  `mathew-chamberlain2022bonda-didayi`.
- Route this citation/source to `conversion/sil-bonda-didayi.txt` while
  preserving the source transcription fields.
- Run the focused test first, then the shared full build/test/browser gates.
- The survey contributes no etymology/graph edges; similarity group numbers are
  deliberately descriptive notes only.
- Update the discovery census: four, not three, prompts are DISQUALIFIED; item
  174 physically omits the ORA row.
