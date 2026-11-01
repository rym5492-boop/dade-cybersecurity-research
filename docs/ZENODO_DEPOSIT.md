# Archiving a release on Zenodo (DOI)

Zenodo archives a copy of a GitHub release and gives it a DOI that cannot be changed afterwards. The
metadata are read from `.zenodo.json` in the repository root.

1. Sign in at https://zenodo.org with the GitHub account that owns this repository.
2. Open the account menu, choose GitHub, press Sync now, and switch this repository on.
3. On GitHub, create a release from `main` (Releases, Draft a new release; tag `v2.0-revision2`; title
   "Second revision, 1 November 2026"), and publish it.
4. Within a few minutes the release appears under Zenodo, Uploads, with its DOI. The DOI is then cited in the
   Data availability statement of the manuscript and in thesis Appendix E.

What is not deposited: the original private repository, its Git bundle and the named form exports. They are
kept unchanged and offline by the first author, under the control of the research ethics determination.

Done on 1 November 2026: release v2.0-revision2 of commit 310b029, DOI 10.5281/zenodo.23856795.
