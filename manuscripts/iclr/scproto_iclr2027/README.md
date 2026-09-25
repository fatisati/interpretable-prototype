# scProto --- ICLR 2027 submission

Self-contained LaTeX project. Content was **copied** from the NeurIPS 2026
submission at `../../neurips_manuscript/`, which is treated as **read-only** and
was not modified. Only the shell (`main.tex`) and `statements/` are new.

Deadlines this folder exists for: NeurIPS decision **24 Sep 2026** (AoE),
ICLR 2027 full-paper deadline **25 Sep 2026** (AoE).

---

## Build

```powershell
cd manuscripts\iclr\scproto_iclr2027
.uild.ps1
```

Requires MiKTeX (already installed: `pdflatex` 4.26 / MiKTeX 26.2). The script
runs pdflatex -> bibtex -> pdflatex -> pdflatex, keeps all aux/log clutter in
`build/`, and copies the result to `scproto_iclr2027.pdf` in this folder. It
then prints the page count, any unresolved `
ef`/`\cite`, and any placeholder
figures still present.

To build by hand or on Overleaf, compile `main.tex` from this folder --- the
style files are referenced as `style/iclr2027_conference` so relative paths work
without any TEXINPUTS setup.

## Layout

```
main.tex                   ICLR shell: documentclass, packages, title, \input order
references.bib             bibliography (copied verbatim from NeurIPS)
build.ps1                  4-pass build + sanity report
.gitignore                 ignores build/ and LaTeX aux files
style/                     OFFICIAL ICLR 2027 files, UNMODIFIED -- do not edit
  iclr2027_conference.sty  (tweaking the style files can be grounds for rejection)
  iclr2027_conference.bst
  fancyhdr.sty  natbib.sty  math_commands.tex
sections/                  main text, in \input order
  abstract.tex  introduction.tex  related_work.tex  method.tex
  results.tex  discussion.tex
statements/                ICLR-specific, NEW -- all still DRAFT, see below
  ai_use.tex               REQUIRED by ICLR 2027
  ethics.tex               recommended
  reproducibility.tex      recommended
appendix/
  training.tex             everything after the bibliography
tools/
  check_refs.py            verifies every .bib entry against Crossref/DataCite/arXiv
  verified_manually.txt    keys with no API record, hand-checked + evidence URL
  ref_report.json          last run's full report
  references.bib.bak       bib as inherited from NeurIPS, before the fixes
figures/                   only the figures the source actually references
  unused/                  figures carried over that nothing \includegraphics
build/                     generated; safe to delete
```

## What changed vs. the NeurIPS version

| | NeurIPS 2026 | ICLR 2027 |
|---|---|---|
| style | `neurips_2026.sty` | `style/iclr2027_conference.sty` + `times` |
| citations | numeric (`[numbers,compress]` to natbib) | **author-year** (ICLR's natbib default) |
| bib style | `unsrtnat` | `style/iclr2027_conference` |
| author block | named author | `Anonymous authors` (ICLR is double-blind) |
| checklist | `checklist.tex` (NeurIPS Paper Checklist) | **dropped** --- not an ICLR artifact |
| statements | none | `statements/` AI use (required) + ethics + reproducibility |
| appendix | before the checklist | after the bibliography (ICLR asks for this order) |
| section labels | none | added `\label{sec:method}`, `\label{sec:experiments}` so the reproducibility statement can cross-reference them |

Everything else in `sections/` and `appendix/` is byte-identical to the NeurIPS
source apart from those two added `\label` lines and the two `\missingfig`
swaps described next.

## Known issues to fix before submitting

1. **Two figure files do not exist anywhere in the repo.** `appendix/training.tex`
   references `figures/pancreas_aff.png` (Fig. 4, the affinity-graph UMAP showing
   batch structure) and `figures/nsclc_affinity_comparison.png` (Fig. 5,
   transcriptomic vs. spatial affinity). The NeurIPS source therefore cannot
   compile as-is. They are currently rendered by the `\missingfig` macro defined
   in `main.tex`, which draws a red-outlined "MISSING FIGURE FILE" box so that
   `\label`/`
ef` numbering stays correct and the gap is visible in the PDF.
   Once the files are regenerated, drop them into `figures/` and replace each
   `\missingfig{...}` with `\includegraphics[width=	extwidth]{...}`.

2. **Main text is over the 9-page limit.** ICLR 2027 allows 9 pages of main text
   at submission (10 during rebuttal / camera-ready) and papers over the limit
   are **desk-rejected**. In the current build the Discussion spills onto page 10.
   Roughly half a page has to come out of Sections 1--5. (References, the three
   statements, and the appendix do *not* count toward the limit.)

3. ~~`references.bib` has an empty `year`~~ **FIXED** --- and the whole
   bibliography has now been verified, see "Reference checking" below.

4. **`statements/*.tex` are drafts**, each marked with a red
   `[DRAFT --- review and finalise before submission]` in the PDF. The AI use
   statement is *required* by ICLR and must be accurate about actual AI use. The
   reproducibility statement still needs an anonymised code link.

5. **Content is still the NeurIPS version.** None of the rebuttal experiments,
   the reviewer-driven edits, or the new CPM/`nassoc` loss results are in here
   yet --- see `../plan.txt` and `../../../my-tasks/doing/iclr.txt`.

## Reference checking

The NeurIPS Citation Checker flagged 3/37 references on the submitted version.
Two hallucinated references is a presumptive reject, so the whole bibliography
was re-verified against authoritative metadata rather than by eye.

```powershell
python tools\check_refs.py references.bib --json tools
ef_report.json
```

For each entry it looks the paper up by DOI (Crossref, falling back to DataCite
for figshare/Zenodo datasets), or by title if there is no DOI, or on arXiv for
preprints, then compares title, year, venue, authors, volume and pages. It
never edits the .bib.

Verdicts: `WRONG-DOI` (the DOI resolves to a different paper) and `NOT-FOUND`
are the two the conference treats as hallucination. `CHECK` means a real paper
with wrong metadata. `MANUAL-OK` means no index has it --- NeurIPS proceedings
papers have no DOI, workshop papers live only on OpenReview --- and a human
confirmed it; those keys and their evidence URLs live in
`tools/verified_manually.txt`. Add a key there only after opening the URL.

Current state: **39 OK, 3 MANUAL-OK, 0 CHECK, 0 WRONG-DOI, 0 NOT-FOUND**, and
bibtex reports 0 warnings. Every entry either matches an authoritative
record field-for-field or is hand-verified with its evidence URL on file.

Re-run this after any edit to `references.bib`, and always once before
submitting.

## Camera-ready

Uncomment `\iclrfinalcopy` in `main.tex` and restore the real author block.
Leave it commented for submission --- a non-anonymous submission is desk-rejected.
