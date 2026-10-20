# Local Git and publication preparation

## Repository state

Repository name: `ICSI436_ProjectP_NewsRepLab`. Local branch: `main`.
The project originally had no `.git` directory. The initial commit
`6d582d3` (`chore: preserve verified midterm baseline`) captured 77 reviewed files
before source/documentation edits. Existing history was not overwritten.

The following commits separate consolidation from publication hygiene. Inspect
their exact IDs and changes with `git log --oneline` and `git show`.

- `09b7a85` — `refactor: consolidate educational code and project documentation`.
- The following hygiene commit records this review, the publication list,
  artifact hashes, ignore rules, Git attributes, and formatter configuration.

No remote is configured; no repository has been created on GitHub and nothing
has been pushed. Commit identity uses the existing local Git configuration.

## Candidate files and exclusions

The tracked tree contains source, offline tests, Markdown guides, requirements,
formatting configuration, reviewed midterm PPTX/PDF/LaTeX/script, figures, and
selected small historical results. The original 62 selected result/report files
are listed with SHA-256 hashes in [artifact_manifest.json](artifact_manifest.json).
They include sample row IDs/labels, metadata, aggregate and individual metrics,
per-class/confusion diagnostics, analyses, and provenance required by later rounds.

The complete candidate list is saved in [publication_files.txt](publication_files.txt).
Review it against the live Git index locally:

```powershell
git ls-files
git status --short --branch
git log --oneline
git remote -v
git ls-files -ci --exclude-standard
```

The final command should be empty: no tracked file should conflict with the
ignore policy. `.gitignore` allows only the reviewed historical result paths;
new runs remain local until explicitly reviewed. It excludes datasets, Hub/model
caches, numeric embedding arrays, weights, Conda/virtual environments, credentials,
`.env` files, Python/Spyder state, temporary/build files, and raw prediction/error
CSVs containing article text. Those existing local files were not deleted.

No file is excluded merely because it contains an unfavorable score. The same
artifact policy applies to both representations and every seed. Per-class counts
and confusion matrices remain available without publishing article text.

## Paths, identity, and secret review

The short Windows cache default `D:\hf_cache\datasets` is intentional and
configurable through `HF_DATASETS_CACHE`. Project paths derive from `__file__`.
Two interpreter error messages were generalized; historical metadata still
contain the original D: project/interpreter/cache paths and hardware/software
details. These are provenance, and are included in the proposed publication.
They are not required installation locations for a new checkout.

The review inspected selected text and embedded PPTX XML for common token/private
key patterns and account-home paths; no matches were found. This targeted review
is not a guarantee that every possible secret format was detected. Group names
and the existing Git author identity are intentional publication information.
Original artifacts and commit identity should be included in the final review.

## GitHub authentication and pending decisions

Git is available. The connected GitHub account reports **ZeyuPan2005**, and Windows
Git Credential Manager lists **ZeyuPan2005** as its stored GitHub account. The
`gh` CLI is not installed. No tools or packages were installed for authentication.
These checks do not verify permission to create a repository in an organization
or prove a future push will succeed.

The user must confirm these three items before publication:

1. Repository owner: `ZeyuPan2005` or a specified organization/account.
2. Visibility: Public or Private.
3. Approval of the final tracked file list, including midterm deliverables,
   historical workstation metadata, group names, and commit attribution.

After confirmation, check whether the named remote already exists, create it only
if appropriate, add `origin`, and push `main` normally. Never force push or
overwrite an existing repository. No license has been selected on the group's
behalf; repository visibility alone does not choose reuse permissions.
