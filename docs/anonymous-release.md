# Releasing this anonymously

For a double-blind venue the repository has to survive a motivated reviewer
looking for the authors, and most deanonymisation is accidental: a commit
email, a notebook path with a username in it, a licence with a real name.

This document is the checklist and the three hosting options, best first.

## Option 1 — Anonymous GitHub (anonymous.4open.science)

The standard for OpenReview submissions, and the one to use unless something
rules it out.

You give it a GitHub repository URL; it serves an anonymised mirror at a
URL like `https://anonymous.4open.science/r/abc123de/`, rewriting the repository
name and stripping the owner. Reviewers browse and download without ever
reaching your account.

1. Push to a **private** GitHub repository. Private is fine and is what you
   want: the anonymiser reads it through an access grant, and nobody can find
   the public original by searching.
2. Go to anonymous.4open.science, authorise it against that repository, set an
   expiry past the decision date, and generate the link.
3. **Open the generated link in a private browser window and read it as a
   reviewer would.** This is the step people skip. The anonymiser rewrites the
   repository name and owner; it does not read your files.
4. Put the link in the response.

What it does not protect you from: anything inside the files. Run the checklist
below first.

## Option 2 — A throwaway GitHub account

Use when you want a normal-looking repository, or when the anonymiser is down
during a response window, which happens.

The risk is linkage through metadata rather than through content. A fresh
account created from your usual browser session, with your usual email, pushed
with your usual `git config`, is not anonymous.

```bash
# a fresh clone with no history from your working repo
mkdir nichecast-audit-anon && cd nichecast-audit-anon
git init

# repo-local identity, set BEFORE the first commit
git config user.name  "Anonymous Author"
git config user.email "anonymous@example.com"

# copy only tracked content, not .git
rsync -a --exclude='.git' ../nichecast-audit/ .

git add -A
git commit -m "Initial commit"
```

A single squashed commit is the point. Importing real history carries every
author name, email and timestamp in it, and `git log` is the first place
anyone looks.

Use a different email provider, a private browsing session, and ideally a
different network than your institution's. Do not star it, fork it, or follow
it from your real account — the activity graph links them.

## Option 3 — Zenodo or OSF

Worth knowing for the camera-ready rather than for review.

Zenodo mints a DOI and will archive a GitHub release automatically, which is
what you want for the final version, but it is not anonymous. OSF supports
genuine anonymised view-only links and handles large files well, so it is the
better choice if the repository ever needs to carry the prediction cache.

## The pre-release checklist

Content leaks matter more than hosting. Work through these against this
repository.

### Git metadata

```bash
git log --format='%an <%ae>%n%cn <%ce>' | sort -u   # every name and email in history
git config user.email                               # what the next commit will carry
```

Any real identity here means starting from a fresh `git init`, not amending.
Rewriting published history does not retract what was already pushed.

### File content

```bash
grep -rniE 'your-?surname|your-?university|@[a-z]+\.(edu|ac\.[a-z]{2})|orcid' .
grep -rn '/Users/\|/home/[a-z]\|C:\\\\Users\|MyDrive\|/content/drive' .
grep -rniE 'api[_-]?key|hf_[A-Za-z0-9]{20,}|secret|token\s*=' .
```

The three that actually catch people:

- **Absolute paths.** `/Users/jsmith/...` and
  `/content/drive/MyDrive/<project>/...` both name you. Colab notebooks are the
  worst offender because the Drive mount path is in every cell that writes a
  file. In this repository, `REPRODUCING.md` deliberately shows the export cell
  with a relative `AUDIT_DIR` for exactly this reason.
- **Notebook outputs.** A `.ipynb` carries executed output, and a traceback or
  a `print(path)` will contain your home directory. Clearing outputs defeats
  the purpose of shipping a notebook that carries its results, so read them
  instead:
  ```bash
  jupyter nbconvert --to script --stdout notebooks/*.ipynb | grep -n '/Users/\|MyDrive\|/content/'
  ```
- **Tokens.** `HF_TOKEN` must be read from the environment and never appear in
  a file. A leaked HuggingFace token identifies the account that owns it.

### Names that are not names

- `LICENSE` — this one says "Copyright (c) 2026 Anonymous Authors". Check it
  after any copy, because editors and templates helpfully substitute a real
  name.
- The repository name itself. A project codename that appears in your lab's
  public materials, a talk title, or a prior preprint is a search away from
  you. If the codename is public anywhere, rename for the anonymous mirror.
- `CITATION.cff`, `setup.py`, `pyproject.toml`, `.zenodo.json` and badge URLs
  all carry author or account fields. None are present here; check again if you
  add packaging.
- Image and PDF metadata. `exiftool -Author -Creator <file>` on any figure you
  add. LaTeX-produced PDFs embed the username in `/Creator` surprisingly often.

### Final read

Clone the anonymous link fresh into a clean directory, run `pytest tests/`, and
read `README.md` and `REPRODUCING.md` as a stranger. If a reviewer cannot run
the test suite from a cold clone, the anonymisation has cost you the thing the
link was for.

## What this repository currently exposes

As shipped: nothing. No author names, no institutional identifiers, no account
handles, no absolute paths, no tokens, and a licence naming "Anonymous
Authors". The test suite runs from a cold clone with no data and no
credentials.

Two files you are likely to add do carry risk, and both are worth a second
look before the link goes out:

- `notebooks/multiphi_audit.ipynb` — Colab Drive paths and executed outputs.
  This is the single highest-risk file in the planned release.
- `results/audit_long.csv` — check the `slide` column. HEST identifiers are
  public dataset identifiers and are safe; internal sample names from your own
  pipeline would not be.
