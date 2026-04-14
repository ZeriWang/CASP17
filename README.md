# CASP17

Strict batch submission helper for CASP17 server-group uploads.

## Scope

This tool validates and uploads CASP17 submission files for:
- TS
- QA
- LG

It follows the server-group submission endpoint rule:
- HTTPS POST only
- endpoint: https://predictioncenter.org/casp17/submit
- form fields: email, prediction_file

## Files

- casp17_bulk_submit.py: main script
- config.example.json: example configuration

## Quick start

1. Prepare files in one directory (default pattern: *.pdb).
2. Copy and edit config.example.json.
3. Run validation-only dry run first.
4. Run real upload.

## Commands

Validation only:

```bash
python3 casp17_bulk_submit.py \
  --config config.example.json \
  --dry-run
```

Real upload:

```bash
python3 casp17_bulk_submit.py \
  --config config.example.json
```

Force overwrite previously successful target/category/model keys:

```bash
python3 casp17_bulk_submit.py \
  --config config.example.json \
  --force-resubmit
```

Only retry files that failed previously:

```bash
python3 casp17_bulk_submit.py \
  --config config.example.json \
  --retry-failed-only
```

If your official checks confirm LG allows top poses (for example top 5), run:

```bash
python3 casp17_bulk_submit.py \
  --config config.example.json \
  --max-lg-models 5
```

If QA for your target set allows multiple MODEL blocks, run:

```bash
python3 casp17_bulk_submit.py \
  --config config.example.json \
  --max-qa-models 5
```

## Key CLI options

- --input-dir: directory to scan
- --glob: file pattern (default *.pdb)
- --email: submitter email
- --allowed-domain: optional domain guard for email
- --endpoint: submission endpoint
- --timeout: request timeout in seconds
- --max-retries: retries for network/HTTP 5xx failures
- --checkpoint: checkpoint JSON path
- --report: report JSON path
- --target-filter: comma-separated targets (for partial runs)
- --dry-run: validate only
- --force-resubmit: allow overwrite behavior
- --retry-failed-only: only process known failed hashes
- --verbose: print per-file validation summary
- --max-qa-models: max QA MODEL blocks allowed (default 1)
- --max-lg-models: max LG MODEL blocks allowed (default 1)

## What strict validation checks

Common checks:
- ASCII file content only
- max 80 columns per line
- mandatory header order at top: PFRMAT, TARGET, AUTHOR
- single PFRMAT and single TARGET per file
- METHOD must appear before first MODEL
- at least one MODEL ... END block
- no duplicate non-contiguous residues in one MODEL
- occupancy in {0.00} or [0.01, 1.00]
- B-factor in [0, 100]
- residue-level B-factors cannot be all identical in one MODEL

TS checks:
- up to 6 models
- each model index in [1, 6]
- model index uniqueness
- PARENT required in each model
- TER required in each model
- ATOM/HETATM required

QA checks:
- max MODEL block count controlled by --max-qa-models (default 1)
- model index in [1, max_qa_models]
- no duplicate model index
- each score line must contain overall score in [0,1]
- interface scores format CH:score and each in [0,1]

LG checks:
- max MODEL block count controlled by --max-lg-models (default 1)
- model index in [1, max_lg_models]
- no duplicate model index
- each LIGAND block must close with M  END
- LIGAND id must be numeric
- LSCORE in [0,1]
- AFFNTY format: AFFNTY <value> <aa|ra|lr>
- lr value must be integer

## Output files

- checkpoint.json: persistent run state for idempotency and retries
- submission_report.json: per-file run report and summary

## Notes

- Defaults remain strict (QA=1, LG=1) for safety.
- Increase QA/LG model limits only after validating your current CASP17 target-specific rules.
- CASP may still reject files based on server-side logic not fully inferable from format text.
- For model 6 in TS, CASP mentions organizer-provided MSA special case. This script warns but does not auto-block model 6.
- Always test with --dry-run before real upload.
