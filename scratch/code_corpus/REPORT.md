# Deterministic Stack v2 Python corpus — build report

Corpus built and verified. **No training has been started.**

| | |
| --- | --- |
| commit | see `git log -1` on `sidecar-distill` |
| tests | 842 pass (42 new in `tests/test_code_corpus.py`) |
| HF access | authenticated as `zcomputerwiz`, OAuth token, expires 2026-10-14 |
| dataset | `bigcode/the-stack-v2-dedup`, config `Python`, gated `auto`, access granted |
| revision | `611ef38fbfab78b65b38be47e870e4b3d70de038` |
| tokenizer | `Qwen2Tokenizer` from `student-2b-hf`, vocab 248,044, `tokenizer.json` sha256 `5f9e4d49…` |
| split seed | `20260914` |
| build time | 99 s after resume (≈200 s total), 80.3 MB retrieved |
| disk | 42.9 MiB parquet (zstd), 43 parts |

## Acquisition probe

500 records from shard 0, retrieved from Software Heritage's public object store over
unsigned HTTPS. **No AWS credentials were used or needed.**

```
attempted                500
retrieved                500
missing / 404              0
decompression failures     0
checksum mismatches        0     <- blob_id IS the sha1 of the content, so every
other failures             0        retrieval validates itself
```

```
385.5 files/s      2.08 MB/s      598,233 model tokens/s end to end (64 workers)
```

Blobs are gzip members; `blob_id` is the SHA-1 of the decompressed bytes, so
verification is free and was applied to every file in the corpus, not just the probe.

### Record schema

`blob_id`, `directory_id`, `path`, `content_id`, `detected_licenses`, `license_type`,
`repo_name`, `snapshot_id`, `revision_id`, `branch_name`, `visit_date`, `revision_date`,
`committer_date`, `github_id`, `star_events_count`, `fork_events_count`,
`gha_license_id`, `gha_event_created_at`, `gha_created_at`, `gha_language`,
`src_encoding`, `language`, **`is_vendor`**, **`is_generated`**, `length_bytes`,
`extension`, `filename`.

Everything the task asked for is present, including both metadata quality flags.

## Corpus

| split | repositories | files | model tokens | target |
| --- | ---: | ---: | ---: | ---: |
| train | 32,519 | 34,606 | 30,725,434 | 30,000,000 |
| calibration | 1,978 | 2,218 | 2,034,946 | 2,000,000 |
| heldout | 5,137 | 5,405 | 5,011,233 | 5,000,000 |

Median 363 tokens/file in train, 360 in calibration, 369 in heldout — the splits are
drawn from the same distribution, as they must be.

### Repository overlap

```
train/calibration      0
train/heldout          0
calibration/heldout    0
```

Zero by construction, not by luck: a repository's split is
`blake2b-64("<seed>:<repo_name>") / 2**64` tiled over the splits in sorted name order.
Every file of a repository hashes identically, so straddling is impossible, and the
assignment does not depend on arrival order, corpus contents, or how full a counter was.

Independently re-derived from the stored parquet by `verify.py`, which re-reads every
row and recomputes the hash: **0 files whose split disagrees with their repository.**

### Filters

```
generated removed            17
vendor removed               10
empty / too small           829
oversized (bytes)             8
token-oversized              81
token-dense                 284
undecodable                   0
duplicate blobs               0
benchmark-contaminated      124   (117 train + 7 calibration)
```

Two of these are not in the original spec and were added because the data demanded it.
One retrieved file was **1,035,623 bytes tokenizing to 1,035,618 tokens** — one token per
byte, obfuscated data rather than Python — and it sat comfortably under a 1 MiB byte cap
while being a quarter of an entire training split on its own. Byte-size filtering cannot
see this. `max_tokens` (32,768) and `min_bytes_per_token` (1.8) catch it and 364 others.
Ordinary Python runs 3–4 bytes/token; machine-generated but genuine Python (SNMP MIB
tables) runs around 2; obfuscated content runs 1.

### Verification against the stored corpus

`verify.py` reopens the parquet the way a training loader would and recomputes every
claim independently of the builder's counters:

```
manifest token/file/repo counts agree for all three splits    yes
repository overlap                                            0 / 0 / 0
duplicate blobs across all splits                             0
files whose split disagrees with their repository hash        0
re-tokenized token_count mismatches (6,000 files sampled)     0
empty stored sources                                          0

VERIFIED
```

## Benchmark decontamination

Indexed MBPP+ (378) and HumanEval+ (164) problem statements and reference solutions as
13-token shingles: 25,934 shingles, **513 of 542 problems detectable**. The remaining 29
are solutions too short to yield four distinctive shingles — a real limitation of the
method, stated rather than papered over, and one no threshold fixes because what such a
file shares with an ordinary one genuinely is the same content.

**141 files flagged; 124 excluded from train and calibration; 17 kept in heldout and
recorded** in `v1/contamination.json` with the matched problem and hit count. Most-matched
problems: `mbppplus:475` (31 files), `mbppplus:265` (16), `humanevalplus:HumanEval/129` (11).

### The matcher was wrong twice before it was right

Worth recording, because both failures produced plausible-looking output.

**First version indexed the EvalPlus `test` column.** Those are extended generated test
suites — one HumanEval+ row is 77 KB, 38,772 tokens of numeric literals. Shingling them
put millions of generic runs in the index. It flagged 3.4% of ordinary Python.

**Second version counted raw shingles pooled across all problems.** It still flagged 1.7%,
and tracing the matches showed why: files were hit on `1 , 2 , 3 , 4 , 5 , 6`,
`dp [ i ] [ j ] = max ( dp`, and `for _ in range ( N + 1 ) ] for _ in`. Punctuation
tokenizes one character at a time, so a 13-token window can be eleven commas and digits.
Every flagged file was an ordinary dynamic-programming or competitive-programming file —
that is, the matcher was systematically deleting the algorithmic Python this experiment is
about.

The fix is two changes. A window must contain **4 distinct alphabetic tokens** to be
shingled at all, at both index and query time. And scoring is **per problem**: a real copy
shares a long consecutive run with *one* problem (median 31 shingles), while a coincidence
shares one or two each with a scatter of unrelated ones.

The threshold was then measured rather than chosen (`calibrate.py`, sweep in
`manifests/calibration.json`):

| threshold | verbatim caught | reformatted | renamed | ordinary flagged |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 504 / 542 | 503 | 174 | 1 / 582 |
| 4 | 462 | 461 | 141 | 1 / 582 |
| 8 | 410 | 407 | 99 | 1 / 582 |
| 12 | 358 | 355 | 73 | 0 / 582 |

The false-positive rate is flat from 2 through 8 while recall climbs as the threshold
falls, so **4** sits at the top of the flat region. The single ordinary file flagged at
every threshold is a competitive-programming solution — plausibly a true positive.

**Known limitation:** systematic identifier renaming defeats this (141/542 caught). A
renaming-robust matcher needs AST normalization, which was not built. Verbatim and
reformatted copies — what actually appears on GitHub — are caught.

## Resume

Demonstrated on the real build, not just unit-tested. The run was hard-killed (`SIGKILL`)
at 100 s with 20 parts written, resumed from its checkpoint, and completed. `verify.py`
then found **0 duplicate blobs**, which is the property a broken resume would violate.

Checkpoints carry `(shard, row_group, row_offset)` and the list of every parquet part they
know about. Two bugs were found and fixed getting here:

- The final checkpoint read `shard_index` after the outer loop had already advanced past
  it, so a resume would have **skipped an entire shard's first row group**.
- Checkpoints originally fired only at row-group boundaries. A row group is ~100k metadata
  rows and takes minutes to drain; the first real run was killed at 120 s having written
  **22,000 files with no checkpoint to resume from**. Checkpoints are now periodic
  (every 8 retrieval rounds, ~4,200 rows).

A part on disk that no checkpoint claims is deleted on resume rather than kept, since
keeping it would duplicate every row it holds when the stream replays.

## Representative files

Ranked by a hash of file identity, so the sample is reproducible and is neither the head
of the file nor a popularity ranking. Full lists in `manifests/v1-sample.json`.

**train** — `danilkaz/Geocoder/organizations.py`,
`MarMel197/Python_and_Flask_Solo_Project/.../product_controller.py`,
`cainingning/leetcode/tree_117.py`, `Sliver94/Laboratory_OON/OLD/Lab1/lab1_ex8.py`,
`choderalab/openmoltools/openmoltools/tests/test_openeye.py`,
`ryu19-1/atcoder_python/joi2016yo/e/main.py`, `alexandermuehle/rplx_crawler/rplx_crawler.py`,
`dronedeploy/skydio-skills/skillset/__init__.py`, `lngka/mllangid-dec/train_dec.py`,
`Jarvl/red-canary-coding-project/main.py`

**calibration** — `Ashikunnabi/DjReactPortfolio/.../0002_auto_20200812_1911.py`,
`Opentrons/opentrons/api/src/opentrons/hardware_control/emulation/magdeck.py`,
`beck/django-http-proxy/setup.py`, `taoketao/explore-combinatorics/expenv.py`,
`LefterisJP/rotkehlchen/rotkehlchen/tests/unit/test_evm_contracts.py`,
`mauler/django-elastic-transcoder/dj_elastictranscoder/utils.py`

**heldout** — `antw0/Feeding-Canadian-Kids/project/profiles/models.py`,
`DenMaslov/fastapi_blog/blog_api/apps/posts/dependencies.py`,
`zzzevaka/findchat/backend/main_app/api_v1/comment.py`,
`corba777/openai_gym/test/test_rl.py`,
`GillesVandewiele/InterpretableEnsembles/constructors/xgboostconstructor.py`,
`jonjomckay/jenna/jenna.py`, `pranavtbhat/CS249/process_vertex_data.py`

Django models and migrations, Flask controllers, test suites, scientific scripts, course
exercises, competitive-programming solutions, crawlers. Ordinary Python, which is what
was asked for — no benchmark-driven sampling was applied.

## Two things that need a decision before training

**License mix.** 76% of the corpus is `license_type: no_license` (26,397 of 34,606 train
files); the rest is `permissive`. The task did not specify a license filter and none was
applied, so this reflects The Stack v2's actual Python distribution. Restricting to
permissive is a one-flag change (`--license-types permissive`) and would cost roughly
three-quarters of the corpus, requiring a proportionally longer stream. Given this is
noncommercial research on a local machine the current setting is defensible, but it is a
judgement call rather than a measurement and it is yours.

**Partial repositories.** Almost every repository in the corpus contributes only some of
its files. This is inherent to stopping a 45M-file stream after ~42k files, not an artefact
of the targets: a split stops admitting *new* repositories at its target and closes at
1.10× it, so the targets never influence *which* split a repository lands in. Actual counts
are 1.024×, 1.017× and 1.002× target.

## Files

```
distillkit/code_corpus.py          split policy, filters, blob handling, matcher (tracked)
tests/test_code_corpus.py          42 tests, synthetic fixtures, no network (tracked)
scratch/code_corpus/stack.py       metadata streaming, threaded retrieval, index
scratch/code_corpus/probe.py       the acquisition probe above
scratch/code_corpus/calibrate.py   threshold sweep against real positives and negatives
scratch/code_corpus/build.py       the builder
scratch/code_corpus/verify.py      independent re-derivation from stored parquet
scratch/code_corpus/sample.py      deterministic sample and license distribution
scratch/code_corpus/manifests/     probe.json, calibration.json, v1-sample.json (tracked)
scratch/code_corpus/v1/            the corpus itself (gitignored: 43 MiB of third-party source)
```

## Next

Awaiting review before any GPU work. The planned next step is identical `B0` continued
pretraining on the 30.7M train tokens, then freezing that code-trained backbone and
fitting fresh `G_code` and `S_code` against it.
