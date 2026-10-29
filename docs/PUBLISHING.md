# Publishing original files

[Documentation home](../README.md)

Published source, reconstructed databases, and saved output members retain their
original bytes. ZIP containers may be repacked without changing those bytes.
Git LFS, Git-annex placeholders, symbolic links, submodules, and download stubs
are prohibited. `.gitattributes` disables content filters and newline conversion.

The complete distribution is stored in Git. The five large databases use
ordinary ZIP byte chunks below 100,000,000 bytes under `_database_archives/`,
with ordered chunks, original target paths, sizes, and SHA-256 hashes recorded
in [`_database_archives/manifest.json`](../_database_archives/manifest.json).
The launcher reconstructs and verifies the original files before opening the
browser; `python database_archives.py` performs the same preparation explicitly.
Individual ZIPs are valid containers, but each holds only a byte chunk: the
helper concatenates decoded payloads in manifest order. ZIP files sit directly
in `_database_archives/` and use `<database_name>_part<number>.zip`, for example
`thermoml_raw_part1.zip`. Part numbers are unpadded; byte offsets and lengths
remain in the manifest rather than the filenames.

The five reconstructed `.db` paths are ignored generated files, as is
`.database_cache/`, which holds verification receipts and an operating-system
lock. Keep the ZIP payloads and manifest in Git. No placeholder file is committed
at an original database path, and no separate release download is required.
[RELEASE_ASSETS.json](RELEASE_ASSETS.json) remains unchanged as the legacy
original-hash/provenance inventory used by the packaging builder.

[`scripts/package_databases.py`](../scripts/package_databases.py) packages the
original database files using that inventory by default and checks the expected
SHA-256 hashes. Repackage from the originals, then verify the generated manifest
and reconstructed files before publication. Keep the exact five database ignore
entries; do not broaden them to ignore database ZIP payloads.

Benchmark collections are stored in Git under each agent's
`test_run_full_capability/` and `test_run_ablation/` folders. Every ZIP part is
smaller than 100,000,000 bytes and keeps each example whole. These parts are read
directly, without reassembly or extraction. Split filenames identify the first
and last examples; exact membership, sizes, and SHA-256 hashes are recorded in
[`_benchmark/ARCHIVES.json`](../_benchmark/ARCHIVES.json). Collection-level shared
metadata is repeated in each part. See [BENCHMARKS.md](BENCHMARKS.md) for the
naming scheme and direct browsing of saved evaluation records.

Publish evaluation statistics and the reproducible workflow as
`_benchmark/_eval_stats/eval_stats.zip` with its `manifest.json`. The current
package fits in one ZIP below 100,000,000 bytes. Extracted payloads and the
local preparation cache are ignored; first launch restores them automatically.
`python eval_stats_archives.py` performs the same preparation explicitly.
Use [`scripts/package_eval_stats.py`](../scripts/package_eval_stats.py) to
repackage updated payloads, then verify restoration before publication.

Keep all other source, smaller databases, benchmark ZIPs, and shared `_output`
artifacts in Git. Changes to the distribution plan must preserve original bytes
and the no-pointer policy.

Hook activation requires an existing Git checkout. For a workspace snapshot
without a `.git` directory, first initialize or restore the intended checkout
as a separate authorized operation. The activation command does not initialize
a repository, stage files, commit, or push:

```shell
python scripts/check_no_pointers.py --install-hook
```

When first adding the hook, preserve its executable bit in Git:

```shell
git add --chmod=+x .githooks/pre-push
```

Check staged files before committing, then check the committed refs before
pushing (replace `HEAD` with the actual refs you will publish):

```shell
python scripts/check_no_pointers.py --staged
python scripts/check_no_pointers.py --refs HEAD
```

The pre-push hook checks every object reachable from the refs being pushed,
including historical commits and annotated tags. It rejects recognized LFS,
Git-fat, and Git-annex pointer files, symbolic links, and submodules even when
the current working copy contains the original file. It reads object metadata
first and inspects only trees and small blobs, so large original database/ZIP
blobs are not loaded for pointer detection. The size bound is 8 KiB, above the
standard Git LFS pointer size; the written policy also forbids other kinds of
placeholder that cannot be identified from a standard signature.

Hooks are local Git configuration and are not automatically enabled by cloning.
Run the activation command in each checkout. Do not use `--no-verify` or otherwise
bypass this policy. Hosting services can impose independent file-size limits;
if a push is rejected, retain the originals and resolve that limit explicitly
instead of converting files to pointers.

The five original database files exceed the ordinary GitHub file-size limit.
Their complete bytes are distributed in the committed ZIP chunks, with every
archive below 100,000,000 bytes. This does not authorize pointer substitutes,
loss of files, or replacement of payloads with download instructions.

When updating a repository that already used LFS, a new commit containing original
files still retains pointer blobs in its ancestry. The committed-object check
rejects that history too. The user approved replacing `main` with pointer-free history for the
`workspace-20260918` publication. That authorization covers this replacement;
future remote-history replacements require separate explicit approval. Never
bypass the committed-object check.
