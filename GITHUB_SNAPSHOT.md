# Independent source snapshot (2026-10-26 JST)

This repository contains the current tuning source files copied from the working project, including uncommitted source changes. It has a new Git history; the original repository, branches, remotes, files, images, containers, datasets, and checkpoints were left unchanged.

The snapshot uses the existing source package allowlist. `piper_pi0/`, OpenPI examples/third_party/docs, local data, logs, credentials, caches, environments, and binary outputs are excluded. Current runtime code uses `conversion/` and `training/`; the Dockerfile explicitly excludes the obsolete `piper_pi0/` tree. Required OpenPI source, client, custom trainer, statistics script, metadata and license files are included.

On a new PC, install Docker Compose, an appropriate NVIDIA driver and NVIDIA Container Toolkit, then clone this repository and run:

```sh
sh build.sh
sh run.sh
```

The build requires network access for the pinned Python packages and base images. Model assets, datasets and pretrained weights are obtained separately; they are intentionally not stored in Git. Runtime output directories are created by the shell helpers, so empty data directories need not be committed.

The code and Docker recipes were checked statically for this snapshot. No Docker build, training or conversion was run during snapshot creation. Earlier validation reports are historical evidence, not a new build verification.

Historical reports and developer verification scripts refer to old Git commit IDs, local datasets and ignored logs. The new one-commit repository does not contain that old history or those data artifacts. Normal build/run/conversion/training does not require them. Cross-version regression scripts need a separately supplied historical reference.

For updates, work in this repository/clone, review `git status` and `git diff`, stage only intended source files, inspect `git diff --cached --stat` and `git diff --cached`, then commit and push `main`. Never add data, weights, logs or credentials.
