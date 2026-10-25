# Third-party sources in this snapshot

- `openpi/` is the vendored source used by this environment. Its Apache-2.0 license is retained in `openpi/LICENSE`; preserve this file and existing copyright/license headers when redistributing. No separate upstream NOTICE was found in the supplied tree.
- Preserve `openpi/LICENSE_GEMMA.txt` (Gemma Terms of Use) and the Big Vision/Google notices in model source files. Model weights, tokenizers, and datasets are not bundled. Obtaining or using model assets is subject to their applicable terms.
- LeRobot is fetched during Docker build from the commit pinned in `requirements.lock`. Its source and binary data are not vendored here; the pinned dependency retains its upstream license.
- PiPER-derived conversion/training code has already been relocated to `conversion/` and `training/`. The supplied `piper_pi0/` reference tree has no standalone LICENSE; no additional license grant is asserted. Existing attribution/docstrings and `SOURCE_PROVENANCE.md` are retained. That reference tree is omitted because the current Dockerfile and runtime do not depend on it.
- A private source backup does not establish permission for unrestricted public redistribution. Verify the underlying PiPER-derived source permissions before broader redistribution.

No new blanket license is applied to the project-specific code by creating this snapshot.
