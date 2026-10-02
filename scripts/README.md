# Commands

- `benchmark.py`: select the core task's unchanged collection/training/evaluation backend.
- `paper_release.py`: prepare pinned runtimes or run an original recorded evaluator.
- `fetch_paper_artifacts.py`: list or download selected models and evidence with SHA-256 verification.
- `verify_publication.py`: validate source packaging and frozen scientific files without a simulator.
- `bootstrap.sh`, `setup_policy_dependencies.py`: install the pinned core environment.

The remaining compatibility backends retain the source bytes used by recorded
experiments. Prefer the documented entry points in the root README and `docs/`.
HotStab's original tools live in its pinned runtime branch. Archived local
experimentation commands and development-diary directories are not the public API.
