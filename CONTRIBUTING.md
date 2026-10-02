# Contributing

Keep benchmark definitions explicit and preserve the recorded protocols. Report
changes to physics, success criteria, observations, action rates, train/test splits
and checkpoint selection as new versions rather than silently replacing old scores.

Use fresh output directories, retain failed episodes, and record source/config/
checkpoint hashes. Do not choose changes on the published test states and then
call a repeat on those states an independent evaluation.

Format Python with Ruff using `pyproject.toml`. Run the source checks and applicable
CPU tests before proposing a change; run simulator checks when the changed behavior
requires them. Unit CI is not a GPU benchmark. New experiments and training are not
started automatically by CI. Include the concrete change, validation and limits in
a pull request; keep unrelated tasks and exploratory artifacts out of it.

Large datasets, intermediate weights, caches and licensed CAD do not belong in
Git. Preserve third-party notices. Source releases and sites remain private until
the repository owner explicitly authorizes public access.

Implementation guides: [add a task](https://dancher00.github.io/wasserman/docs/add-task/), [robot or controller](https://dancher00.github.io/wasserman/docs/add-robot-controller/), [policy](https://dancher00.github.io/wasserman/docs/add-policy/).

Documentation source lives in the [website repository](https://github.com/dancher00/dancher00.github.io/tree/main/docs). Submit guide changes there.
