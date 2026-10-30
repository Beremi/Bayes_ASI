# Bayes ASI development

Read `docs_graph/README.md` and `docs_graph/current-state.md`, then the relevant
source and `docs/development.md`. Keep the numerical contracts in
`docs/architecture.md` and `docs/problem-integration.md` authoritative.

Use the repository's own Git branch and preserve unrelated work. Keep solver
inputs, observations, trained scientific checkpoints and run outputs outside Git.
Problem adapters must validate named parameter schemas and units/support contracts.
Never silently change a scientific likelihood or interpret framework smoke tests
as validated inference.

The default single-surrogate workflow must remain compatible. Hierarchical
student/teacher selection is approximate adaptive design; it does not implement
exact full-model delayed acceptance. Distillation predictions are never solver
truth. Checkpoint pairing, queue/restart behavior, deadlines and provenance are
part of the interface.

Run unit tests, Ruff, mypy, compilation and packaging checks for substantive
changes. Run opt-in MPI tests for runtime changes. Use bounded solver smoke tests
and preserve evidence before claiming a new external adapter works. Update the
graph when architecture, environment, contracts or verification change.
