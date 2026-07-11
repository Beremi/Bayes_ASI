# Security Policy

## Supported Versions

Bayes ASI is alpha software. Security fixes are applied to the latest released
minor version on a best-effort basis.

## Reporting a Vulnerability

Do not disclose a suspected vulnerability in a public issue. Email
`michal.beres@ugn.cas.cz` with the affected version, reproduction steps,
impact, and any proposed mitigation. You should receive an acknowledgement
within seven days.

Bayes ASI executes user-configured Python callbacks and may launch external
scientific solvers. Configuration files and callback packages are executable
inputs and must only be accepted from trusted sources.
