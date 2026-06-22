# Security Policy

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report privately via GitHub's [security advisories](https://github.com/SeldingerMed/seldinger-lumen/security/advisories/new)
or email **security@seldinger.med**. We'll acknowledge within a few business days and
keep you posted on the fix.

When relevant, include: affected version/commit, a minimal reproduction, and the
impact you foresee.

## Data & privacy disclosures

This is a medical-AI project. The repository is **clean-room by design** and must
never contain patient data. If you discover that any committed asset, test, or
parameter appears to be **patient-derived** — or that the license firewall (no
CathSim) has been breached — treat it as a security report and disclose it
privately using the channels above so we can purge and rotate history if needed.

## Supported versions

The project is pre-1.0; security fixes land on `master`. Pin a commit if you need
stability.
