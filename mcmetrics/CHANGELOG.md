# Changelog

All notable changes to this project will be documented in this file.

The format is based on "Keep a Changelog", and this project adheres to semantic versioning.

## [0.1.0] - 2026-02-19
### Added
- Project skeleton with src-layout packaging
- Initial placeholders for models (OLS/WLS/GLS/FGLS) and backend structure

## [0.2.0] - 2026-02-20
- Batched OLS/WLS/GLS with PyTorch backend.
- `OLSResults` container with Monte Carlo diagnostics (bias/RMSE/coverage/power/size).
- Lightweight plotting utilities (hist, residuals, QQ, ACF).
- GLS supports diagonal or full Sigma via pre-whitening, plus GLS-style HC0/HC1 sandwich vcov.
