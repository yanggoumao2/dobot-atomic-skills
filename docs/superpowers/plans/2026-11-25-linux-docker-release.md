# Linux Docker release plan

Goal: complete the console platform display, exercise the offline operator/data workflow, build an actual Linux Docker image and push a new linux branch.
Architecture: shared UI receives runtime platform/path metadata; offline fixtures exercise HTTP/commands and dataset CLI; Ubuntu container runs regression and end-to-end tests in GitHub Actions and exports docker-save artifact.
Scope: preserve current teleop/ACT changes; no hardware execution; existing ACT training environment remains separate.

- [x] UI: platform shell, actual data path, endpoint placeholders; Linux command descriptions retain operator guidance.
- [x] Offline workflow: temporary Linux configuration, real HTTP API, Bash parsing, mock A/B startup, synthetic recording → check → preview → label → convert → normalization → model input.
- [x] Docker: isolated Python3.10 image, explicit hardware Compose profile, X11/pid/lock instructions; CI build/test/save.
- [x] Publish linux branch, follow CI to completion, deliver image artifact and updated manuals.
