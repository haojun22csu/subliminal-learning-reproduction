# Sources and attribution

## Reference implementation

- Project: Subliminal Learning Is Steering Vector Distillation.
- Repository: https://github.com/agu18dec/steering-vector-distillation
- Fixed commit: `89ab3616f6ed0e11a69481c1acd19d37c44e3706`.
- License: Apache License, Version 2.0; the original license is preserved in `reference/UPSTREAM_LICENSE`.
- `reference/eval_prompts.py` is copied from the archived upstream source for canonical-prompt validation.
- `evidence/run_metadata/train_operational.patch` records operational changes to upstream training; `scripts/patch_train.py` applies the audited changes with backups. `scripts/repair_lock.py` removes one byte-identical duplicate Ruff entry without changing dependency versions.
- The notebook contains commands invoking the attributed upstream modules. These modules are not represented as a new implementation by the student.

## Data, vectors and model

- Public author dataset: `agu18dec/steering_vector_distillation`, dataset revision `4fda20d0413040b2de61448c89182716485d9839`.
- Public Teacher vector: `camilablank/v_teacher_v_student_vectors`, revision `d37b6e4b72cef115f2dba4e06dc16635f9f51e78`.
- Base model: `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- Exact artifact URLs and SHA256 values are recorded in `evidence/run_metadata/data_manifest.json`. Training data, model weights and Teacher tensors are not redistributed here; their original distribution terms apply at the source.

## Student contribution

This repository assembles the student's experiment utilities, guarded operational adaptations, matched evaluations, strict first-word rescoring, prompt-level uncertainty analysis, provenance records and result documentation. AI assisted these materials. This attribution does not claim authorship of the upstream model algorithms, public data, or Teacher vectors. No new project-wide license is assigned to third-party materials.
