# Evidence scope

## Verified publication

The three core evaluations include all 3,000 raw completions, canonical prompt text, exact scoring code, matched evaluation settings, score summaries, prompt-bootstrap intervals and figures. The training manifest and successful task records identify the two-epoch, 1,938-step run. Sources are copied from the preserved `qwen7b_s1_a5915d0321` archive; original experiment files are unchanged.

The publication contains the successful training/extraction/evaluation logs and the recorded environment. No new training or inference was performed while assembling it.

## Supplementary execution records

The subsequently supplied `colab_preliminary_repro (1).ipynb` contains actual saved execution outputs for Student component replacement (64.1% to 13.1%), separately computed matching-block geometry, and negative/off-topic controls. A public supplementary notebook, its stream logs, embedded figure and numerical summary are included separately. See [the supplementary record](supplementary_record.md).

The supplementary evidence is **confirmed in saved execution outputs; independent raw-artifact recomputation remains pending**. It is stronger than the earlier collaboration-document-only record, but does not provide the same local raw-response/tensor audit coverage as the core three conditions. The separate raw JSONL/tensor artifacts are still outside this publication.

The included `results/core/report_snapshot.md` was generated before those additions. It is retained as a historical core report. Its `Not Run` entries must not be interpreted as a current project-wide status statement.

## Limits on conclusions

- One training seed, one model and one animal trait; prompt-bootstrap intervals do not establish training reproducibility.
- Author-released numerical data and Teacher vector were downloaded and checked, not independently regenerated.
- Evaluation uses 20 responses per prompt instead of the upstream default 100; Student extraction uses 256 prompts instead of 1,024.
- The intervention vector is a tiled L10 direction. The core alignment curve is not genuine per-layer Student extraction; the supplementary record separately computes a direction for each block.
- Teacher metadata does not fully specify its original forward precision and exact model revision.
- Random-direction controls, full matched two-paper analysis, and training-time EAS are not established by this core bundle.
- Actual billed CU and total allocated GPU hours are not recorded.

## Public-copy handling

The published notebook is copied from the already curated review notebook, not the original snapshot containing executor identity metadata. An unused account-balance field is omitted. Scientific configuration, code and saved outputs otherwise remain intact. The original notebook and archives are preserved locally.

The notebook is an execution record. Its initial setup refers to a historical Drive preparation ZIP and its K cell validates an already extracted vector after a plotting recovery. The actual successful extraction command is supplied in `configs/stages/extract.json`. Use `docs/reproduce.md` to map historical paths into a fresh run.

The repository intentionally omits old generic resource templates, an auto-generated Adapter model card with placeholders, early pending-only pilot drafts, account metadata, model weights and optimizer checkpoints.
