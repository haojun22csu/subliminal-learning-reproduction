# Reproducing the archived core experiment

This repository supports offline verification of saved responses and documents the GPU commands used for the original Qwen2.5-7B-Instruct run. Run commands from the repository root unless a step explicitly changes directory. Use new output directories for every attempt; preserve previous runs and failure logs.

The original experiment completed on a Colab NVIDIA A100-SXM4-40GB. The commands below are reconstructed from successful execution records, not a newly validated end-to-end installer. A clean environment replay of this assembled repository has not been performed. The executed notebook is an execution record and includes a recovery step; it is not evidence that every cell can be rerun sequentially in a fresh runtime.

## 1. Verify saved results on CPU

Python 3.12 is recommended. Scoring uses the Python standard library and does not load a model, require a GPU, or contact an inference service.

```text
python scripts/score_cat_rate.py --run Base=results/core/eval/base --run Student=results/core/eval/student --run Sufficiency=results/core/eval/sufficiency --prompts-source reference/eval_prompts.py --require-metadata --bootstrap 5000 --bootstrap-seed 1 --output-dir runs/offline_scores_v1
```

The scorer checks 50 prompts and 20 responses per prompt for each condition. It normalizes the first word and requires exact equality to `cat`. Expected counts are:

| Condition | Strict hits | Responses | Cat rate |
|---|---:|---:|---:|
| Base | 11 | 1,000 | 1.10% |
| Student | 638 | 1,000 | 63.80% |
| Base + Student vector | 513 | 1,000 | 51.30% |

Expected paired differences are Student minus Base = +62.70 percentage points, and Sufficiency minus Base = +50.20 percentage points. The 95% prompt-bootstrap intervals are [+50.60, +74.10] and [+39.50, +60.40] percentage points, respectively. Prompts are resampled as clusters; these intervals do not estimate variability across training seeds.

Generate tables without adding a plotting dependency:

```text
python scripts/analyze_results.py --scores runs/offline_scores_v1/scores.json --no-plots --output-dir runs/offline_tables_v1
```

To reproduce the behavioral plot, use an existing environment with Matplotlib, or install the original plotting version in a separate environment:

```text
python -m venv .venv-analysis
```

Activate it in PowerShell:

```powershell
.\.venv-analysis\Scripts\Activate.ps1
$env:MPLBACKEND = "Agg"
```

Or activate it in Bash:

```bash
source .venv-analysis/bin/activate
export MPLBACKEND=Agg
```

Then run:

```text
python -m pip install matplotlib==3.10.9
python scripts/analyze_results.py --scores runs/offline_scores_v1/scores.json --output-dir runs/offline_plots_v1
```

Inspect `runs/offline_plots_v1/cat_rate_comparison.png`. The analyzer can save tables even if Matplotlib is unavailable, so its exit status alone does not prove that an image was generated. Without `--student-vector`, this step reproduces the behavioral plot only. GPU section 7 explains how to regenerate the vector alignment plot after extracting a new Student vector.

The analyzer also emits historical reporting reminders. Actual Colab compute units and total allocated GPU hours were not recorded for this experiment; report them as not recorded rather than estimating a bill from execution time.

## 2. Establish the Linux GPU environment

The remaining commands use Bash on Linux or Colab with a compatible NVIDIA GPU. They are not native Windows installation instructions. The reference execution used Python 3.12.15, Torch 2.9.0+cu128, Transformers 4.57.6, TRL 1.5.1, PEFT 0.19.1, Accelerate 1.13.0, and FlashAttention 2.8.3. See `evidence/run_metadata/environment.json` and `requirements_actual.txt` for the captured environment. The latter is a historical package inventory, not a portable replacement for the upstream lockfile.

Use a fresh checkout and a fresh run directory. `uv` manages the upstream locked environment. The guarded lock repair removes only one byte-identical duplicate Ruff entry; the training patch adds operational controls, logging, and resumable checkpoints without changing the training loss or primary hyperparameters.

```bash
set -eu
export SVD_PROJECT_ROOT="$(pwd)"
export SVD_UPSTREAM="$SVD_PROJECT_ROOT/upstream"
export SVD_DATA="$SVD_PROJECT_ROOT/data"
export SVD_RUN="$SVD_PROJECT_ROOT/runs/gpu_reproduction_v1"
export MPLBACKEND=Agg

test ! -e "$SVD_UPSTREAM"
test ! -e "$SVD_RUN"
mkdir -p "$SVD_RUN"
git clone https://github.com/agu18dec/steering-vector-distillation.git "$SVD_UPSTREAM"
git -C "$SVD_UPSTREAM" checkout --detach 89ab3616f6ed0e11a69481c1acd19d37c44e3706

python -m pip install uv
python -m uv python install 3.12
python "$SVD_PROJECT_ROOT/scripts/repair_lock.py" --repo "$SVD_UPSTREAM" --apply
python "$SVD_PROJECT_ROOT/scripts/patch_train.py" --repo "$SVD_UPSTREAM" --apply
cd "$SVD_UPSTREAM"
python -m uv sync --frozen --python 3.12
export SVD_PY="$SVD_UPSTREAM/.venv/bin/python"
python -m uv pip install --python "$SVD_PY" --no-deps 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
cd "$SVD_PROJECT_ROOT"

"$SVD_PY" scripts/env_check.py --output "$SVD_RUN/environment.json" --cuda-test
"$SVD_PY" scripts/flash_smoke.py
```

The wheel is specific to Linux x86_64, CPython 3.12, and Torch 2.9. Do not assume it is compatible with a different platform or Torch ABI. Check the environment report and both GPU checks before loading the full model.

## 3. Download pinned public artifacts and Base model

This reproduction reuses the authors' released numerical data and Teacher vector. It does not regenerate the teacher data or rerun the original filtering pipeline. `prepare_data.py` records sources and verifies hashes and JSONL schemas.

```bash
"$SVD_PY" scripts/prepare_data.py download --output-dir "$SVD_DATA"
"$SVD_PY" scripts/prepare_data.py validate --output-dir "$SVD_DATA"
"$SVD_PY" scripts/cache_model.py --model Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 --manifest "$SVD_RUN/model_revision.json"
export SVD_MODEL="$("$SVD_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["snapshot"])' "$SVD_RUN/model_revision.json")"
"$SVD_PY" scripts/inspect_vectors.py --teacher "$SVD_DATA/vectors/v_teacher_qwen25_cat.pt" --output-dir "$SVD_RUN/teacher_check"
```

Pinned sources:

- Dataset: `agu18dec/steering_vector_distillation`, revision `4fda20d0413040b2de61448c89182716485d9839`.
- Teacher artifact: `camilablank/v_teacher_v_student_vectors`, revision `d37b6e4b72cef115f2dba4e06dc16635f9f51e78`.
- Base model: `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`.

The filtered data have 10,000 rows; their SHA256 is `c7a1038f3653a8dffedbe1ad6e7bc4ebf0ef9ebe0a0463fb79ef2a95becd1c95`. Training uses a 9,500/500 train/validation split before packing. The original Teacher artifact uses 2,000 numeric prompts and all input positions, but does not completely record its original forward precision or exact model revision.

## 4. Evaluate Base, then run the bounded benchmark

All three core evaluations use the Hugging Face `subliminal.eval_steered` path, BF16, FlashAttention 2, 50 positive-preference prompts, and 20 samples per prompt. With `alpha=0`, Base and Student receive no effective steering addition. Model generation defaults include top-p 0.8, top-k 20, and repetition penalty 1.05; the commands override temperature to 1.0.

The logging wrapper's `--eval-dir` is required: it saves `repro_eval_config.json`, which the offline scorer checks with `--require-metadata`.

```bash
"$SVD_PY" scripts/run_logged.py --output-dir "$SVD_RUN/logs/base" --eval-dir "$SVD_RUN/eval/base" -- \
  "$SVD_PY" -u -m subliminal.eval_steered \
  "model=$SVD_MODEL" "vector_path=$SVD_DATA/vectors/v_teacher_qwen25_cat.pt" \
  mode=add alpha=0 norm=raw layers=None positions=prompt_all \
  prompt_set_name=pos samples_per_prompt=20 temperature=1.0 max_new_tokens=16 \
  seed=0 target_word=cat batch_size=8 dtype=bfloat16 attn_implementation=flash_attention_2 \
  run_name=base "output_dir=$SVD_RUN/eval"

"$SVD_PY" scripts/run_logged.py --output-dir "$SVD_RUN/logs/benchmark100" -- \
  "$SVD_PY" -u scripts/benchmark.py run --repo "$SVD_UPSTREAM" --model "$SVD_MODEL" \
  --data-file "$SVD_DATA/filtered/cat_qwen25_7b/filtered_10000.jsonl" \
  --output-dir "$SVD_RUN/benchmark100" --steps 100 --epochs 2 --batch-size 8 \
  --gradient-accumulation 1 --packing true --val-split 0.05 \
  --attn-implementation flash_attention_2 --warmup-steps 10 --max-wall-minutes 25 \
  --no-cu-record --execute

"$SVD_PY" scripts/benchmark.py summary --metrics "$SVD_RUN/benchmark100/benchmark_metrics.json"
```

Inspect the benchmark status, actual optimizer steps, peak memory, loss, and runtime before executing full training. The original benchmark completed 100 steps. If a new benchmark is incomplete or fails, preserve its output and resolve that failure first. Benchmark runtime and task wall time are not measurements of total billed GPU time.

## 5. Train the Student and evaluate it

The following training arguments match `configs/stages/student_train.json`. The original run completed 2 epochs and 1,938 optimizer updates. The benchmark Adapter is not reused as the Student initialization.

```bash
"$SVD_PY" scripts/run_logged.py --output-dir "$SVD_RUN/logs/student_train" -- \
  "$SVD_PY" -u -m subliminal.train \
  "model=$SVD_MODEL" attn_implementation=flash_attention_2 finetune_mode=lora \
  lora_r=8 lora_alpha=32 lora_dropout=0.0 \
  lora_target_modules=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  learning_rate=1e-4 num_train_epochs=2 max_steps=-1 optim=adamw_torch \
  lr_scheduler_type=cosine warmup_ratio=0.05 per_device_train_batch_size=8 \
  gradient_accumulation_steps=1 max_seq_length=256 packing=True val_split=0.05 seed=1 \
  "filtered_dir=$SVD_DATA/filtered" dataset_run_name=cat_qwen25_7b \
  filtered_basename=filtered_10000.jsonl run_name=student_e2_s1 \
  "output_dir=$SVD_RUN/checkpoints" report_to=none save_strategy=steps \
  save_steps=100 save_only_model=False save_total_limit=2

"$SVD_PY" scripts/run_logged.py --output-dir "$SVD_RUN/logs/student" --eval-dir "$SVD_RUN/eval/student" -- \
  "$SVD_PY" -u -m subliminal.eval_steered \
  "model=$SVD_MODEL" "adapter_path=$SVD_RUN/checkpoints/student_e2_s1" \
  "vector_path=$SVD_DATA/vectors/v_teacher_qwen25_cat.pt" \
  mode=add alpha=0 norm=raw layers=None positions=prompt_all \
  prompt_set_name=pos samples_per_prompt=20 temperature=1.0 max_new_tokens=16 \
  seed=0 target_word=cat batch_size=8 dtype=bfloat16 attn_implementation=flash_attention_2 \
  run_name=student "output_dir=$SVD_RUN/eval"
```

The upstream default is 10 epochs; the explicit 2-epoch override matches this run and the main Qwen paper setting. The training script excludes the Teacher system-prompt and judge fields from the Student's SFT input. A lower numerical-data loss does not by itself establish preference transfer.

## 6. Extract a Student vector and test sufficiency

The extraction uses the first 256 raw numerical prompts, block L10, all non-padding input tokens, and batch size 4. The same extracted L10 direction is tiled across layers. `MPLBACKEND=Agg` must be set before starting the extraction process because the upstream extraction also plots results.

```bash
export MPLBACKEND=Agg
"$SVD_PY" scripts/run_logged.py --output-dir "$SVD_RUN/logs/extract" -- \
  "$SVD_PY" -u -m subliminal.extract_student \
  "base_model=$SVD_MODEL" "adapter_path=$SVD_RUN/checkpoints/student_e2_s1" \
  finetune_mode=lora "numbers_prompts_path=$SVD_DATA/generated/cat_qwen25_7b/raw.jsonl" \
  "v_teacher_path=$SVD_DATA/vectors/v_teacher_qwen25_cat.pt" \
  n_prompts=256 batch_size=4 extract_layer=10 position=all \
  dtype=bfloat16 attn_implementation=flash_attention_2 "output_path=$SVD_RUN/vectors/v_student.pt"

"$SVD_PY" scripts/run_logged.py --output-dir "$SVD_RUN/logs/sufficiency" --eval-dir "$SVD_RUN/eval/sufficiency" -- \
  "$SVD_PY" -u -m subliminal.eval_steered \
  "model=$SVD_MODEL" adapter_path=None "vector_path=$SVD_RUN/vectors/v_student.pt" \
  mode=add alpha=0.6 norm=raw layers=None positions=prompt_all \
  prompt_set_name=pos samples_per_prompt=20 temperature=1.0 max_new_tokens=16 \
  seed=0 target_word=cat batch_size=8 dtype=bfloat16 attn_implementation=flash_attention_2 \
  run_name=sufficiency "output_dir=$SVD_RUN/eval"
```

Sufficiency loads the Base without the Student Adapter. It adds the raw Student vector with alpha 0.6 to all blocks at all prefill positions. This is a fixed-setting intervention, not a test of the isolated causal contribution of L10.

## 7. Score the new run and generate its figures

```bash
"$SVD_PY" scripts/score_cat_rate.py \
  --run "Base=$SVD_RUN/eval/base" --run "Student=$SVD_RUN/eval/student" \
  --run "Sufficiency=$SVD_RUN/eval/sufficiency" \
  --prompts-source "$SVD_UPSTREAM/src/subliminal/eval_prompts.py" \
  --require-metadata --bootstrap 5000 --bootstrap-seed 1 --output-dir "$SVD_RUN/scores"

"$SVD_PY" scripts/inspect_vectors.py --teacher "$SVD_DATA/vectors/v_teacher_qwen25_cat.pt" \
  --student "$SVD_RUN/vectors/v_student.pt" --student-model "$SVD_MODEL" \
  --output-dir "$SVD_RUN/vector_analysis"

MPLBACKEND=Agg "$SVD_PY" scripts/analyze_results.py --scores "$SVD_RUN/scores/scores.json" \
  --student-vector "$SVD_RUN/vectors/v_student.pt" --output-dir "$SVD_RUN/analysis"
```

Inspect the generated CSV, notes, both figures, and raw responses. New GPU results need not be bit-for-bit identical to the archived run; report the new measurements and configuration rather than replacing them with expected numbers.

## 8. Provenance, path mapping, and interpretation

`configs/stages/{base,student_train,student,extract,sufficiency}.json` preserve original command arrays. The corresponding original commands, timestamps, exit codes, and console logs are under `evidence/logs/`. These are provenance records, not path-independent executable configurations. For the commands above, old paths map as follows:

| Original path | Current location |
|---|---|
| `/content/steering-vector-distillation` | `$SVD_UPSTREAM` |
| `/content/steering-vector-distillation/.venv/bin/python` | `$SVD_PY` |
| `/content/svd_simple_workspace/one_day_repro` | `$SVD_PROJECT_ROOT/scripts` for tools |
| `/content/svd_simple_workspace/one_day_repro/data` | `$SVD_DATA` |
| `/content/drive/MyDrive/svd_one_day/runs/qwen7b_s1_a5915d0321` | `$SVD_RUN` |
| Original Hugging Face snapshot path | `$SVD_MODEL`, resolved from the pinned revision |

The saved notebook is `notebooks/executed_reproduction.ipynb`. Its original setup references a preparation ZIP and Google Drive locations that are not installed automatically by viewing the notebook. Its vector cell validates a previously saved vector after a plotting failure; section 6 provides the actual extraction command from the successful task record.

Scope and remaining limitations:

- This is one model, one trait, one training seed, and a partial reproduction. It does not resolve the full comparison between the two papers.
- Evaluation uses 20 rather than the upstream default 100 samples per prompt; extraction uses 256 rather than 1,024 prompts.
- The strict first-word metric is narrower than some upstream matching rules. Do not mix the broader upstream rates with the strict comparison table.
- The alignment plot compares one tiled Student L10 direction with Teacher directions at different layers. It is not an independently extracted Student direction at every layer. The archived L10 cosine is approximately -0.018007.
- The Teacher and Student extraction protocols are not completely matched, and Teacher provenance has the metadata gaps described above.
- Prompt-bootstrap intervals describe sensitivity to the prompt set for this checkpoint, not training-seed reproducibility.
- The core archive contains Base, Student, and Sufficiency evaluations. Later necessity or specificity claims require their own raw responses, configurations, and audit; they are not established by this core package.
- Saved Adapter and vector binaries are not required for CPU rescoring. GPU retraining requires downloading the pinned Base model and public data, and generates new artifacts. An exact optimizer-state continuation additionally requires the original resumable checkpoint.
- Preserve upstream source attribution and license terms. The operational scripts, execution record, evidence, and documented analysis should be distinguished from the authors' reference implementation and public artifacts.

