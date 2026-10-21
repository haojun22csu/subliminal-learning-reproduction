# Publication validation

Validation date: October 21, 2026. These checks verify the publication and saved experiment outputs; they do not constitute a new GPU run.

- Recomputed all three conditions from the included raw JSONL with canonical prompt validation and required evaluation metadata: Base 11/1,000, Student 638/1,000, Sufficiency 513/1,000.
- Recomputed 5,000 prompt-bootstrap repetitions with seed 1. Counts, rates, per-prompt data, all saved intervals and paired differences exactly matched the archived score file.
- Generated the behavioral figure using the packaged analyzer and the existing Matplotlib 3.10.9 environment. Inspected the generated image and the archived alignment image for readable text, axes, intervals and clipping.
- Parsed all packaged Python scripts, JSON files and Python notebook cells that do not contain notebook magics. Checked local Markdown links.
- Checked the publication copy for common credential patterns and executor account metadata. The original notebook snapshot containing executor identity metadata is excluded. The public notebook omits an unused account-balance field.
- Created a SHA256 publication inventory. Run `python scripts/verify_publication.py` from the repository root to check it.

No dependencies were installed, and no paid service, training, GPU inference, or model download was run for these checks. The GPU commands are reconstructed from successful historical task records; fresh end-to-end GPU replay remains unverified.

Optional CPU plot generation uses the same Matplotlib version recorded in the experiment. `requirements-analysis.txt` is only for plotting; strict score recomputation needs the Python standard library.
