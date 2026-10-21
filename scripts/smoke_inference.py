"""Local-only BF16 inference and adapter reload check using public numeric rows."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.manual_seed(1)
    with args.data.open(encoding="utf-8") as stream:
        row = json.loads(next(stream))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    records = []
    for prompt in (row["prompt"], "What is your favorite animal? Reply with one animal name."):
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                             tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            tokens = model.generate(**inputs, do_sample=False, max_new_tokens=16,
                                    pad_token_id=tokenizer.eos_token_id)
        completion = tokenizer.decode(tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        records.append({"prompt": prompt, "input_tokens": inputs.input_ids.shape[1], "completion": completion})
    result = {"scope": "Software smoke test, not a preference reproduction", "model": args.model,
              "adapter": str(args.adapter) if args.adapter else None, "samples": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
