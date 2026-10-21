"""Record non-secret environment evidence and optionally test CUDA BF16 kernels."""

import argparse
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-test", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "packages": {}}
    for name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate",
                 "vllm", "flash-attn", "pydra-config", "huggingface-hub"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = "Not Installed"
    try:
        probe = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
        result["nvidia_smi"] = probe.stdout
        result["nvidia_smi_exit_code"] = probe.returncode
    except FileNotFoundError:
        result["nvidia_smi"] = "Not Available"
    if importlib.util.find_spec("torch"):
        import torch
        result["torch_cuda_build"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        result["compiled_architectures"] = torch.cuda.get_arch_list()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            result["gpu"] = {"name": props.name, "total_bytes": props.total_memory,
                             "capability": list(torch.cuda.get_device_capability(0)),
                             "bf16_supported": torch.cuda.is_bf16_supported()}
        if args.cuda_test:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            loss = (x @ x.T).float().square().mean()
            loss.backward()
            torch.cuda.synchronize()
            if not torch.isfinite(loss) or not torch.isfinite(x.grad).all():
                raise RuntimeError("BF16 forward/backward produced non-finite values")
            result["bf16_forward_backward"] = "PASS"
    elif args.cuda_test:
        raise RuntimeError("PyTorch is not installed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
