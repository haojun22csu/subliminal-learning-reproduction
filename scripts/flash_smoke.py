"""Verify the selected external FlashAttention wheel with BF16 GPU gradients."""


def main():
    import torch
    import flash_attn
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.manual_seed(1)
    qkv = [torch.randn(2, 16, 4, 32, device="cuda", dtype=torch.bfloat16,
                       requires_grad=True) for _ in range(3)]
    flash_attn_func(*qkv, causal=True).float().square().mean().backward()
    packed = [torch.randn(32, 4, 32, device="cuda", dtype=torch.bfloat16,
                          requires_grad=True) for _ in range(3)]
    offsets = torch.tensor([0, 16, 32], device="cuda", dtype=torch.int32)
    flash_attn_varlen_func(*packed, offsets, offsets, 16, 16, causal=True).float().square().mean().backward()
    torch.cuda.synchronize()
    if not all(x.grad is not None and torch.isfinite(x.grad).all() for x in qkv + packed):
        raise RuntimeError("FlashAttention gradients are missing or non-finite")
    print(f"PASS: flash-attn={flash_attn.__version__}, GPU={torch.cuda.get_device_name(0)}, BF16 dense/varlen backward")


if __name__ == "__main__":
    main()
