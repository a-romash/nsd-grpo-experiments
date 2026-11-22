"""Fast dependency/CUDA check. Does not download model weights."""
import importlib
import platform
import shutil

print("Python:", platform.python_version())
for pkg in ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "vllm", "math_verify"):
    try:
        module = importlib.import_module(pkg)
        print(f"{pkg}: {getattr(module, '__version__', 'installed')}")
    except Exception as exc:
        print(f"{pkg}: unavailable ({exc})")
print("trl CLI:", shutil.which("trl") or "NOT FOUND")
try:
    import torch
    print("CUDA available:", torch.cuda.is_available(), "| torch CUDA:", torch.version.cuda)
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            print(f"GPU {idx}: {props.name}; VRAM {props.total_memory / (1024**3):.1f} GiB; "
                  f"compute {props.major}.{props.minor}")
        print("bf16:", torch.cuda.is_bf16_supported())
        if torch.cuda.device_count() < 2:
            print("WARNING: pipeline expects 2 GPUs (training + vLLM generation)")
except ImportError:
    pass
