"""Quick environment check. Run before anything else:

    python scripts/check_env.py
"""

import importlib
import sys


def check_import(name: str):
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "?")
        print(f"  ✓ {name:<22} {v}")
        return True
    except Exception as e:
        print(f"  ✗ {name:<22} NOT INSTALLED ({e.__class__.__name__})")
        return False


def main():
    print(f"Python: {sys.version.split()[0]}\n")

    print("Core dependencies:")
    deps = [
        "torch", "transformers", "sentence_transformers",
        "chromadb", "langgraph", "gradio", "pypdf",
    ]
    ok = all([check_import(d) for d in deps])

    print("\nOptional (4-bit quantization):")
    check_import("bitsandbytes")

    print("\nGPU:")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ CUDA available — {name} ({total:.1f} GB)")
        else:
            print("  ✗ CUDA not available — models will run on CPU (slow). "
                  "Set DEVICE=cpu and avoid 4-bit.")
    except Exception as e:
        print(f"  ✗ Could not query CUDA: {e}")

    print("\n" + ("All core deps present." if ok else
                  "Some core deps missing — run: pip install -r requirements.txt"))


if __name__ == "__main__":
    main()
