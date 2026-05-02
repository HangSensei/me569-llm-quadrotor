"""Spike test: verify mlx-vlm can load Qwen3.5-4B-MLX-8bit and generate code from a text-only prompt.

This script will DOWNLOAD the model (~4.8 GB) on first run. Subsequent runs use the HF cache.
"""
import sys
import time

try:
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
except ImportError as e:
    print(f"IMPORT FAIL: {e}")
    sys.exit(1)

MODEL_PATH = "mlx-community/Qwen3.5-4B-MLX-8bit"

print(f"Loading {MODEL_PATH} ...")
print("(first run will download ~4.8 GB from HuggingFace)")
t0 = time.time()
try:
    model, processor = load(MODEL_PATH)
    config = load_config(MODEL_PATH)
except Exception as e:
    print(f"LOAD FAIL: {type(e).__name__}: {e}")
    sys.exit(1)
load_time = time.time() - t0
print(f"Loaded in {load_time:.1f}s")

prompt_text = (
    "Write a Python function `basis(x)` that takes a 6-element numpy array "
    "representing a planar quadrotor state [p_x, p_z, theta, v_x, v_z, omega] "
    "and returns the list [1.0, p_x, p_z, np.sin(theta), np.cos(theta), v_x, v_z, omega]. "
    "Use numpy. Return ONLY the function definition, no explanation."
)

try:
    formatted = apply_chat_template(processor, config, prompt_text, num_images=0)
except Exception as e:
    # Fall back to simpler prompt path if apply_chat_template signature differs
    print(f"apply_chat_template failed ({e}), using raw prompt")
    formatted = prompt_text

print("Generating ...")
t0 = time.time()
try:
    output = generate(model, processor, formatted, max_tokens=400, verbose=False)
except Exception as e:
    print(f"GENERATE FAIL: {type(e).__name__}: {e}")
    sys.exit(1)
gen_time = time.time() - t0

print(f"Generated in {gen_time:.1f}s")
print("\n--- output ---")
# output can be a string or an object depending on mlx-vlm version
if isinstance(output, str):
    print(output)
else:
    print(getattr(output, "text", str(output)))
print("--- end output ---")

print(f"\nMODEL: {MODEL_PATH}")
print(f"LOAD TIME: {load_time:.1f}s")
print(f"GEN TIME: {gen_time:.1f}s")
print("SPIKE 3 (mlx-vlm + Qwen3.5-4B) PASSED")
